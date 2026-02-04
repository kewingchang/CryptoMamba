import pandas as pd
import numpy as np
import argparse
import sys

def process_interactions(input_file, output_file):
    print(f"Reading data from {input_file}...")
    try:
        df = pd.read_csv(input_file)
        # 【新增这行代码】将 Date 列转换为 datetime 对象，修复 .dt 访问器错误
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
    except FileNotFoundError:
        print(f"Error: File {input_file} not found.")
        sys.exit(1)
    
    # ==========================================
    # 1. 严格的列检查函数
    # ==========================================
    def get_series_strict(col_name):
        """
        获取列数据，如果不存在则直接抛出异常。
        不进行 0 填充，只进行前向填充(ffill)以处理周末/节假日产生的空洞。
        """
        if col_name not in df.columns:
            # 特殊处理：Log_Volume 如果没有，看有没有 Volume
            if col_name == 'Log_Volume' and 'Volume' in df.columns:
                print(f"Note: 'Log_Volume' not found, calculating from 'Volume'...")
                return np.log(df['Volume'] + 1)
            
            # 特殊处理：fng_value 和 fng_norm 的兼容
            if col_name == 'fng_value' and 'fng_norm' in df.columns:
                return df['fng_norm'] * 100
                
            raise ValueError(f"CRITICAL ERROR: Column '{col_name}' is missing from the input CSV. Cannot generate interaction features.")
        
        return df[col_name].ffill()

    # ==========================================
    # 2. 物理意义中心化 (Domain Centering)
    # 禁止使用 min(), max(), mean() 以防止数据泄露
    # ==========================================
    def get_centered_state(col_name, center_constant):
        """
        获取并中心化特征。
        例如 RSI (0~100) -> Center 50 -> (-50 ~ +50)
        这样做的目的是让 交叉特征 携带 方向性 (正=强/贪婪, 负=弱/恐慌)
        """
        series = get_series_strict(col_name)
        return series - center_constant

    print("Generating interaction features with STRICT validation...")

    # 创建输出 DataFrame，首先复制固定特征
    # 这里也进行严格检查，如果固定特征缺失，说明数据集本身有问题
    fixed_features = [
        'Date', 'Open', 'High', 'Low', 'Close', 'Volume', 
        'vol_current', 'day_of_year', 'vol_weekend'
    ]
    
    # 检查固定特征是否存在
    missing_fixed = [c for c in fixed_features if c not in df.columns]
    if missing_fixed:
        raise ValueError(f"CRITICAL ERROR: The following FIXED features are missing: {missing_fixed}")

    df_out = df[fixed_features].copy()

    # 获取核心驱动因子 (Drivers) - 代表“能量”
    driver_vol = get_series_strict('vol_current')
    driver_log_vol = get_series_strict('Log_Volume')
    
    # ADX 代表趋势强度，我们不中心化它，只用它做乘数
    # 如果 ADX 缺失，这行会报错停止
    try:
        driver_adx = get_series_strict('ADX_14') / 50.0 # 归一化到 0~2 左右，保持无量纲
    except ValueError:
        # ADX 不是必须的，如果你的数据集里确实没有，可以注释掉相关交叉
        print("Warning: ADX_14 not found. Skipping Trend interactions.")
        driver_adx = None

    # ==========================================
    # Group 1: 波动率交互 (Volatility x State)
    # 逻辑：波动率 x (指标 - 阈值)。正值=向上爆发，负值=向下崩盘
    # ==========================================
    
    # RSI 交互 (中心点 50)
    # 意义：RSI>50且高波动 -> 强势上涨动能；RSI<50且高波动 -> 恐慌抛售动能
    rsi_state = get_centered_state('RSI_14', 50.0)
    df_out['inter_vol_rsi'] = driver_vol * rsi_state

    # MVRV 交互 (中心点 1.0 - 盈亏平衡线)
    # 意义：高估值(>1)且高波动 -> 逃顶信号；低估值(<1)且高波动 -> 抄底信号
    # 注意：这里不除以标准差，保留原始物理意义
    mvrv_state = get_centered_state('bitbo_mvrv', 1.0)
    df_out['inter_vol_mvrv'] = driver_vol * mvrv_state

    # 情绪交互 (中心点 50)
    # 兼容 fng_value (0-100)
    fng_state = get_centered_state('fng_value', 50.0) 
    df_out['inter_vol_fng'] = driver_vol * fng_state

    # 布林带位置交互 (中心点 0.5)
    # volatility_bbp: 1.0=上轨, 0.0=下轨
    bbp_state = get_centered_state('volatility_bbp', 0.5)
    df_out['inter_vol_bbp'] = driver_vol * bbp_state

    # ==========================================
    # Group 2: 成交量交互 (Volume x State)
    # 逻辑：量价配合，放量上涨 vs 放量下跌
    # ==========================================

    # 资金流向 (Price Change * Volume)
    # 这是一个非常基础且强的特征，如果features.csv没有，这里生成
    # 使用 Close - Open 代表实体方向
    price_change_pct = (df['Close'] - df['Open']) / df['Open']
    df_out['inter_volqty_price_change'] = driver_log_vol * price_change_pct

    # 情绪放量 (FearGreed * Volume)
    # 极度贪婪/恐慌时的成交量权重更高
    # 这里取绝对值，因为“极度恐慌放量”和“极度贪婪放量”都是变盘信号
    # (FNG - 50)^2 * Volume
    fng_intensity = np.square((fng_state / 50.0)) # 归一化到 0~1 的平方
    df_out['inter_volqty_fng_intensity'] = driver_log_vol * fng_intensity

    # ==========================================
    # Group 3: 趋势感知交互 (Trend x Momentum)
    # 逻辑：在趋势强时，RSI/MACD 更可靠；震荡时忽略
    # ==========================================
    
    if driver_adx is not None:
        # 趋势加强版 RSI
        df_out['inter_trend_rsi'] = driver_adx * rsi_state
        
        # 趋势加强版 资金流
        # 强趋势下的放量上涨权重更高
        df_out['inter_trend_moneyflow'] = driver_adx * df_out['inter_volqty_price_change']

    # ==========================================
    # Group 4: 价格回归交互 (Reversion)
    # ==========================================
    
    # 价格偏离度交互
    # 如果有 bitbo_ma200d，计算偏离度 * 波动率
    # 如果偏离很大且波动率很高，意味着可能要剧烈回归或加速
    try:
        ma200 = get_series_strict('bitbo_ma200d')
        # Log distance: >0 price above MA, <0 price below MA
        dist_ma200 = np.log(df['Close'] / ma200)
        df_out['inter_dist_ma200_vol'] = dist_ma200 * driver_vol
    except ValueError:
        print("Skipping MA200 interaction (feature missing).")

    # ==========================================
    # Group 5: 宏观/链上交互
    # ==========================================
    
    # ETH 市占率交互 (如果有)
    try:
        eth_dom = get_series_strict('Coingecko_eth_dominance')
        # Dominance 变化率 * 波动率 -> 捕捉 ETH 独立行情
        # 使用 diff() 是安全的，不泄露未来
        dom_change = eth_dom.diff().fillna(0)
        df_out['inter_eth_dom_change_vol'] = dom_change * driver_vol
    except ValueError:
        print("Skipping ETH Dominance interaction (feature missing).")

    # ==========================================
    # Group 6: 动量指标交互 (Momentum Interactions)
    # 逻辑：结合不同动量指标捕捉多时间框架一致性
    # ==========================================
    
    try:
        # 6.1 Vol * CCI (商品通道波动)
        # CCI > 100 超买， < -100 超卖，高波动时放大信号
        cci_state = get_centered_state('CCI_14', 0.0)
        df_out['inter_vol_cci'] = driver_vol * (cci_state / 100.0)  # 归一化到 -1 ~ +1
        
        # 6.2 Vol * CMO (钱德动量波动)
        # CMO 类似RSI但更敏感，中心化后捕捉极端动量
        cmo_state = get_centered_state('CMO_14', 0.0)
        df_out['inter_vol_cmo'] = driver_vol * (cmo_state / 50.0)  # 归一化到 -1 ~ +1
        
        # 6.3 Volume * MOM (动量成交量)
        # 价格动量 * 成交量，确认动量可持续性
        mom = get_series_strict('MOM_10')
        mom_rel = mom / df['Close']  # 相对动量
        df_out['inter_volqty_mom'] = driver_log_vol * mom_rel
        
        # 6.4 Trend * CCI (趋势加权CCI)
        if driver_adx is not None:
            df_out['inter_trend_cci'] = driver_adx * (cci_state / 100.0)
    except ValueError as e:
        print(f"Skipping Momentum interactions: {e}")

    # ==========================================
    # Group 7: 宏观经济交互 (Macro Interactions)
    # 逻辑：宏观因子与市场指标的耦合，捕捉全球风险偏好
    # ==========================================
    
    try:
        # 7.1 Vol * VIX (波动-恐慌指数交互)
        # VIX高时ETH波动放大，风险资产联动
        vix = get_series_strict('VIX')
        df_out['inter_vol_vix'] = driver_vol * (vix / 20.0)  # 归一化VIX (20为中性)
        
        # 7.2 Vol * CPI (通胀波动)
        # 高通胀期波动率上升，ETH作为通胀对冲的敏感度
        cpi = get_series_strict('CPI')
        cpi_state = get_centered_state('CPI', 2.0)  # 假设2%中性通胀
        df_out['inter_vol_cpi'] = driver_vol * (cpi_state / 1.0)  # 归一化
        
        # 7.3 DXY * NUPL (美元强势下的未实现盈亏压力)
        dxy = get_series_strict('DX_Y.NYB')
        nupl = get_series_strict('bitbo_nupl')
        df_out['inter_dxy_nupl'] = (dxy / 100.0) * nupl  # DXY归一化
    except ValueError as e:
        print(f"Skipping Macro Interactions interactions: {e}")

    # ==========================================
    # Group 8: 链上高级交互 (On-Chain Advanced)
    # 逻辑：链上指标与价格行为的深度融合
    # ==========================================
    
    try:
        # 8.1 Vol * Puell (矿工抛压波动)
        # Puell高时波动放大，矿工套现风险
        puell_state = get_centered_state('bitbo_puell_multiple', 1.0)
        df_out['inter_vol_puell_adv'] = driver_vol * (puell_state / 2.0)  # 归一化
        
        # 8.2 Volume * CDD (成交量与币天销毁)
        # 高量下高CDD表示老币移动，潜在顶部
        cdd = get_series_strict('bitbo_cdd')
        df_out['inter_volqty_cdd'] = driver_log_vol * np.log(cdd + 1)  # log稳定
        
        # 8.3 MVRV * OBV (估值与资金流向)
        # 高MVRV下OBV下降表示派发
        obv = get_series_strict('OBV')
        obv_norm = obv / obv.rolling(14).std()  # 滚动标准化，避免泄露
        df_out['inter_mvrv_obv'] = mvrv_state * obv_norm
        
        # 8.4 NUPL * Hash Rate (盈亏与网络安全)
        hash_rate = get_series_strict('bitbo_hash_rate_value')
        df_out['inter_nupl_hash'] = nupl * np.log(hash_rate + 1)
    except ValueError as e:
        print(f"Skipping On-Chain Advanced interactions: {e}")

    # ==========================================
    # Group 9: 时间与季节交互 (Temporal Interactions)
    # 逻辑：季节性与市场状态的结合
    # ==========================================  
    try:
        # 9.1 Vol * Month Sin (季节波动)
        month_sin = get_series_strict('month_sin')
        df_out['inter_vol_month_sin'] = driver_vol * month_sin
        
        # 9.2 Halving Countdown * RSI (减半前动量)
        halving_cd = get_series_strict('halving_countdown_30')
        df_out['inter_halving_rsi'] = halving_cd * rsi_state
        
        # 9.3 Day of Year * Volume (年度周期成交)
        doy = get_series_strict('day_of_year')
        doy_cycle = np.sin(2 * np.pi * doy / 365)  # 循环编码 sin
        df_out['inter_doy_volqty'] = doy_cycle * driver_log_vol
    except ValueError as e:
        print(f"Skipping Temporal interactions: {e}")

    # ==========================================
    # Group 10: 移动平均交互 (MA Interactions)
    # 逻辑：价格相对MA的位置与动态指标
    # ==========================================
    
    try:
        # 10.1 Vol * EMA Distance (指数MA偏离波动)
        ema100 = get_series_strict('EMA_100')
        dist_ema = np.log(df['Close'] / ema100)
        df_out['inter_vol_ema_dist'] = driver_vol * dist_ema
        
        # 10.2 Volume * SMA Distance (简单MA成交)
        sma100 = get_series_strict('SMA_100')
        dist_sma = np.log(df['Close'] / sma100)
        df_out['inter_volqty_sma_dist'] = driver_log_vol * dist_sma
        
        # 10.3 Trend * Aroon Osc (趋势与Aroon振荡)
        if driver_adx is not None:
            aroon_osc = get_series_strict('AROON_osc_14')
            df_out['inter_trend_aroon'] = driver_adx * (aroon_osc / 100.0)
    except ValueError as e:
        print(f"Skipping MA interactions: {e}")

    # ==========================================
    # Group 11: 蜡烛与模式交互 (Candle Pattern Interactions)
    # 逻辑：K线形态在特定上下文下的权重
    # ==========================================
    try:
        # 11.1 Vol * Doji (十字星波动)
        # Doji在高波动时表示犹豫，转折概率高
        doji = get_series_strict('CDLDOJI')
        df_out['inter_vol_doji'] = driver_vol * doji  # Doji通常为1/0
        
        # 11.2 Volume * Engulfing (吞没成交)
        engulf = get_series_strict('CDLENGULFING')
        df_out['inter_volqty_engulf'] = driver_log_vol * engulf
        
        # 11.3 Weekend * Hammer (周末锤子线)
        is_weekend = get_series_strict('is_weekend')
        hammer = get_series_strict('CDLHAMMER')
        df_out['inter_weekend_hammer'] = is_weekend * hammer
    except ValueError as e:
        print(f"Skipping Candle Pattern interactions: {e}")

    # ==========================================
    # Group 12: 资金流交互 (Money Flow Interactions)
    # 逻辑：资金流入/流出与关键指标
    # ==========================================
    
    try:
        # 12.1 Vol * CMF (柴金资金流波动)
        cmf = get_series_strict('CMF')
        df_out['inter_vol_cmf'] = driver_vol * cmf
        
        # 12.2 Volume * AD (A/D线成交)
        ad = get_series_strict('AD')
        ad_norm = ad / ad.rolling(14).std()  # 滚动标准化
        df_out['inter_volqty_ad'] = driver_log_vol * ad_norm
        
        # 12.3 MVRV * BOP (估值与力量平衡)
        bop = get_series_strict('BOP')
        df_out['inter_mvrv_bop'] = mvrv_state * bop
    except ValueError as e:
        print(f"Skipping Money Flow interactions: {e}")

    # ==========================================
    # Group 13: 链上资金流交互 (On-Chain Flow Interactions)
    # 逻辑：链上资金流动与价格行为的耦合
    # ==========================================
    
    try:
        # 13.1 MVRV * 长期持有者供应变化
        # 高估值时LTH供应下降 = 派发风险；低估值时LTH供应上升 = 积累
        lth_supply = get_series_strict('bitbo_lth_supply')
        lth_supply_change = lth_supply.diff().fillna(0) / lth_supply.rolling(14).mean()
        df_out['inter_mvrv_lth_flow'] = mvrv_state * lth_supply_change
        
        # 13.2 SOPR * 波动率
        # 盈利卖出比例在高波动时更具信息性
        sopr = get_series_strict('bitbo_sopr_7d_ma')
        df_out['inter_sopr_vol'] = (sopr - 1.0) * driver_vol  # SOPR-1为中心化
        
        # 13.3 NUPL * 哈希率变化
        # 盈利状态下算力增长 = 基本面支撑；盈利状态下算力下降 = 矿工投降风险
        hash_rate = get_series_strict('bitbo_hash_rate_value')
        hash_change = hash_rate.pct_change(30).fillna(0)
        df_out['inter_nupl_hash_change'] = nupl * hash_change
        
        # 13.4 CDD * 价格动量
        # 老币移动时若有强动量 = 顶部信号；若弱动量 = 可能是换手
        price_mom = df['Close'].pct_change(10).fillna(0)
        cdd = get_series_strict('bitbo_cdd')
        df_out['inter_cdd_momentum'] = np.log(cdd + 1) * price_mom
    except ValueError as e:
        print(f"Skipping On-Chain Flow interactions: {e}")

    # ==========================================
    # Group 14: 技术指标背离交互 (Divergence Interactions)
    # 逻辑：捕捉价格与技术指标之间的背离信号
    # ==========================================
    
    try:
        # 14.1 RSI-价格背离 (高低点背离)
        # 价格新高但RSI未新高 = 顶背离；价格新低但RSI未新低 = 底背离
        rsi = get_series_strict('RSI_14')
        price_high_14 = df['High'].rolling(14).max()
        rsi_high_14 = rsi.rolling(14).max()
        price_low_14 = df['Low'].rolling(14).min()
        rsi_low_14 = rsi.rolling(14).min()
        
        # 顶背离：价格新高但RSI未创新高
        top_div = ((df['High'] == price_high_14) & (rsi < rsi_high_14)).astype(float)
        # 底背离：价格新低但RSI未创新低
        bottom_div = ((df['Low'] == price_low_14) & (rsi > rsi_low_14)).astype(float)
        
        df_out['inter_divergence_rsi_top'] = top_div * driver_vol  # 顶背离时波动率权重
        df_out['inter_divergence_rsi_bottom'] = bottom_div * driver_vol
        
        # 14.2 成交量-价格背离
        # 价格新高但成交量未新高 = 量价背离
        volume_high_14 = df['Volume'].rolling(14).max()
        volume_div = ((df['High'] == price_high_14) & (df['Volume'] < volume_high_14)).astype(float)
        df_out['inter_divergence_volume'] = volume_div * driver_log_vol
        
        # 14.3 MACD-价格背离 (用MACD柱状图)
        try:
            macd_hist = get_series_strict('MACD_Hist')
            macd_high_14 = macd_hist.rolling(14).max()
            macd_low_14 = macd_hist.rolling(14).min()
            
            macd_top_div = ((df['High'] == price_high_14) & (macd_hist < macd_high_14)).astype(float)
            macd_bottom_div = ((df['Low'] == price_low_14) & (macd_hist > macd_low_14)).astype(float)
            
            df_out['inter_divergence_macd_top'] = macd_top_div * driver_vol
            df_out['inter_divergence_macd_bottom'] = macd_bottom_div * driver_vol
        except:
            print("Skipping MACD divergence (feature missing).")
    except ValueError as e:
        print(f"Skipping Divergence interactions: {e}")

    # ==========================================
    # Group 15: 市场广度交互 (Market Breadth Interactions)
    # 逻辑：衡量整体市场健康度的交互
    # ==========================================
    
    try:
        # 15.1 涨跌家数 * 波动率
        # 上涨家数多且波动率高 = 健康上涨；下跌家数多且波动率高 = 系统性风险
        advance_decline_net = get_series_strict('CDD_advance-decline_net')
        df_out['inter_breadth_vol'] = advance_decline_net * driver_vol
        
        # 15.2 上涨百分比 * ETH市占率
        # ETH市占率上升且市场广度好 = 山寨季；市占率下降但广度好 = BTC主导
        advance_pct = get_series_strict('CDD_advance-decline_advance_percent')
        eth_dom = get_series_strict('Coingecko_eth_dominance')
        df_out['inter_breadth_eth_dom'] = advance_pct * (eth_dom / 100.0)
        
        # 15.3 总交易数 * 价格变化
        # 链上活跃度与价格变化的相互作用
        total_tx = get_series_strict('CDD_total_transactions')
        tx_change = total_tx.pct_change(7).fillna(0)
        df_out['inter_tx_price'] = tx_change * price_change_pct
    except ValueError as e:
        print(f"Skipping Market Breadth interactions: {e}")

    # ==========================================
    # Group 16: 波动率结构交互 (Volatility Structure Interactions)
    # 逻辑：不同波动率指标之间的关系
    # ==========================================
    
    try:        
        # 16.1 ATR vs 布林带宽度
        # 真实波动范围与布林带宽度的比较
        atr = get_series_strict('ATR_14')
        bb_width = get_series_strict('volatility_bbw')
        df_out['inter_atr_bb_width'] = atr / (bb_width + 1e-6) * driver_vol
        
        # 16.2 日内波动 vs 日间波动
        # (High-Low) vs (Close-Open)的相对大小
        intraday_vol = (df['High'] - df['Low']) / df['Open']
        interday_vol = np.abs(df['Close'] - df['Open']) / df['Open']
        vol_structure = intraday_vol / (interday_vol + 1e-6)
        df_out['inter_intra_inter_vol'] = vol_structure * driver_log_vol
    except ValueError as e:
        print(f"Skipping Volatility Structure interactions: {e}")

    # ==========================================
    # Group 17: 季节性强化交互 (Enhanced Seasonal Interactions)
    # 逻辑：结合多个时间维度的季节性
    # ==========================================
    
    try:
        # 17.1 月度季节性 * 减半周期位置
        month_sin = get_series_strict('month_sin')
        halving_pos_sin = get_series_strict('halving_pos_sin')
        df_out['inter_month_halving'] = month_sin * halving_pos_sin * driver_vol
        
        # 17.2 周内效应 * 月末效应
        weekday = get_series_strict('weekday')
        is_month_end = ((df['Date'].dt.day == df['Date'].dt.days_in_month) | 
                       (df['Date'].dt.month != df['Date'].shift(-1).dt.month)).astype(float)
        df_out['inter_weekday_month_end'] = (weekday / 7.0) * is_month_end * driver_log_vol
        
        # 17.3 季度初效应 * 波动率
        # 季度初通常有资金流入，配合高波动可能是趋势起点
        is_quarter_start = (df['Date'].dt.month.isin([1,4,7,10]) & (df['Date'].dt.day == 1)).astype(float)
        df_out['inter_quarter_start_vol'] = is_quarter_start * driver_vol
    except ValueError as e:
        print(f"Skipping Enhanced Seasonal interactions: {e}")

    # ==========================================
    # Group 18: 相关性结构交互 (Correlation Structure Interactions)
    # 逻辑：资产间相关性变化的市场影响
    # ==========================================
    
    try:
        # 18.1 ETH/BTC相关性 * 波动率
        # 相关性下降且波动率高 = 资产分化；相关性上升且波动率高 = 系统性风险
        eth_btc_ratio = get_series_strict('ETHBTC')
        corr_change = eth_btc_ratio.pct_change(14).fillna(0)
        df_out['inter_ethbtc_corr_vol'] = corr_change * driver_vol
        
        # 18.2 BTC市占率 * 市场情绪
        # BTC市占率上升 + 贪婪 = 资金回流比特币；下降 + 贪婪 = 山寨季
        btc_dom = get_series_strict('Coingecko_btc_dominance')
        df_out['inter_btc_dom_fng'] = (btc_dom / 100.0) * fng_state
        
        # 18.3 黄金/比特币比率 * 宏观波动
        try:
            gold = get_series_strict('GC')
            gold_btc_ratio = gold / df['Close']
            ratio_change = gold_btc_ratio.pct_change(30).fillna(0)
            df_out['inter_gold_btc_macro'] = ratio_change * driver_vol
        except:
            print("Skipping Gold/BTC ratio (feature missing).")
    except ValueError as e:
        print(f"Skipping Correlation Structure interactions: {e}")

    # ==========================================
    # Group 19: 高阶动量交互 (Higher-Order Momentum Interactions)
    # 逻辑：动量的变化率（加速度）与市场状态
    # ==========================================
    
    try:
        # 19.1 RSI加速度 * 波动率
        rsi_accel = rsi.diff().diff().fillna(0)  # 二阶差分
        df_out['inter_rsi_accel_vol'] = rsi_accel * driver_vol
        
        # 19.2 成交量加速度 * 价格变化
        volume_accel = df['Volume'].diff().diff().fillna(0) / df['Volume'].rolling(14).mean()
        df_out['inter_volume_accel_price'] = volume_accel * price_change_pct
        
        # 19.3 MACD柱状图变化率 * 趋势强度
        if driver_adx is not None:
            try:
                macd_hist = get_series_strict('MACD_Hist')
                macd_hist_change = macd_hist.diff().fillna(0)
                df_out['inter_macd_change_trend'] = macd_hist_change * driver_adx
            except:
                print("Skipping MACD change interaction (feature missing).")
    except ValueError as e:
        print(f"Skipping Higher-Order Momentum interactions: {e}")

    # ==========================================
    # Group 20: 极端事件交互 (Extreme Event Interactions)
    # 逻辑：市场极端状态下的特殊交互
    # ==========================================
    
    try:
        # 20.1 恐慌/贪婪极端标志 * 波动率
        is_extreme_fear = get_series_strict('is_extreme_fear')
        is_extreme_greed = get_series_strict('is_extreme_greed')
        df_out['inter_extreme_fear_vol'] = is_extreme_fear * driver_vol
        df_out['inter_extreme_greed_vol'] = is_extreme_greed * driver_vol
        
        # 20.2 波动率冲击 * 成交量冲击
        # 波动率和成交量同时出现极端值
        vol_shock = (driver_vol / driver_vol.rolling(14).mean() - 1).clip(-2, 2)  # 限制在±2倍
        volume_shock = (driver_log_vol / driver_log_vol.rolling(14).mean() - 1).clip(-2, 2)
        df_out['inter_vol_volume_shock'] = vol_shock * volume_shock
        
        # 20.3 价格崩盘/暴涨标志 * 链上数据
        # 单日跌幅超过10%且CDD高 = 恐慌性抛售
        price_crash = (price_change_pct < -0.10).astype(float)
        price_surge = (price_change_pct > 0.10).astype(float)
        
        try:
            cdd = get_series_strict('bitbo_cdd')
            cdd_high = (cdd > cdd.rolling(14).quantile(0.8)).astype(float)
            df_out['inter_crash_cdd'] = price_crash * cdd_high * driver_vol
            df_out['inter_surge_cdd'] = price_surge * cdd_high * driver_vol
        except:
            print("Skipping CDD extreme interactions (feature missing).")
    except ValueError as e:
        print(f"Skipping Extreme Event interactions: {e}")

    # ==========================================
    # 3. 最终清理与保存
    # ==========================================
    
    # 再次填充可能因为计算产生的极少数 NaN (比如第一行 diff)
    # df_out.fillna(0, inplace=True)
    
    # 检查是否有无穷大 (除零导致)
    # if np.isinf(df_out.values).any():
    #     print("Warning: Infinity values detected. Replacing with 0.")
    #     df_out.replace([np.inf, -np.inf], 0, inplace=True)
    # 检查是否有无穷大 (除零导致)
    # 【修改说明】先筛选出数值类型的列，避开 Date (datetime) 列，防止报错
    numeric_cols = df_out.select_dtypes(include=[np.number]).columns
    
    if np.isinf(df_out[numeric_cols].values).any():
        print("Warning: Infinity values detected. Replacing with 0.")
        # 只对数值列进行替换，不影响 Date 列
        df_out[numeric_cols] = df_out[numeric_cols].replace([np.inf, -np.inf], 0)

    output_path = output_file
    if output_path is None:
        output_path = input_file
        # output_path = input_file.replace('.csv', '_interactions.csv')

    print(f"Successfully generated {len(df_out.columns)} features.")
    print(f"Saving to {output_path}...")
    df_out.to_csv(output_path, index=False)
    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate strict interaction features.')
    parser.add_argument('--filename', type=str, required=True, help='Input CSV file')
    args = parser.parse_args()

    process_interactions(args.filename, None)