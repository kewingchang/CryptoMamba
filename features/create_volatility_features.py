import pandas as pd
import numpy as np
import argparse
import sys
import warnings

# 忽略计算中的一些运行时警告（如对数0）
warnings.filterwarnings('ignore')

def process_volatility(input_file, output_file):
    print(f"Reading data from {input_file}...")
    try:
        df = pd.read_csv(input_file)
    except FileNotFoundError:
        print(f"Error: File {input_file} not found.")
        sys.exit(1)

    # ==========================================
    # 1. 基础设置
    # ==========================================
    # 必须保留的固定特征
    fixed_features = [
        'Date', 'Open', 'High', 'Low', 'Close', 'Volume', 
        'vol_current', 'day_of_year', 'vol_weekend'
    ]
    
    # 检查固定特征
    missing_fixed = [c for c in fixed_features if c not in df.columns]
    if missing_fixed:
        raise ValueError(f"CRITICAL ERROR: fixed features is missing.")

    # 初始化输出 DataFrame
    df_out = df[[c for c in fixed_features if c in df.columns]].copy()

    # 辅助函数：严格获取列
    def get_series_strict(col_name):
        if col_name not in df.columns:
            return None # 返回 None 表示该特征不存在，跳过计算
        return df[col_name].ffill()

    # ==========================================
    # 2. 波动率计算逻辑
    # ==========================================
    window_size = 14 # 保持与 vol_current 一致

    # A. 对数收益波动率 (适用于价格、量、算力等)
    # StdDev(Ln(t / t-1))
    def calc_log_ret_vol(col_name, output_name):
        s = get_series_strict(col_name)
        if s is None:
            print(f"Skipping {output_name}: Source column '{col_name}' not found.")
            return
        
        # 防止除零和负数
        s = s.replace(0, np.nan).ffill()
        
        # 计算对数收益
        log_ret = np.log(s / s.shift(1))
        
        # 计算滚动标准差
        vol = log_ret.rolling(window=window_size).std(ddof=0)
        
        df_out[output_name] = vol

    # B. 差分波动率 (适用于比率、震荡指标、包含负数的数据)
    # StdDev(t - t-1)
    def calc_diff_vol(col_name, output_name):
        s = get_series_strict(col_name)
        if s is None:
            print(f"Skipping {output_name}: Source column '{col_name}' not found.")
            return
        
        # 计算一阶差分
        diff = s.diff()
        
        # 计算滚动标准差
        vol = diff.rolling(window=window_size).std(ddof=0)
        
        df_out[output_name] = vol

    print("Generating extended volatility features...")

    # ==========================================
    # 3. 定义特征及其计算方式
    # ==========================================
    
    # --- Group 1: 价格类 (Log Return Vol) ---
    # High 和 Low 的波动率可能与 Close 不同，High vol 代表冲高回落的剧烈程度
    calc_log_ret_vol('High', 'vol_high')
    calc_log_ret_vol('Low', 'vol_low')
    calc_log_ret_vol('Open', 'vol_open')
    
    # --- Group 2: 成交量类 (Log Return Vol) ---
    # 成交量的波动率 = 流动性不稳定性。
    # 如果成交量忽大忽小，说明市场分歧巨大。
    calc_log_ret_vol('Volume', 'vol_volume') 

    # --- Group 3: 链上算力与活跃度 (Log Return Vol) ---
    # 算力波动率：矿工是否在频繁开关机？通常对应价格大底。
    calc_log_ret_vol('bitbo_hash_rate_value', 'vol_hashrate')
    # calc_diff_vol('bitbo_sharpe_ratio', 'vol_sharpe_diff')
    # calc_log_ret_vol('bitbo_sharpe_ratio', 'vol_sharpe_log')
    
    # --- Group 4: 链上估值比率 (Diff Vol) ---
    # MVRV 的剧烈抖动通常意味着多空激烈博弈
    calc_diff_vol('bitbo_mvrv', 'vol_mvrv')
    
    # NUPL (未实现净盈亏) 波动率：盈利盘的不稳定性
    calc_diff_vol('bitbo_nupl', 'vol_nupl')
    
    # SOPR (支出产出利润率) 波动率
    calc_diff_vol('bitbo_sopr_7d_ma', 'vol_sopr')

    # --- Group 5: 情绪与技术指标 (Diff Vol) ---
    # 情绪指数的波动率：人心是否不稳？
    # 兼容 fng_value 或 fng_norm
    if 'fng_value' in df.columns:
        calc_diff_vol('fng_value', 'vol_fng')
    elif 'fng_norm' in df.columns:
        calc_diff_vol('fng_norm', 'vol_fng')

    # RSI 的波动率：动量的稳定性。
    # RSI 平稳上升是牛市，RSI 剧烈震荡是盘整。
    calc_diff_vol('RSI_14', 'vol_rsi')

    # --- Group 6: 宏观指标 (Log Ret / Diff Vol) ---
    # 美元指数波动率
    calc_log_ret_vol('DX_Y.NYB', 'vol_dxy')
    
    # 标普500波动率 (相当于 VIX 的 VIX)
    calc_log_ret_vol('GSPC', 'vol_spx')
    
    # 利率波动率 (使用差分，因为利率可以为0或很小)
    calc_diff_vol('Fed_Funds_Rate_D', 'vol_fed_rate')

# --- Group 7: 更多价格与移动平均 (Log Return Vol) ---
    # BTC 价格波动率：作为 ETH 的基准，捕捉 BTC 主导下的 ETH 联动
    calc_log_ret_vol('bitbo_price', 'vol_btc')
    
    # 市场总值波动率：整体市值不稳表示生态风险
    calc_log_ret_vol('marketCap', 'vol_marketcap')
    
    # 200 日均线波动率：长期趋势的稳定性
    calc_log_ret_vol('bitbo_ma200d', 'vol_ma200')
    
    # 已实现价格波动率：全网成本线的抖动，表示平均持有成本变化
    calc_log_ret_vol('bitbo_realized_price', 'vol_realized')
    
    # LTH 已实现价格波动率：长期持有者成本稳定性
    calc_log_ret_vol('bitbo_lth_realized_price', 'vol_lth_realized')

    # --- Group 8: 动量震荡指标 (Diff Vol) ---
    # ADX 波动率：趋势强度的不稳定性，高 vol 表示趋势转折
    calc_diff_vol('ADX_14', 'vol_adx')
    
    # MACD 波动率：动量信号的噪音水平
    calc_diff_vol('MACD', 'vol_macd')
    
    # Stoch K 波动率：超买超卖信号的敏感度
    calc_diff_vol('Stoch_K', 'vol_stoch_k')
    
    # CCI 波动率：商品通道的极端偏差稳定性
    calc_diff_vol('CCI_14', 'vol_cci')
    
    # CMO 波动率：钱德动量的敏感度
    calc_diff_vol('CMO_14', 'vol_cmo')
    
    # Williams %R 波动率：超买超卖的极端性
    calc_diff_vol('WILLR_14', 'vol_willr')
    
    # MFI 波动率：资金流 RSI 的稳定性
    calc_diff_vol('MFI_14', 'vol_mfi')
    
    # ROC 波动率：变化率百分比的动量抖动
    calc_diff_vol('ROC_10', 'vol_roc')

    # --- Group 9: 链上比率指标 (Diff Vol) ---
    # Mayer Multiple 波动率：价格相对 MA 的估值波动
    calc_diff_vol('bitbo_mayer_multiple', 'vol_mayer')
    
    # Puell Multiple 波动率：矿工收入的相对水平不稳
    calc_diff_vol('bitbo_puell_multiple', 'vol_puell')
    
    # NVT 波动率：网络价值交易比的估值噪音
    calc_diff_vol('bitbo_nvt', 'vol_nvt')
    
    # MVRV Z-Score 波动率：标准化估值的极端偏差
    calc_diff_vol('bitbo_mvrv_z_score', 'vol_mvrv_z')
    
    # 1年 HODL Wave 波动率：长期持有比例的变化速度
    calc_diff_vol('bitbo_1yr_hodl', 'vol_hodl')

    # --- Group 10: 主导率与情绪扩展 (Diff Vol) ---
    # BTC 主导率波动率：BTC 强势下的 ETH 压力
    calc_diff_vol('Coingecko_btc_dominance', 'vol_btc_dom')
    
    # ETH 主导率波动率：山寨季信号的强度
    calc_diff_vol('Coingecko_eth_dominance', 'vol_eth_dom')

    # --- Group 11: 波动率之波动率与其他 (Diff Vol) ---
    # VIX 波动率：恐慌指数的 meta 波动，捕捉二级风险
    calc_diff_vol('VIX', 'vol_vix')
    
    # ATR 波动率：真实范围的稳定性，vol of vol
    calc_diff_vol('ATR_14', 'vol_atr')

    # --- Group 12: 商品宏观扩展 (Log Return Vol) ---
    # 原油价格波动率：能源风险对风险资产的影响
    calc_log_ret_vol('CL', 'vol_crude')
    
    # 黄金价格波动率：避险资产的反向信号
    calc_log_ret_vol('GC', 'vol_gold')
    
    # 白银价格波动率：工业金属的周期敏感
    calc_log_ret_vol('SI', 'vol_silver')

    # ==========================================
    # Group 13: 价格衍生波动率扩展 (Extended Price Volatility)
    # ==========================================
    
    # 13.1 ETH/BTC比率波动率 - 捕捉两种加密货币相对强度的变化
    calc_log_ret_vol('ETHBTC', 'vol_eth_btc_ratio')
    
    # 13.2 典型价格波动率 (Typical Price = (High+Low+Close)/3)
    calc_log_ret_vol('TYPPRICE', 'vol_typical_price')
    
    # 13.3 加权收盘价波动率 (Weighted Close = (High+Low+2*Close)/4)
    calc_log_ret_vol('WCLPRICE', 'vol_weighted_close')
    
    # 13.4 平均价格波动率 (Average Price = (Open+High+Low+Close)/4)
    calc_log_ret_vol('AVGPRICE', 'vol_average_price')
    
    # 13.5 中位价格波动率 (Median Price = (High+Low)/2)
    calc_log_ret_vol('MEDPRICE', 'vol_median_price')

    # ==========================================
    # Group 14: 价格比率波动率 (Price Ratio Volatility)
    # ==========================================
    
    # 14.1 实体占比波动率 (实体/影线比率的不稳定性)
    calc_diff_vol('spread_vol_ratio', 'vol_body_shadow_ratio')
    
    # 14.2 K线实体力度波动率
    calc_diff_vol('bar_strength', 'vol_bar_strength')
    
    # 14.3 收盘开盘比率波动率
    calc_diff_vol('Close_to_Open_Ratio', 'vol_close_open_ratio')
    
    # 14.4 收盘最高比率波动率
    calc_diff_vol('Close_to_High_Ratio', 'vol_close_high_ratio')
    
    # 14.5 高低价差波动率
    calc_diff_vol('High_Low_Spread', 'vol_hl_spread')

    # ==========================================
    # Group 15: 移动平均衍生波动率 (MA-Derived Volatility)
    # ==========================================
    
    # 15.1 21日简单移动平均波动率
    calc_log_ret_vol('SMA_21', 'vol_sma21')
    
    # 15.2 12日指数移动平均波动率
    calc_log_ret_vol('EMA_12', 'vol_ema12')
    
    # 15.3 21日指数移动平均波动率
    calc_log_ret_vol('EMA_21', 'vol_ema21')
    
    # 15.4 100日指数移动平均波动率
    calc_log_ret_vol('EMA_100', 'vol_ema100')
    
    # 15.5 100日简单移动平均波动率
    calc_log_ret_vol('SMA_100', 'vol_sma100')

    # ==========================================
    # Group 16: 波动率指标之波动率 (Vol of Vol)
    # ==========================================
    
    # 16.1 布林带宽度波动率 (Bollinger Band Width Volatility)
    calc_diff_vol('volatility_bbw', 'vol_bb_width')
    
    # 16.2 多席安通道宽度波动率 (Donchian Channel Width Volatility)
    calc_diff_vol('volatility_dcw', 'vol_dc_width')
    
    # 16.3 凯尔特纳通道宽度波动率 (Keltner Channel Width Volatility)
    calc_diff_vol('volatility_kcw', 'vol_kc_width')
    
    # 16.4 ATR波动率 (ATR of ATR - 真实范围的稳定性)
    calc_diff_vol('ATR_14', 'vol_atr_vol')
    
    # 16.5 布林带位置波动率 (%B的稳定性)
    calc_diff_vol('volatility_bbp', 'vol_bb_position')
    
    # 16.6 VIX波动率 (恐慌指数的波动)
    calc_diff_vol('VIX', 'vol_vix_vol')

    # ==========================================
    # Group 17: 链上高级波动率 (Advanced On-Chain Volatility)
    # ==========================================
    
    # 17.1 币天销毁波动率 (CDD Volatility)
    calc_diff_vol('bitbo_cdd', 'vol_cdd')
    
    # 17.2 价值天数销毁波动率 (VDD Volatility)
    calc_diff_vol('bitbo_vdd', 'vol_vdd')
    
    # 17.3 哈希丝带波动率
    calc_diff_vol('bitbo_hash_ribbon_30d_value', 'vol_hash_ribbon')
    
    # 17.4 短期持有者MVRV波动率
    calc_diff_vol('bitbo_sth_mvrv', 'vol_sth_mvrv')
    
    # 17.5 盈利供应量波动率
    calc_diff_vol('bitbo_supply_in_profit', 'vol_supply_profit')
    
    # ==========================================
    # Group 18: 市场广度与活跃度波动率 (Breadth & Activity Volatility)
    # ==========================================
    
    # 18.1 涨跌家数净额波动率
    calc_diff_vol('CDD_advance-decline_net', 'vol_advance_decline')
    
    # 18.2 总交易数波动率
    calc_diff_vol('CDD_total_transactions', 'vol_total_transactions')
    
    # 18.3 Gas使用量波动率
    calc_diff_vol('CDD_avg_gas_in_ETH_used_per_block', 'vol_gas_used')
    
    # 18.4 区块复杂度波动率
    calc_diff_vol('CDD_complexity_score', 'vol_complexity')

    # ==========================================
    # Group 19: 宏观经济波动率 (Macroeconomic Volatility)
    # ==========================================
    
    # 19.1 通胀波动率 (CPI Volatility)
    calc_diff_vol('CPI', 'vol_cpi')
    
    # 19.2 美元指数波动率
    calc_log_ret_vol('DX_Y.NYB', 'vol_dxy')
    
    # 19.3 标普500波动率
    calc_log_ret_vol('GSPC', 'vol_spx')
    
    # 19.4 纳斯达克波动率
    calc_log_ret_vol('IXIC', 'vol_nasdaq')
    
    # 19.5 谷歌搜索热度波动率
    calc_diff_vol('Google_Trends_Bitcoin', 'vol_google_btc')
    
    # 19.6 谷歌ETH搜索波动率
    calc_diff_vol('Google_Trends_Ethereum', 'vol_google_eth')

    # ==========================================
    # Group 20: 时间序列分解能量波动率 (TS Decomposition Energy Volatility)
    # 注意：这些特征有实际意义，捕捉不同频率成分的能量变化
    # ==========================================
    
    # 20.1 EMD高频分量能量波动率
    calc_diff_vol('emd_energy_imf1', 'vol_emd_high_freq')
    
    # 20.2 EMD中频分量能量波动率
    calc_diff_vol('emd_energy_imf2', 'vol_emd_mid_freq')
    
    # 20.3 小波高频细节能量波动率
    calc_diff_vol('wavelet_energy_detail1', 'vol_wavelet_high_freq')
    
    # 20.4 小波低频趋势能量波动率
    calc_diff_vol('wavelet_energy_approx', 'vol_wavelet_low_freq')
    
    # 20.5 EWT模式能量波动率
    calc_diff_vol('ewt_energy_mode1', 'vol_ewt_energy')
    
    # 20.6 傅里叶主频振幅波动率
    calc_diff_vol('fourier_amp_1', 'vol_fourier_amp')

    # ==========================================
    # Group 21: 衍生波动率指标 (Derived Volatility Measures)
    # ==========================================
    
    # 21.1 波动率曲面 (Volatility Surface)
    # 计算不同时间尺度波动率的标准差
    try:
        # 使用多个时间窗口
        windows = [7, 14, 21, 30]
        vol_series = []
        
        for w in windows:
            if 'Close' in df.columns:
                log_ret = np.log(df['Close'] / df['Close'].shift(1))
                vol = log_ret.rolling(window=w).std(ddof=0)
                vol_series.append(vol)
        
        if len(vol_series) >= 2:
            # 创建DataFrame以便计算行标准差
            vol_df = pd.DataFrame({f'vol_{w}d': vol_series[i] for i, w in enumerate(windows)})
            vol_surface = vol_df.std(axis=1)
            df_out['vol_surface'] = vol_surface.rolling(window=window_size).std(ddof=0)
    except Exception as e:
        print(f"Skipping volatility surface: {e}")

    # ==========================================
    # Group 22: 情绪衍生波动率 (Sentiment Derived Volatility)
    # ==========================================
    
    # 22.1 情绪加速度波动率
    calc_diff_vol('fng_accel', 'vol_fng_accel')
    
    # 22.2 情绪变化率波动率
    calc_diff_vol('fng_diff', 'vol_fng_diff')
    
    # 22.3 情绪7日波动率之波动率
    calc_diff_vol('fng_volatility_7d', 'vol_fng_vol_of_vol')
    
    # 22.4 情绪范围交互波动率
    calc_diff_vol('fng_range_interaction', 'vol_fng_range_interaction')
    
    # 22.5 背离标志波动率
    calc_diff_vol('divergence_flag', 'vol_divergence_flag')

    # ==========================================
    # Group 23: 链上持有者行为波动率 (On-Chain Holder Behavior Volatility)
    # ==========================================
    
    # 23.1 1年+HODL波浪波动率
    calc_diff_vol('bitbo_1yr_hodl', 'vol_1yr_hodl')
    
    # 23.2 1周HODL波浪波动率
    calc_diff_vol('bitbo_wave_2', 'vol_1w_hodl')
    
    # 23.3 1月HODL波浪波动率
    calc_diff_vol('bitbo_wave_3', 'vol_1m_hodl')
    
    # 23.4 长期持有者供应量波动率
    calc_diff_vol('bitbo_lth_supply', 'vol_lth_supply')
    
    # 23.5 彩虹图光束波动率
    calc_diff_vol('bitbo_beam', 'vol_rainbow_beam')

    # ==========================================
    # Group 24: 特殊波动率指标 (Special Volatility Measures)
    # ==========================================
    
    # 24.1 赫斯特指数波动率 (捕捉市场记忆性的变化)
    calc_diff_vol('hurst_exponent', 'vol_hurst')
    
    # 24.2 加尔曼-克拉斯波动率之波动率
    calc_diff_vol('gk_vol', 'vol_gk_vol')
    
    # 24.3 相对成交量波动率
    calc_diff_vol('rvol', 'vol_rvol')
    
    # 24.4 波动能量波动率
    calc_diff_vol('vol_force', 'vol_force_vol')
    
    # 24.5 恐慌波动率之波动率
    calc_diff_vol('fear_vol', 'vol_fear_vol')
    
    # 24.6 高频/低频能量比波动率
    calc_diff_vol('wavelet_energy_ratio', 'vol_wavelet_ratio')

    # ==========================================
    # 4. 最终清理与保存
    # ==========================================
    
    # 填充前 14 天产生的 NaN (用后面的值回填或填0)
    # df_out.fillna(0, inplace=True)
    
    # 处理无穷大
    # df_out.replace([np.inf, -np.inf], 0, inplace=True)

    output_path = output_file
    if output_path is None:
        output_path = input_file
        # output_path = input_file.replace('.csv', '_volatility_features.csv')

    print(f"Successfully generated {len(df_out.columns)} features.")
    print(f"Added features: {[c for c in df_out.columns if c not in fixed_features]}")
    print(f"Saving to {output_path}...")
    df_out.to_csv(output_path, index=False)
    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate extended volatility features.')
    parser.add_argument('--filename', type=str, required=True, help='Input CSV file')
    args = parser.parse_args()

    process_volatility(args.filename, None)