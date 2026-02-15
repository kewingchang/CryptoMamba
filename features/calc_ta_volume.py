import argparse
import pandas as pd
import numpy as np
import talib 
import pandas_ta as pta 

def main():
    parser = argparse.ArgumentParser(description="Calculate Advanced Volume Features")
    parser.add_argument('--filename', type=str, required=True, help="Input CSV filename")
    args = parser.parse_args()
    filename = args.filename

    # 1. 读取数据
    try:
        df = pd.read_csv(filename)
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    # 确保有 Date 列并处理
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.set_index('Date').sort_index()
    else:
        # 如果没有 Date 列，尝试识别 index 是否已经是日期
        try:
            df.index = pd.to_datetime(df.index)
            df = df.sort_index()
        except:
            print("Warning: No 'Date' column found and index is not datetime. Assuming sequential data.")

    # 必需列检查
    req_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    for c in req_cols:
        if c not in df.columns:
            raise ValueError(f"Missing column: {c}")
        df[c] = df[c].astype(float)

    # 2. 防除零因子
    epsilon = 1e-9

    print("Generating Advanced Volume Features...")

    # ==========================================
    # 0. 预计算基础指标 (VWAP, OBV Raw)
    # ==========================================
    
    # 修复 Bug: 计算 Rolling VWAP (14天) 用于后续距离计算
    # 这里用 Typical Price * Volume 的加权平均
    tp = (df['High'] + df['Low'] + df['Close']) / 3
    tp_v = tp * df['Volume']
    # 也就是 sum(TP*V, 14) / sum(V, 14)
    df['VWAP_14'] = tp_v.rolling(14).sum() / (df['Volume'].rolling(14).sum() + epsilon)

    # 计算原始 OBV (用于衍生，本身不作为特征保存或最后删除)
    obv_raw = talib.OBV(df['Close'].values, df['Volume'].values)

    # ==========================================
    # 1. 多尺度窗口循环 (Multi-scale)
    # 建议覆盖：周线逻辑(7)，半月逻辑(14)，月线逻辑(30)
    # ==========================================
    windows = [7, 14, 30]

    for w in windows:
        suffix = f"_{w}"
        
        # --- A. 相对成交量 (Stationarity) ---
        vol_ma = df['Volume'].rolling(window=w).mean()
        vol_std = df['Volume'].rolling(window=w).std()

        # Vol_Ratio: 当日量 / 均量 (最强特征之一)
        df[f'Vol_Ratio{suffix}'] = df['Volume'] / (vol_ma + epsilon)
        
        # Vol_Z: 标准化成交量
        df[f'Vol_Z{suffix}'] = (df['Volume'] - vol_ma) / (vol_std + epsilon)

        # --- B. OBV 衍生特征 (Stationary OBV) ---
        # 替代你原来的 OBV_14。
        # 1. OBV Slope: OBV 在窗口 w 内的线性斜率，表示资金流入流出的速率
        df[f'OBV_Slope{suffix}'] = talib.LINEARREG_SLOPE(obv_raw, timeperiod=w)
        # 归一化 Slope: 除以成交量均值，消除绝对值膨胀的影响
        df[f'OBV_Slope_Norm{suffix}'] = df[f'OBV_Slope{suffix}'] / (vol_ma + epsilon)

        # 2. OBV Oscillator: 乖离率 (OBV - MA(OBV)) / MA(OBV) (如果 OBV 有负数，分母不建议用 OBV，改用 vol_ma * 累计系数)
        # 简单版：OBV 这里的绝对数值没意义，重要的是相对变化。
        # 我们用 (OBV - MA(OBV)) / (Window * MeanVolume) 来近似归一化
        obv_ma = talib.SMA(obv_raw, timeperiod=w)
        df[f'OBV_Osc{suffix}'] = (obv_raw - obv_ma) / (vol_ma * w + epsilon)

        # --- C. 量价动量 (Momentum) ---
        # Force Index (强力指数)
        # 原始公式: V * (Close - Prev_Close)
        fi_raw = df['Volume'] * df['Close'].diff()
        # 平滑处理
        fi_ema = talib.EMA(fi_raw.fillna(0), timeperiod=w)
        # 归一化: 非常重要，否则价格翻倍后 FI 也会翻倍
        close_ma = df['Close'].rolling(window=w).mean()
        df[f'Force_Index{suffix}'] = fi_ema / (vol_ma * close_ma + epsilon) * 10000

        # --- D. 资金流向 (Flow) ---
        # MFI (Money Flow Index) - 自带归一化 (0-100)
        df[f'MFI{suffix}'] = talib.MFI(df['High'], df['Low'], df['Close'], df['Volume'], timeperiod=w)

        # EOM (Ease of Movement)
        # [修复点] 兼容 Series 和 DataFrame 返回值
        eom_res = pta.eom(df['High'], df['Low'], df['Close'], df['Volume'], length=w)
        if eom_res is not None:
            if isinstance(eom_res, pd.DataFrame):
                # 如果是 DataFrame，取第一列
                df[f'EOM{suffix}'] = eom_res.iloc[:, 0]
            else:
                # 如果是 Series，直接使用
                df[f'EOM{suffix}'] = eom_res
        
        # --- E. 波动率归一化成交量 ---
        # 逻辑：在波动率低的时候，少量成交量就能拉升，价值不同
        # Garman-Klass Volatility (用于归一化)
        log_hl = np.log(df['High'] / df['Low'])
        log_co = np.log(df['Close'] / df['Open'])
        gk_vol = np.sqrt(0.5 * log_hl**2 - (2 * np.log(2) - 1) * log_co**2)
        # 滚动平均波动率
        gk_vol_ma = gk_vol.rolling(w).mean()
        # Vol / Volatility: 如果这个值很高，说明成交量很大但波动很小(吸筹/出货)，或者反之
        df[f'Vol_to_Volat{suffix}'] = df[f'Vol_Ratio{suffix}'] / (gk_vol_ma * 100 + epsilon)

    # ==========================================
    # 2. 单一特征 (不需要多尺度或固定窗口)
    # ==========================================
    
    # 距离 VWAP 的距离 (使用之前计算的 VWAP_14 作为基准，或者用更长周期的)
    # 这里我们保留一个短周期的和一个长周期的
    df['Dist_VWAP_14'] = (df['Close'] - df['VWAP_14']) / (df['VWAP_14'] + epsilon)
    
    # 简单 ROC
    df['Vol_ROC'] = df['Volume'].pct_change()
    
    # Log Return Volatility (Price Feature, but useful context)
    df['Log_Ret'] = np.log(df['Close'] / df['Close'].shift(1))
    df['Price_Vol_14'] = df['Log_Ret'].rolling(14).std()
    df['Log_Volume'] = np.log(df['Volume'] + 1)

    # ==========================================
    # 3. 清理与保存
    # ==========================================
    
    # 恢复 index 为列以便保存
    df = df.reset_index()

    df.to_csv(filename, index=False)
    print(f"Successfully added features to {filename}")
    print(f"Features added for windows: {windows}")

if __name__ == "__main__":
    main()