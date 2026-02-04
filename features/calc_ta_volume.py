import argparse
import pandas as pd
import talib  # pip install TA-Lib
import numpy as np
import pandas_ta as pta  # pip install pandas_ta


def apply_rolling_df(df, window_size, func):
    """Custom rolling apply to pass sub-DataFrame to func."""
    result = pd.Series(np.nan, index=df.index)
    # 修正循环范围，确保不发生 Index 越界
    for i in range(window_size - 1, len(df)):
        window = df.iloc[i - window_size + 1 : i + 1]
        result.iloc[i] = func(window)
    return result

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Calculate technical indicators and update CSV file.")
    parser.add_argument('--filename', type=str, required=True, help="The CSV filename to process.")
    args = parser.parse_args()

    filename = args.filename

    # Read the CSV file
    df = pd.read_csv(filename, parse_dates=['Date'])

    # Set index to Date and sort
    df = df.set_index('Date')
    df = df.sort_index()

    # Ensure required columns exist
    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # Convert columns to float
    df['Open'] = df['Open'].astype(np.float64)
    df['High'] = df['High'].astype(np.float64)
    df['Low'] = df['Low'].astype(np.float64)
    df['Close'] = df['Close'].astype(np.float64)
    df['Volume'] = df['Volume'].astype(np.float64)

    # Prevent Division by Zero errors
    epsilon = 1e-9 

    # --- 基础特征提取 ---
    
    # 1. Volatility Feature (vol_current)
    df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))
    df['vol_current'] = df['Log_Return'].rolling(window=14).std(ddof=0)
    df['vol_shock'] = df['vol_current'] / (df['vol_current'].rolling(30).mean() + epsilon)

    # 2. OBV (Original - Rolling Implementation)
    def calc_obv(window):
        # 严格使用窗口内数据
        return talib.OBV(window['Close'].values, window['Volume'].values)[-1]
    df['OBV'] = apply_rolling_df(df, 14, calc_obv)

    # 3. VWAP (Original Calculation)
    def calc_vwap(window):
        v = window['Volume'].values
        tp = (window['High'].values + window['Low'].values + window['Close'].values) / 3
        return np.sum(v * tp) / (np.sum(v) + epsilon)
    df['VWAP'] = apply_rolling_df(df, 14, calc_vwap)

    # --- NEW: 深度成交量特征工程 (无泄露版) ---

    print("Generating Advanced Volume Features (Leakage Free)...")

    vol_window = 14
    
    # === A. 相对成交量 (Stationarity) ===
    # 使用 rolling mean/std，严格限制在过去窗口
    vol_ma = df['Volume'].rolling(window=vol_window).mean()
    vol_std = df['Volume'].rolling(window=vol_window).std()

    # 1. Vol_Ratio
    df['Vol_Ratio'] = df['Volume'] / (vol_ma + epsilon)
    
    # 2. Vol_Z_Score
    df['Vol_Z'] = (df['Volume'] - vol_ma) / (vol_std + epsilon)

    # === B. 方向性成交量 (Directionality) ===
    price_change = df['Close'].diff()
    
    # 3. Signed Volume
    signed_vol = df['Volume'] * np.sign(price_change)
    df['Signed_Vol_Norm'] = signed_vol / (vol_ma + epsilon)

    # === C. 量价动量 (Momentum) ===
    
    # 4. Force Index (强力指数) - FIXED LEAKAGE
    fi_raw = df['Volume'] * price_change
    close_ma = df['Close'].rolling(window=vol_window).mean()
    
    # 计算 Force Index 的 EMA，然后除以 (平均成交量 * 平均价格) 进行无量纲化
    # 这是一个非常安全的相对指标
    fi_ema = talib.EMA(fi_raw.fillna(0), timeperiod=vol_window)
    df['Force_Index'] = fi_ema / (vol_ma * close_ma + epsilon) * 10000

    # 5. Volume ROC
    df['Vol_ROC'] = df['Volume'].pct_change()

    # === D. 资金流向 (Flow) ===
    # TA-Lib 和 Pandas-TA 的这些函数内部都是基于 rolling window 的，安全。

    # 6. MFI
    df['MFI'] = talib.MFI(df['High'], df['Low'], df['Close'], df['Volume'], timeperiod=vol_window)

    # 7. CMF
    df['CMF'] = df.ta.cmf(length=20)

    # 8. EOM
    df['EOM'] = df.ta.eom(length=14)

    df['rvol'] = df['Volume'] / df['Volume'].rolling(20).mean()

    # === E. 相对位置指标 ===

    # 9. Dist_VWAP
    # VWAP 之前是 rolling 算出来的，Close 是当前的，安全。
    df['Dist_VWAP'] = (df['Close'] - df['VWAP']) / (df['VWAP'] + epsilon)

    df['rvol'] = df['Volume'] / df['Volume'].rolling(20).mean()
    log_hl = np.log(df['High'] / df['Low'])
    log_co = np.log(df['Close'] / df['Open'])
    df['gk_vol'] = np.sqrt(0.5 * log_hl**2 - (2 * np.log(2) - 1) * log_co**2)

    # --- 其他已有特征 ---
    # Bitbo Sharpe Ratio (Rolling 365 -> Safe)
    rolling_returns = df['Log_Return'].rolling(window=365)
    annualized_return = rolling_returns.mean() * 365
    annualized_volatility = rolling_returns.std() * np.sqrt(365)
    df['sharpe_ratio'] = annualized_return / (annualized_volatility + epsilon)

    # Save
    df.to_csv(filename)
    print(f"Successfully added LEAKAGE-FREE advanced volume features to {filename}")

if __name__ == "__main__":
    main()