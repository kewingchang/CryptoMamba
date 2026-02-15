import argparse
import pandas as pd
import numpy as np
import sys

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', type=str, required=True)
    args = parser.parse_args()

    try:
        df = pd.read_csv(args.filename)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    # 确保 Date 存在且格式正确
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date')
    
    # 转换为数值
    cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # 计算收益率
    df['Return'] = df['Close'].pct_change()

    # 滚动窗口：通常夏普看 30天，90天或 1年
    # Bitbo 可能用 1年，但对于特征工程，增加多尺度更好
    windows = [30, 90, 365]
    
    rf_annual = 0.0
    rf_daily = rf_annual / 365.0

    for w in windows:
        if len(df) < w:
            continue
        
        roll_mean = df['Return'].rolling(window=w).mean()
        roll_std = df['Return'].rolling(window=w).std()
        
        # 防止除以0
        roll_std = roll_std.replace(0, np.nan)
        
        # 年化夏普
        df[f'Sharpe_{w}d'] = (roll_mean - rf_daily) / roll_std * np.sqrt(365)

    # 清理计算产生的 NaN
    # df = df.dropna()
    
    # 保存时保持 CSV 格式一致性 (Date 不做 index)
    df.to_csv(args.filename, index=False)
    print(f"Added Sharpe Ratios to {args.filename}")

if __name__ == "__main__":
    main()