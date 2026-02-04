import argparse
import pandas as pd
import numpy as np


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

    # 2. 计算每日收益率 (Daily Return)
    df['Return'] = df['Close'].pct_change()

    # 3. 设定参数
    # Bitbo 使用 "Rolling 52 Week"，即 1 年。加密货币市场全年无休，通常取 365 天。
    window_size = 364

    # 设定无风险利率 (Risk Free Rate, Rf)
    # 如果假设 Rf 为 0 (最常见且简单的做法)
    rf_annual = 0.0

    # 如果你想更严谨，可以使用例如 4% 的年化国债收益率
    # rf_annual = 0.01

    # 将年化无风险利率转换为日化
    rf_daily = rf_annual / 365

    # 4. 计算滚动夏普比率
    # 公式：(平均日收益率 - 日无风险利率) / 日收益率标准差 * sqrt(365)

    # 计算滚动平均收益 (Rolling Mean)
    rolling_mean = df['Return'].rolling(window=window_size).mean()

    # 计算滚动标准差 (Rolling Std Dev)
    rolling_std = df['Return'].rolling(window=window_size).std()

    # 应用夏普比率公式
    # 注意：np.sqrt(365) 是年化因子
    df['Bitbo_Sharpe'] = (rolling_mean - rf_daily) / rolling_std * np.sqrt(365)

    # Save
    df.to_csv(filename)
    print(f"Successfully added LEAKAGE-FREE advanced volume features to {filename}")

if __name__ == "__main__":
    main()