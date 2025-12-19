# prepare_data.py
import pandas as pd
import numpy as np
import argparse
from datetime import datetime, timedelta
import requests
import yfinance as yf
import os
import ta

# Parse command line arguments
parser = argparse.ArgumentParser(description='Add additional features to CSV file.')
parser.add_argument('--ticker', type=str, required=True, help='Yahoo Finance Ticker (e.g., BTC-USD)')
parser.add_argument('--filename', type=str, required=False, help='The CSV file name to save (optional)')
args = parser.parse_args()

# Determine filename
if args.filename:
    filename = args.filename
else:
    filename = f"{args.ticker}.csv"

# 1. Fetch Data from Yahoo Finance (Last 40 days to ensure enough buffer for TA windows)
print(f"Fetching last 40 days of data for {args.ticker}...")
ticker_obj = yf.Ticker(args.ticker)
df = ticker_obj.history(period="40d")

if df.empty:
    raise ValueError(f"No data found for ticker {args.ticker}")

# Reset index to make Date a column (yfinance sets Date as index by default)
df = df.reset_index()

# Yfinance returns timezone-aware datetimes. We convert to timezone-naive
# to prevent issues with other libraries or merging.
if pd.api.types.is_datetime64_any_dtype(df['Date']):
    df['Date'] = df['Date'].dt.tz_localize(None)

# 2. 转换日期格式
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

# 2. 按日期升序排列 (Oldest to Newest)
# 金融时间序列计算(shift/rolling/ta-lib)必须基于时间顺序
df = df.sort_values('Date', ascending=True)

# 3. 设置为索引
df = df.set_index('Date')

# Ensure required columns exist
required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"Missing required column: {col}")

# --- 特征工程开始 ---

# 4.1 添加 TA 库的 momentum_stoch 特征
# Stochastic Oscillator (随机指标)
# 公式: 100 * (Close - Lowest Low) / (Highest High - Lowest Low)
# 默认窗口 window=14, smooth_window=3
try:
    df['momentum_stoch'] = ta.momentum.stoch(
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        window=14,
        smooth_window=3
    )
    print("Feature 'momentum_stoch' added.")
except Exception as e:
    print(f"Error calculating momentum_stoch: {e}")

# 4.2 添加 TA 库的 ATR_14 特征
# Average True Range (平均真实波幅)
# 用于衡量波动率，window=14
try:
    df['ATR_14'] = ta.volatility.average_true_range(
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        window=14
    )
    print("Feature 'ATR_14' added.")
except Exception as e:
    print(f"Error calculating ATR_14: {e}")
# --- 特征工程结束 ---

# Reset index to make Date a column again for saving
df = df.reset_index()

# 5. Clean Data: Remove empty rows
# (Rolling window of 14 creates NaNs at the start)
original_len = len(df)
df = df.dropna()
print(f"Removed {original_len - len(df)} rows containing NaN values.")

# Optional: Sort descending (Newest first)
df = df.sort_values('Date', ascending=False)

# 6. Save to 'data' directory
output_dir = "data"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

output_path = os.path.join(output_dir, filename)
df.to_csv(output_path, index=False)
print(f"File saved successfully to: {output_path}")
