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

# 1. Fetch Data from Yahoo Finance
print(f"Fetching last 1 year of data for {args.ticker}...")
ticker_obj = yf.Ticker(args.ticker)
df = ticker_obj.history(period="1y")

if df.empty:
    raise ValueError(f"No data found for ticker {args.ticker}")

# Reset index to make Date a column
df = df.reset_index()

# Yfinance returns timezone-aware datetimes. We convert to timezone-naive.
if pd.api.types.is_datetime64_any_dtype(df['Date']):
    df['Date'] = df['Date'].dt.tz_localize(None)

# 转换日期格式
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

# 按日期升序排列 (Oldest to Newest) - 关键步骤
df = df.sort_values('Date', ascending=True)

# 3. 设置为索引
df = df.set_index('Date')

# Ensure required columns exist
required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"Missing required column: {col}")

# --- 特征工程开始 ---

# 1. 使用 ta.add_all_ta_features 一次性生成所有特征
# 这样可以保证与你之前的数据完全一致（包括 fillna=True 的行为）
print("Calculating all TA features using library defaults...")
try:
    # 这个函数会生成大约 80+ 个特征列
    df = ta.add_all_ta_features(
        df, 
        open="Open", 
        high="High", 
        low="Low", 
        close="Close", 
        volume="Volume", 
        fillna=True
    )
    print("All TA features added.")
except Exception as e:
    print(f"Error in add_all_ta_features: {e}")

# 2. 时间特征
try:
    next_dates = df.index + pd.Timedelta(days=1)
    df['next_weekday'] = next_dates.weekday
    df['next_is_weekend'] = df['next_weekday'].isin([5, 6]).astype(int)
    weekday = df.index.weekday
    df['wd_angle'] = np.arctan2(np.sin(2 * np.pi * weekday / 7), np.cos(2 * np.pi * weekday / 7))
    print("Time features added.")
except Exception as e:
    print(f"Error calculating time features: {e}")

# --- 特征筛选与重命名 ---

# 定义我们最终想要保留的列
keep_columns = [
    # 基础列
    'Open', 'High', 'Low', 'Close', 'Volume',
    
    # 从 add_all_ta_features 中提取的目标特征
    'volatility_bbli',      # 布林带下轨指示器
    'volatility_bbl',       # BB_Lower
    
    # 时间特征
    'next_weekday', 'next_is_weekend', 'wd_angle'
]

# 过滤 DataFrame，只保留存在的列
existing_cols = [col for col in keep_columns if col in df.columns]
df = df[existing_cols]

print(f"Selected {len(df.columns)} features from the generated set.")

# --- 特征工程结束 ---

# Reset index to make Date a column again for saving
df = df.reset_index()

# 5. Clean Data: Remove empty rows
# 虽然使用了 fillna=True，但为了保险起见还是检查一下
original_len = len(df)
df = df.dropna()
if original_len != len(df):
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
