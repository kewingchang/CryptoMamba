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
# 明确 save_to 接收完整文件路径
parser.add_argument('--save_to', type=str, required=True, help='Full file path to save the CSV file (e.g., /content/data/data.csv)')

args = parser.parse_args()

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

# --- 关键修改开始：解决 FutureWarning ---
# 原代码在这里直接 set_index('Date')，导致后续 ta 库使用整数下标时报错。
# 我们改为：先重置为标准的整数索引 (0, 1, 2...)，让 ta 库计算时“位置”和“索引”一致。
df = df.reset_index(drop=True)

# Ensure required columns exist
required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"Missing required column: {col}")

# --- 特征工程开始 ---

# 1. 使用 ta.add_all_ta_features 一次性生成所有特征
print("Calculating all TA features using library defaults...")
try:
    # 此时 df 的索引是 RangeIndex(整数)，ta 库内部使用整数访问不会触发 FutureWarning
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

# --- 关键修改：TA 计算完毕后，再设置 Date 为索引 ---
# 因为下面的时间特征计算依赖于 df.index 是日期类型
df = df.set_index('Date')

# 2. 时间特征
try:
    # 此时 index 已经是 Date 了，可以正常计算
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
original_len = len(df)
df = df.dropna()
if original_len != len(df):
    print(f"Removed {original_len - len(df)} rows containing NaN values.")

# Optional: Sort descending (Newest first)
df = df.sort_values('Date', ascending=False)

# 6. Save to specified file path (修改 2)
output_path = args.save_to

# 获取目录路径
output_dir = os.path.dirname(output_path)

# 如果路径包含目录，且目录不存在，则创建
if output_dir and not os.path.exists(output_dir):
    os.makedirs(output_dir)

df.to_csv(output_path, index=False)
print(f"File saved successfully to: {output_path}")