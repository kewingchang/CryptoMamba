import numpy as np
import pandas as pd
import argparse
import yfinance as yf
import os

# Parse command line arguments
parser = argparse.ArgumentParser(description='Fetch and clean data from Yahoo Finance.')
parser.add_argument('--ticker', type=str, required=True, help='Yahoo Finance Ticker (e.g., BTC-USD)')
parser.add_argument('--save_to', type=str, required=True, help='Full file path to save the CSV file (e.g., /content/data/data.csv)')
parser.add_argument('--period', type=str, required=True, help='time period (e.g., 1y, 5d, max)')

args = parser.parse_args()

# 1. Fetch Data from Yahoo Finance
print(f"Fetching last {args.period} of data for {args.ticker}...")
try:
    ticker_obj = yf.Ticker(args.ticker)
    df = ticker_obj.history(period=args.period)
except Exception as e:
    raise RuntimeError(f"Failed to fetch data: {e}")

if df.empty:
    raise ValueError(f"No data found for ticker {args.ticker}. Please check the ticker symbol.")

# 2. Reset index to make Date a column
# yfinance 返回的数据 Date 是索引，我们需要将其变成普通列
df = df.reset_index()

# 3. Handle Date format and Timezones
# 确保 Date 列存在
if 'Date' not in df.columns:
    raise ValueError("Date column missing after reset_index. Data source structure might have changed.")

# 转换为 datetime 对象
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

# 去除时区信息 (Timezone-naive)，防止保存或后续处理报错
if pd.api.types.is_datetime64_any_dtype(df['Date']):
    df['Date'] = df['Date'].dt.tz_localize(None)

# 4. Filter Columns
# 修正2: 在保留列中必须加入 'Date'，否则日期会被删掉
keep_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']

# 检查缺失列
missing_cols = [col for col in keep_columns if col not in df.columns]
if missing_cols:
    raise ValueError(f"Missing required columns: {missing_cols}")

# 只保留需要的列
df = df[keep_columns]

# 5. Clean Data
original_len = len(df)
df = df.dropna()
if original_len != len(df):
    print(f"Removed {original_len - len(df)} rows containing NaN values.")

# 按日期排序
df = df.sort_values('Date', ascending=True)

# 重置索引为 0, 1, 2... (不保留旧的乱序索引)
df = df.reset_index(drop=True)

print(f"Processed {len(df)} rows with columns: {list(df.columns)}")

# 6. Save to specified file path
output_path = args.save_to
output_dir = os.path.dirname(output_path)

# 如果路径包含目录，且目录不存在，则创建
if output_dir and not os.path.exists(output_dir):
    os.makedirs(output_dir)

df.to_csv(output_path, index=False)
print(f"File saved successfully to: {output_path}")