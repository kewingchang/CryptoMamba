# prepare_data.py
import pandas as pd
import numpy as np
import argparse
from datetime import datetime, timedelta
import requests
import yfinance as yf
import os
import ta
# 【新增】引入 Dune Client
from dune_client.client import DuneClient

# Parse command line arguments
parser = argparse.ArgumentParser(description='Add additional features to CSV file.')
parser.add_argument('--ticker', type=str, required=True, help='Yahoo Finance Ticker (e.g., BTC-USD)')
parser.add_argument('--filename', type=str, required=False, help='The CSV file name to save (optional)')
# 【新增】增加 Dune API Key 参数
parser.add_argument('--dune_api', type=str, required=False, default=None, help='Dune Analytics API Key')

args = parser.parse_args()

# Determine filename
if args.filename:
    filename = args.filename
else:
    filename = f"{args.ticker}.csv"

# 1. Fetch Data from Yahoo Finance (Last 60 days to ensure enough buffer for TA windows)
print(f"Fetching last 60 days of data for {args.ticker}...")
ticker_obj = yf.Ticker(args.ticker)
df = ticker_obj.history(period="60d")

if df.empty:
    raise ValueError(f"No data found for ticker {args.ticker}")

# Reset index to make Date a column (yfinance sets Date as index by default)
df = df.reset_index()

# Yfinance returns timezone-aware datetimes. We convert to timezone-naive
# to prevent issues with other libraries or merging.
if pd.api.types.is_datetime64_any_dtype(df['Date']):
    df['Date'] = df['Date'].dt.tz_localize(None)

# 转换日期格式
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

# 按日期升序排列 (Oldest to Newest) - 关键步骤
df = df.sort_values('Date', ascending=True)

# 3. 设置为索引
# 注意：所有的 shift/rolling/ta 计算都应在此之后进行
df = df.set_index('Date')

# Ensure required columns exist
required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"Missing required column: {col}")

# --- 【新增】Dune Analytics 数据获取开始 ---
if args.dune_api:
    print("Fetching Gas Used data from Dune Analytics (Query 741)...")
    try:
        dune = DuneClient(args.dune_api)
        # 获取 Query 741 的最新结果
        query_result = dune.get_latest_result(741)
        
        # 将结果转换为 DataFrame
        dune_data = query_result.result.rows
        dune_df = pd.DataFrame(dune_data)
        
        # Dune 返回的列名通常是小写的，但为了保险起见进行标准化
        dune_df.columns = [c.lower() for c in dune_df.columns]
        
        # 查找日期列
        date_col = 'time'
        if date_col and 'gas_used' in dune_df.columns:
            # 处理日期格式并去除时区，以便与 Yahoo Finance 数据对齐
            dune_df['Date'] = pd.to_datetime(dune_df[date_col]).dt.tz_localize(None)
            
            # 确保 gas_used 是数值类型
            dune_df['gas_used'] = pd.to_numeric(dune_df['gas_used'], errors='coerce')
            
            # 设置索引以便合并
            dune_df = dune_df.set_index('Date')
            
            # 只保留 gas_used 列
            dune_subset = dune_df[['gas_used']]
            
            # 【关键】合并数据 (Left Join)
            # 以 Yahoo Finance 的日期为准。
            # 因为 df 只有过去60天，merge 后 dune 的数据也会自动限制在过去60天。
            original_cols = len(df.columns)
            df = df.join(dune_subset, how='left')
            
            print(f"Dune 'gas_used' merged. Added {len(df.columns) - original_cols} column(s).")
            
            # 简单的缺失值填充策略：如果当天没数据，用前一天的数据填充 (Forward Fill)
            # 链上数据偶尔会有几小时延迟，ffill 能防止 dropna 删除掉最新的那一行
            if 'gas_used' in df.columns:
                df['gas_used'] = df['gas_used'].ffill()

        else:
            print(f"Warning: Expected columns ('day'/'date' and 'gas_used') not found in Dune response. Keys found: {dune_df.columns}")

    except Exception as e:
        print(f"Error fetching/merging Dune data: {e}")
        # 如果获取失败，代码继续运行，不中断，只是缺少该特征
else:
    print("Skipping Dune data (API Key is missing or default).")

# --- Dune Analytics 数据获取结束 ---


# --- 特征工程开始 ---

# 添加 TA 库的 momentum_stoch 特征
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

# 添加 TA 库的 stoch_rsi_indicator 特征
try:
    # 直接计算 StochRSI K
    df['momentum_stoch_rsi'] = ta.momentum.stochrsi(
        close=df['Close'], 
        window=14, 
        smooth1=3, 
        smooth2=3
    )
    print("Feature 'momentum_stoch_rsi' added.")
except Exception as e:
    print(f"Error calculating momentum_stoch_rsi: {e}")

# 添加 TA 库的 ATR_14 特征
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

# 添加 EMA 7 和 EMA 14
# Exponential Moving Average (指数移动平均线)
try:
    df['EMA_7'] = ta.trend.ema_indicator(
        close=df['Close'],
        window=7
    )
    df['EMA_14'] = ta.trend.ema_indicator(
        close=df['Close'],
        window=14
    )
    print("Features 'EMA_7' and 'EMA_14' added.")
except Exception as e:
    print(f"Error calculating EMA: {e}")

# 添加 DPO (Detrended Price Oscillator)
# 去趋势价格震荡指标，通常 window=20
try:
    df['trend_dpo'] = ta.trend.dpo(
        close=df['Close'],
        window=20
    )
    print("Feature 'trend_dpo' added.")
except Exception as e:
    print(f"Error calculating trend_dpo: {e}")

# Next day of year (Cyclical Time Features)
# 【逻辑修复】: 直接使用 df.index 计算，确保与 DataFrame 行对齐
try:
    # 获取索引中的日期并加1天
    next_dates = df.index + pd.Timedelta(days=1)
    # 计算 day of year
    next_day_of_year = next_dates.dayofyear    
    # 计算 cyclical features (Sin/Cos)
    # 使用 365.25 来考虑闰年平均值
    next_day_sin = np.sin(2 * np.pi * next_day_of_year / 365.25)
    next_day_cos = np.cos(2 * np.pi * next_day_of_year / 365.25)
    
    # 计算角度 (Arctan2)
    df['next_yr_angle'] = np.arctan2(next_day_sin, next_day_cos)
    print("Feature 'next_yr_angle' added.")
except Exception as e:
    print(f"Error calculating date features: {e}")

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
