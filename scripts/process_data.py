# process_data.py
import pandas as pd
import numpy as np
import argparse
import sys
import os
from ta import add_all_ta_features
from ta.volatility import AverageTrueRange

def process_args():
    parser = argparse.ArgumentParser(description="Feature Engineering & Label Generation")
    parser.add_argument("--input", type=str, required=True, help="Path to input CSV (e.g., BTC-USD.csv)")
    parser.add_argument("--factor", type=float, default=1.0, help="ATR Multiplier for Labeling (Barrier width)")
    parser.add_argument("--time_period", type=int, default=14, help="Window size for ATR calculation")
    parser.add_argument("--barrier_factor", type=float, default=1.0, help="Same as factor (redundant but kept for request)")
    
    # 兼容 Jupyter/Colab
    try: args = parser.parse_args()
    except: args, _ = parser.parse_known_args()
    return args

def load_data(filepath):
    print(f"[1/5] Loading data from {filepath}...")
    try:
        df = pd.read_csv(filepath)
        # 确保 Date 列存在并转换格式
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.sort_values('Date').reset_index(drop=True)
        else:
            print("Error: 'Date' column not found in CSV.")
            sys.exit(1)
        return df
    except Exception as e:
        print(f"Error loading CSV: {e}")
        sys.exit(1)

def add_features(df):
    print("[2/5] Engineering Features (ta library)...")
    # 使用 ta 库的一键添加所有特征功能
    # 这会增加几十列特征 (RSI, MACD, Bollinger, etc.)
    try:
        df = add_all_ta_features(
            df, open="Open", high="High", low="Low", close="Close", volume="Volume", fillna=True
        )
        print(f"      -> Added {len(df.columns)} features in total.")
    except Exception as e:
        print(f"Error in feature engineering: {e}")
        sys.exit(1)
    return df

def generate_labels(df, atr_window, barrier_multiplier):
    print(f"[3/5] Generating Labels (Triple Barrier Method)...")
    print(f"      -> ATR Window: {atr_window}, Multiplier: {barrier_multiplier}")
    
    # 1. 计算用于生成标签的 ATR (Dynamic Horizon)
    # 我们单独计算这一列，确保它是基于用户指定的 time_period
    atr_indicator = AverageTrueRange(high=df['High'], low=df['Low'], close=df['Close'], window=atr_window)
    df['Label_ATR'] = atr_indicator.average_true_range()
    
    labels = []
    
    # 2. 遍历生成标签
    # 我们根据 "今天(i)" 的数据(ATR)，来判断 "明天(i+1)" 的走势
    # 所以 Target 是归属于 "今天" 的特征的
    
    for i in range(len(df) - 1):
        # 获取今天的 ATR 作为波动率基准
        curr_atr = df.iloc[i]['Label_ATR']
        
        # 获取明天的价格数据
        next_open = df.iloc[i+1]['Open']
        next_high = df.iloc[i+1]['High']
        next_low = df.iloc[i+1]['Low']
        
        # 异常处理：如果 ATR 是 NaN (前几行)，无法生成标签
        if pd.isna(curr_atr) or curr_atr == 0:
            labels.append(0) # 默认为 Wait
            continue
            
        # 设定动态障碍 (Dynamic Barriers)
        # 基准价：明天的开盘价 (因为我们是在明天开盘时刻入场)
        # 上下轨宽度 = 今天的 ATR * 系数
        barrier_width = curr_atr * barrier_multiplier
        upper_barrier = next_open + barrier_width
        lower_barrier = next_open - barrier_width
        
        # 判断逻辑 (Triple Barrier)
        # 1. 涨破上轨 且 没跌破下轨 -> Long (1)
        hit_upper = next_high > upper_barrier
        hit_lower = next_low < lower_barrier
        
        if hit_upper and not hit_lower:
            labels.append(1) # Long
        elif hit_lower and not hit_upper:
            labels.append(2) # Short
        else:
            # 情况 A: 既没破上也没破下 (波动太小) -> 0
            # 情况 B: 既破上又破下 (双爆/扫损) -> 0 (太危险，标记为观望)
            labels.append(0) # Wait
            
    # 最后一行没有明天，无法生成标签，填 0 (后续会删除)
    labels.append(0)
    
    df['label'] = labels
    return df

def clean_and_save(df, filepath):
    print("[4/5] Cleaning Data...")
    # 1. 之前的 ta 库计算会导致前 N 行出现 NaN (例如 SMA200 会导致前200行无效)
    # 2. 倒数第一行没有 Label (或者是虚假的0)
    
    # 删除含有 NaN 的行
    original_len = len(df)
    df = df.dropna()
    
    # 删除最后一行 (因为它没有未来的 Target)
    if len(df) > 0:
        df = df.iloc[:-1]
        
    print(f"      -> Dropped {original_len - len(df)} rows (NaNs + Last row).")
    print(f"      -> Final Data Shape: {df.shape}")
    
    # 检查 label 分布
    print(f"      -> Label Distribution:\n{df['label'].value_counts()}")

    print(f"[5/5] Saving to {filepath}...")
    # 确保 label 是整数
    df['label'] = df['label'].astype(int)
    
    # 这里的 filepath 就是 --input 的路径，即覆盖原文件
    # 如果不想覆盖，可以修改这里
    df.to_csv(filepath, index=False)
    print("Done.")

def main():
    args = process_args()
    # 1. 加载
    df = load_data(args.input)    
    # 2. 特征工程
    df = add_features(df)
    # 3. 生成标签
    # 优先使用 barrier_factor，如果没传则使用 factor
    # 这里的逻辑是： ATR 窗口用 time_period，宽度倍数用 barrier_factor
    df = generate_labels(df, atr_window=args.time_period, barrier_multiplier=args.barrier_factor)
    
    # 4. 清洗并保存
    clean_and_save(df, args.input)

if __name__ == "__main__":
    main()