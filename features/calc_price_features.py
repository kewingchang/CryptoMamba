import pandas as pd
import numpy as np
import argparse
import sys

def process_args():
    parser = argparse.ArgumentParser(description="Price & MarketCap Feature Engineering (Stationary Transformation)")
    parser.add_argument("--input", type=str, required=True, help="Path to input CSV (e.g., BTC-USD.csv)")
    
    try: args = parser.parse_args()
    except: args, _ = parser.parse_known_args()
    return args

def load_data(filepath):
    print(f"[1/4] Loading data from {filepath}...")
    try:
        df = pd.read_csv(filepath)
        # 简单检查必要列
        required = ['Open', 'High', 'Low', 'Close', 'marketCap']
        missing = [c for c in required if c not in df.columns]
        if missing:
            print(f"[Warning] Missing columns: {missing}. Features relying on them will be skipped.")
        return df
    except Exception as e:
        print(f"[Fatal] Error loading CSV: {e}")
        sys.exit(1)

def engineer_features(df):
    print("[2/4] Engineering Price & MarketCap features...")
    
    # 为了计算相对昨日的变化，我们需要 Shift(1) 的 Close
    # Prev_Close: 昨日收盘价
    df['Prev_Close'] = df['Close'].shift(1)
    
    # ==========================================
    # 1. 价格变化率 (相对于昨日收盘) - 趋势类
    # ==========================================
    # Log Return (对数收益率) 通常比 pct_change 分布更正态，但也可用 pct_change
    # 这里使用 pct_change 保持直观
    
    # Close_Chg: 今天收盘相对于昨天收盘的涨跌幅
    df['PF_Close_Chg'] = (df['Close'] - df['Prev_Close']) / df['Prev_Close']
    
    # Open_Gap: 今天开盘相对于昨天收盘的跳空幅度
    df['PF_Open_Gap'] = (df['Open'] - df['Prev_Close']) / df['Prev_Close']
    
    # High_Dev: 今天最高价相对于昨天收盘的偏离度
    df['PF_High_Dev'] = (df['High'] - df['Prev_Close']) / df['Prev_Close']
    
    # Low_Dev: 今天最低价相对于昨天收盘的偏离度
    df['PF_Low_Dev'] = (df['Low'] - df['Prev_Close']) / df['Prev_Close']

    # ==========================================
    # 2. K线形态特征 (日内结构) - 形态类
    # ==========================================
    # 分母通常使用 Open 或 Prev_Close。使用 Open 更能体现当天的 K 线实体比例。
    
    # K_Body: 实体大小 (Close - Open) / Open
    # 正值代表阳线，负值代表阴线
    df['PF_K_Body'] = (df['Close'] - df['Open']) / df['Open']
    
    # 计算上影线和下影线长度
    # 上影线 = High - Max(Open, Close)
    # 下影线 = Min(Open, Close) - Low
    _max_oc = df[['Open', 'Close']].max(axis=1)
    _min_oc = df[['Open', 'Close']].min(axis=1)
    
    df['PF_K_UpShadow'] = (df['High'] - _max_oc) / df['Open']
    df['PF_K_LowShadow'] = (_min_oc - df['Low']) / df['Open']
    
    # K_Range: 全天振幅 (High - Low) / Open
    df['PF_K_Range'] = (df['High'] - df['Low']) / df['Open']

    # ==========================================
    # 3. MarketCap 处理
    # ==========================================
    if 'marketCap' in df.columns:
        # MarketCap_Chg: 市值变化率
        # 通常这跟 Close_Chg 99% 相关，但还是算一下
        df['PF_MCap_Chg'] = df['marketCap'].pct_change()
        
        # 既然有了变化率，原始的绝对值 marketCap 就可以（也建议）在训练时丢弃了
        # 但我们这里只负责加特征，不删原数据
    
    # 清理中间变量
    if 'Prev_Close' in df.columns:
        df.drop(columns=['Prev_Close'], inplace=True)
        
    return df

def clean_and_save(df, filepath):
    print("[3/4] Cleaning NaNs...")
    # 因为用了 shift(1) 和 pct_change，第一行会变成 NaN
    # 简单策略：填充0 或者 删除第一行。
    # 对于时间序列训练，删除第一行是最安全的
    
    print(f"[4/4] Saving to {filepath}...")
    # 覆盖保存
    df.to_csv(filepath, index=False)
    print("Done.")

def main():
    args = process_args()
    
    # 1. 加载
    df = load_data(args.input)
    
    # 2. 计算特征
    df = engineer_features(df)
    
    # 3. 保存
    clean_and_save(df, args.input)

if __name__ == "__main__":
    main()