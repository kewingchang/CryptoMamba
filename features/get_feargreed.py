import argparse
import pandas as pd
import numpy as np
import requests
import time
import random
from datetime import datetime, timedelta

def get_fng_history_direct():
    """
    直接从 Alternative.me API 获取全量历史数据。
    """
    url = "https://api.alternative.me/fng/"
    params = {
        "limit": 0,  # 0 代表获取全部历史数据
        "format": "json"
        # 移除 'date_format' 参数，确保 API 返回原始的 Unix timestamp (如 "1517443200")
        # 这样 pd.to_numeric 才能正确处理
    }
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    max_retries = 3
    for attempt in range(max_retries):
        try:
            print(f"正在从 API 拉取 F&G 全量数据 (尝试 {attempt + 1}/{max_retries})...")
            
            time.sleep(random.uniform(1, 3)) 
            
            response = requests.get(url, params=params, headers=headers, timeout=20)
            response.raise_for_status()
            
            data = response.json()
            
            # 检查 API 是否返回了错误信息
            if isinstance(data, dict) and data.get('metadata', {}).get('error'):
                raise Exception(f"API Error: {data['metadata']['error']}")
            
            records = data['data']
            print(f"成功获取 {len(records)} 条 F&G 数据。")
            
            # 转为 DataFrame
            fng_df = pd.DataFrame(records)
            return fng_df

        except Exception as e:
            print(f"API 请求失败: {e}")
            if attempt < max_retries - 1:
                wait_time = random.uniform(3, 10)
                print(f"等待 {wait_time:.1f} 秒后重试...")
                time.sleep(wait_time)
            else:
                raise Exception("达到最大重试次数，无法获取 F&G 数据。")

def process_fng_data(fng_df):
    """
    清洗 F&G 数据：时间戳转换、去重、排序
    """
    # 调试信息：打印第一行 timestamp 确保格式正确
    # print(f"DEBUG: First raw timestamp: {fng_df['timestamp'].iloc[0]}")

    # 1. 转换时间戳
    # 这里的 timestamp 应该是 Unix Epoch (秒)，例如 1517443200
    try:
        fng_df['timestamp'] = pd.to_numeric(fng_df['timestamp'])
    except Exception as e:
        print(f"Error parsing timestamps: {e}")
        print("First 5 invalid timestamps:", fng_df['timestamp'].head().tolist())
        raise

    fng_df['datetime'] = pd.to_datetime(fng_df['timestamp'], unit='s')
    
    # 2. 提取日期 (UTC 0点)
    fng_df['Date'] = fng_df['datetime'].dt.date
    
    # 3. 转换数值
    fng_df['fng_value'] = pd.to_numeric(fng_df['value'], errors='coerce')
    
    # 4. 关键修正：按时间戳严格排序 (旧 -> 新)
    fng_df = fng_df.sort_values('timestamp', ascending=True)
    
    # 5. 去重：如果有同一天的数据，保留该天“最晚”发布的那一条
    fng_df = fng_df.drop_duplicates(subset=['Date'], keep='last')
    
    return fng_df[['Date', 'fng_value']].reset_index(drop=True)

def calculate_advanced_features(df):
    """
    计算高级特征。注意：任何涉及 F&G 的计算，如果 F&G 缺失，结果应为 NaN。
    """
    print("正在计算衍生特征...")
    
    # 确保按日期升序排列，这对 shift/diff 至关重要
    df = df.sort_values('Date', ascending=True).reset_index(drop=True)
    
    # --- 0. 基础预处理 ---
    numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'fng_value']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # 计算价格 Log Return
    df['log_ret'] = np.log(df['Close'] / df['Close'].shift(1))
    
    # F&G 归一化 (0-1)
    df['fng_norm'] = df['fng_value'] / 100.0

    # --- 1. 滞后特征 (Lag Features) ---
    for lag in [1, 2, 3]:
        df[f'fng_lag_{lag}'] = df['fng_norm'].shift(lag)

    # --- 2. 动量与波动 ---
    df['fng_diff'] = df['fng_norm'].diff()
    df['fng_volatility_7d'] = df['fng_norm'].rolling(window=7).std()

    # --- 3. 极端状态 ---
    # 如果 fng 是 NaN，Flag 也是 NaN
    df['is_extreme_fear'] = np.where(df['fng_value'].isna(), np.nan, (df['fng_value'] <= 20).astype(float))
    df['is_extreme_greed'] = np.where(df['fng_value'].isna(), np.nan, (df['fng_value'] >= 80).astype(float))

    # --- 4. 价格与情绪背离 ---
    price_trend_5d = df['Close'].diff(5)
    fng_trend_5d = df['fng_norm'].diff(5)
    
    df['divergence_flag'] = np.sign(price_trend_5d) * np.sign(fng_trend_5d)
    
    price_rank = df['Close'].rolling(14).rank()
    fng_rank = df['fng_norm'].rolling(14).rank()
    df['divergence_rank_diff'] = price_rank - fng_rank

    # --- 5. 融合特征 ---
    if 'Volume' in df.columns:
        log_vol = np.log1p(df['Volume'])
        df['vol_fear_interaction'] = log_vol * (1 - df['fng_norm'])
    
    # 独立计算价格波动率，不依赖 F&G
    df['daily_range_pct'] = (df['High'] - df['Low']) / df['Close']
    # 交互项依赖 F&G
    df['fng_range_interaction'] = df['daily_range_pct'] * df['fng_norm']

    return df

def main():
    parser = argparse.ArgumentParser(description="为加密货币数据添加 Fear & Greed 高级特征 (API直连版)")
    parser.add_argument("--filename", type=str, required=True, help="输入的CSV文件路径")
    args = parser.parse_args()

    # 1. 读取本地数据
    try:
        print(f"正在读取文件: {args.filename}")
        df = pd.read_csv(args.filename)
        
        # 智能识别日期列
        date_col = None
        candidates = ['Date', 'date', 'timestamp', 'time', 'snapped_at']
        for col in candidates:
            if col in df.columns:
                date_col = col
                break
        
        if not date_col:
            raise ValueError(f"在CSV中未找到日期列，尝试过: {candidates}")
            
        print(f"检测到日期列: {date_col}")
        
        # 统一转换为 datetime.date 对象
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce').dt.date
        df = df.sort_values(date_col, ascending=True)
        
        print(f"本地数据范围: {df[date_col].min()} 到 {df[date_col].max()} (共 {len(df)} 行)")

    except Exception as e:
        print(f"读取CSV失败: {e}")
        return

    # 2. 获取 F&G 数据
    try:
        raw_fng_df = get_fng_history_direct()
        fng_df = process_fng_data(raw_fng_df)
        print(f"API 数据处理完毕，范围: {fng_df['Date'].min()} 到 {fng_df['Date'].max()}")
    except Exception as e:
        print(f"Fatal Error: 获取 F&G 数据彻底失败: {e}")
        return

    # 3. 合并数据 (Left Join)
    print("正在合并数据 (Left Join)...")
    merged_df = pd.merge(df, fng_df, left_on=date_col, right_on='Date', how='left')
    
    if 'Date' in merged_df.columns and date_col != 'Date':
        merged_df = merged_df.drop(columns=['Date'])

    # 4. 缺失值报告
    missing_fng = merged_df[merged_df['fng_value'].isna()]
    if not missing_fng.empty:
        print("="*50)
        print(f"警告: 发现 {len(missing_fng)} 行缺少 F&G 数据!")
        print("示例缺失日期:")
        print(missing_fng[date_col].head(3).to_list())
        print("="*50)
    else:
        print("完美: 所有行均匹配到了 F&G 数据。")

    # 5. 计算特征
    final_df = calculate_advanced_features(merged_df)

    # 6. 保存
    # output_filename = args.filename.replace('.csv', '_fng.csv')
    output_filename = args.filename
    
    final_df.to_csv(output_filename, index=False)
    print(f"处理完成! 文件已保存至: {output_filename}")
    
    # 预览
    print("\n数据预览 (Last 5 rows):")
    cols_to_show = [date_col, 'Close', 'fng_value', 'fng_lag_1', 'daily_range_pct']
    print(final_df[[c for c in cols_to_show if c in final_df.columns]].tail())

if __name__ == "__main__":
    main()