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
    
    注意：虽然需求提到分10次拉取，但该API的 'limit=0' 可以在一次请求中
    返回自2018年以来的所有数据(仅约2500条，几十KB)，这是最安全、
    最不容易出现日期拼接错误的方法。为了防止请求失败，加入了重试机制。
    """
    url = "https://api.alternative.me/fng/"
    params = {
        "limit": 0,  # 0 代表获取全部历史数据
        "format": "json",
        "date_format": "cn" # 使用易读格式辅助调试，实际处理用 timestamp
    }
    
    # 伪装成浏览器，防止被简单反爬拦截
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    max_retries = 3
    for attempt in range(max_retries):
        try:
            print(f"正在从 API 拉取 F&G 全量数据 (尝试 {attempt + 1}/{max_retries})...")
            
            # 模拟人为延迟，虽然对于单次请求意义不大，但符合“不频繁请求”的精神
            time.sleep(random.uniform(2, 5)) 
            
            response = requests.get(url, params=params, headers=headers, timeout=20)
            response.raise_for_status()
            
            data = response.json()
            if data['metadata']['error'] is not None:
                raise Exception(f"API Error: {data['metadata']['error']}")
            
            records = data['data']
            print(f"成功获取 {len(records)} 条 F&G 数据。")
            
            # 转为 DataFrame
            fng_df = pd.DataFrame(records)
            return fng_df

        except Exception as e:
            print(f"API 请求失败: {e}")
            if attempt < max_retries - 1:
                wait_time = random.uniform(5, 15)
                print(f"等待 {wait_time:.1f} 秒后重试...")
                time.sleep(wait_time)
            else:
                raise Exception("达到最大重试次数，无法获取 F&G 数据。")

def process_fng_data(fng_df):
    """
    清洗 F&G 数据：时间戳转换、去重、排序
    """
    # 1. 转换时间戳 (API返回的是字符串类型的unix timestamp)
    fng_df['timestamp'] = pd.to_numeric(fng_df['timestamp'])
    fng_df['datetime'] = pd.to_datetime(fng_df['timestamp'], unit='s')
    
    # 2. 提取日期 (UTC 0点)
    fng_df['Date'] = fng_df['datetime'].dt.date
    
    # 3. 转换数值
    fng_df['fng_value'] = pd.to_numeric(fng_df['value'], errors='coerce')
    
    # 4. 关键修正：按时间戳严格排序 (旧 -> 新)
    # 这解决了之前代码中可能因当天有多条数据而取错值的问题
    fng_df = fng_df.sort_values('timestamp', ascending=True)
    
    # 5. 去重：如果有同一天的数据，保留该天“最晚”发布的那一条
    fng_df = fng_df.drop_duplicates(subset=['Date'], keep='last')
    
    # 只保留需要的列
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

    # 计算价格 Log Return (特征工程常用)
    df['log_ret'] = np.log(df['Close'] / df['Close'].shift(1))
    
    # F&G 归一化 (0-1)
    df['fng_norm'] = df['fng_value'] / 100.0

    # --- 1. 滞后特征 (Lag Features) ---
    # 只有当 df 已经按旧到新排序时，shift(1) 才是“昨天”的数据
    for lag in [1, 2, 3]:
        df[f'fng_lag_{lag}'] = df['fng_norm'].shift(lag)

    # --- 2. 动量与波动 ---
    df['fng_diff'] = df['fng_norm'].diff() # 今天 - 昨天
    # 如果 fng_norm 是 NaN，rolling std 也会是 NaN (除非设置 min_periods)
    # 这里我们保持严格，如果窗口内有缺失，结果就是 NaN，防止脏数据进入模型
    df['fng_volatility_7d'] = df['fng_norm'].rolling(window=7).std()

    # --- 3. 极端状态 ---
    # 注意：如果 fng_value 是 NaN，这里比较结果是 False (0)，这是安全的吗？
    # 更安全的做法是：如果是 NaN，Flag 也设为 NaN 或 0。这里设为 0，但在逻辑上要小心。
    # 为了严谨，如果 fng 是 NaN，我们让 Flag 也是 NaN
    df['is_extreme_fear'] = np.where(df['fng_value'].isna(), np.nan, (df['fng_value'] <= 20).astype(float))
    df['is_extreme_greed'] = np.where(df['fng_value'].isna(), np.nan, (df['fng_value'] >= 80).astype(float))

    # --- 4. 价格与情绪背离 ---
    price_trend_5d = df['Close'].diff(5)
    fng_trend_5d = df['fng_norm'].diff(5)
    
    # 符号函数：np.sign(NaN) 会返回 NaN，符合预期
    df['divergence_flag'] = np.sign(price_trend_5d) * np.sign(fng_trend_5d)
    
    # Rank 差值
    price_rank = df['Close'].rolling(14).rank()
    fng_rank = df['fng_norm'].rolling(14).rank()
    df['divergence_rank_diff'] = price_rank - fng_rank

    # --- 5. 融合特征 ---
    # 即使 Volume 或 Close 存在，只要 fng_norm 是 NaN，交互项就应该是 NaN
    if 'Volume' in df.columns:
        log_vol = np.log1p(df['Volume'])
        df['vol_fear_interaction'] = log_vol * (1 - df['fng_norm'])
    
    # 修正：daily_range_pct 是纯价格特征，不应该受 fng 缺失影响
    # 先独立计算
    df['daily_range_pct'] = (df['High'] - df['Low']) / df['Close']
    # 再计算交互项 (交互项会受 fng 缺失影响)
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
        
        # 统一转换为 datetime.date 对象，并处理无效日期
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce').dt.date
        
        # 检查是否有日期解析失败的行
        invalid_dates = df[df[date_col].isna()]
        if not invalid_dates.empty:
            print(f"警告: 有 {len(invalid_dates)} 行日期解析失败，将被保留在末尾或开头，但不参与合并。")

        # 关键：按日期升序排列 (Old -> New)
        df = df.sort_values(date_col, ascending=True)
        
        start_date = df[date_col].min()
        end_date = df[date_col].max()
        print(f"本地数据范围: {start_date} 到 {end_date} (共 {len(df)} 行)")

    except Exception as e:
        print(f"读取CSV失败: {e}")
        return

    # 2. 获取 F&G 数据 (全量)
    try:
        raw_fng_df = get_fng_history_direct()
        fng_df = process_fng_data(raw_fng_df)
        print(f"API 数据处理完毕，范围: {fng_df['Date'].min()} 到 {fng_df['Date'].max()}")
    except Exception as e:
        print(f"Fatal Error: 获取 F&G 数据彻底失败: {e}")
        return

    # 3. 合并数据
    print("正在合并数据 (Left Join)...")
    # Left Join: 保证左边(原始数据)一行都不会少
    merged_df = pd.merge(df, fng_df, left_on=date_col, right_on='Date', how='left')
    
    # 清理重复的 Date 列
    if 'Date' in merged_df.columns and date_col != 'Date':
        merged_df = merged_df.drop(columns=['Date'])

    # 4. 缺失值检查 (不做填充，只报告)
    missing_fng = merged_df[merged_df['fng_value'].isna()]
    if not missing_fng.empty:
        print("="*50)
        print(f"警告: 发现 {len(missing_fng)} 行缺少 F&G 数据!")
        print("常见原因: 数据早于2018年2月，或API今日数据尚未更新。")
        print("示例缺失日期:")
        print(missing_fng[date_col].head(3).to_list())
        print("="*50)
    else:
        print("完美: 所有行均匹配到了 F&G 数据。")

    # 5. 计算特征
    final_df = calculate_advanced_features(merged_df)

    # 6. 不删除任何数据 (No Dropna)
    # 再次确认行数一致
    if len(final_df) != len(df):
        print(f"严重警告: 行数发生变化! 原: {len(df)}, 现: {len(final_df)}")
    else:
        print(f"行数校验通过: {len(final_df)} 行。")

    # 7. 保存
    # output_filename = args.filename.replace('.csv', '_fng.csv')
    output_filename = args.filename
    
    # 转换 Date 为字符串以便保存
    # final_df[date_col] = final_df[date_col].astype(str)
    
    final_df.to_csv(output_filename, index=False)
    print(f"处理完成! 文件已保存至: {output_filename}")
    
    # 打印最后几行用于人工检查
    print("\n数据预览 (Last 5 rows):")
    cols_to_show = [date_col, 'Close', 'fng_value', 'fng_lag_1', 'daily_range_pct']
    print(final_df[[c for c in cols_to_show if c in final_df.columns]].tail())

if __name__ == "__main__":
    main()