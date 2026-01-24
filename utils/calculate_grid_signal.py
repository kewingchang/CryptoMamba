import pandas as pd
import numpy as np
import argparse
import sys
import os

def get_grid_signal(abs_chg, sym, rng):
    """
    核心策略逻辑: 黄金网格过滤器
    """
    # 1. 波动率过滤 (Sweet Spot: 3% - 6%)
    if not (3.0 <= rng <= 6.0):
        return False
        
    # 2. 动量过滤 (Sweet Spot: < 0.4%)
    if abs_chg > 0.4:
        return False
        
    # 3. 对称性过滤 (Trap Avoidance)
    # 极高胜率区: >= 0.9
    # 良好胜率区: 0.5 - 0.8
    # 死亡陷阱区: 0.8 - 0.9 (剔除)
    if sym >= 0.9:
        return True
    elif 0.5 <= sym <= 0.8:
        return True
    else:
        return False

def get_reference_price(df, target_idx, target_date_str):
    """
    获取计算用的参考价格 (Reference Price)
    逻辑: 优先用 Open，如果没有，则用 last_close
    """
    # 1. 尝试获取 Open
    if 'Open' in df.columns:
        val = df.at[target_idx, 'Open']
        if pd.notna(val) and float(val) > 0:
            return float(val)
    
    # 2. 尝试获取 last_close (当前行)
    if 'last_close' in df.columns:
        val = df.at[target_idx, 'last_close']
        if pd.notna(val) and float(val) > 0:
            print(f"[{target_date_str}] 'Open' price missing. Using 'last_close' ({val}) as reference.")
            return float(val)
            
    return None

def main():
    parser = argparse.ArgumentParser(description="Calculate Grid Signal for a specific date.")
    parser.add_argument('--data_path', type=str, default='/content/data/Pred-BTC-USD.csv', help='Path to the prediction CSV file')
    parser.add_argument('--date', type=str, required=True, help='Target date (e.g., 2026/1/5 or 2026-01-05)')
    
    args = parser.parse_args()
    
    # 1. 读取文件
    if not os.path.exists(args.data_path):
        print(f"Error: File not found at {args.data_path}")
        sys.exit(1)
        
    try:
        df = pd.read_csv(args.data_path)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        sys.exit(1)
        
    # 2. 智能日期匹配逻辑
    target_date_input = args.date.strip()
    
    try:
        # A. 将命令行输入的日期标准化 (去除时间，统一格式)
        target_dt = pd.to_datetime(target_date_input).normalize()
        
        # B. 将 DataFrame 中的 Date 列临时转换为 datetime 对象进行比较
        # errors='coerce' 会将无法解析的日期变为 NaT，防止报错
        file_dates = pd.to_datetime(df['Date'], errors='coerce').dt.normalize()
        
        # C. 查找匹配的索引
        matches = df.index[file_dates == target_dt].tolist()
        
    except Exception as e:
        print(f"Error parsing dates: {e}")
        print(f"Input date: {target_date_input}")
        sys.exit(1)
    
    if not matches:
        print(f"Info: Date '{target_date_input}' not found in {args.data_path}.")
        # 调试信息：打印文件中的前几个日期格式，方便排查
        print(f"First few dates in file: {df['Date'].head(3).tolist()}")
        sys.exit(0)
        
    target_idx = matches[0]
    # 获取 CSV 中原始的日期字符串，用于打印显示
    original_date_str = str(df.at[target_idx, 'Date'])
    print(f"Found record for input '{target_date_input}' (File says: '{original_date_str}') at index {target_idx}.")
    
    # 3. 准备数据
    try:
        # 必须存在的列
        req_cols = ['High-q0.8', 'Low-q0.2', 'pred_chg%']
        for col in req_cols:
            if col not in df.columns:
                print(f"Error: Column '{col}' missing in CSV.")
                sys.exit(1)
                
        high_q = float(df.at[target_idx, 'High-q0.8'])
        low_q = float(df.at[target_idx, 'Low-q0.2'])
        pred_chg = float(df.at[target_idx, 'pred_chg%'])
        
        # 获取参考价格 (Open 或 last_close)
        ref_price = get_reference_price(df, target_idx, original_date_str)
        
        if ref_price is None:
            print(f"Error: Neither 'Open' nor 'last_close' columns provided (or are empty) for date {original_date_str}.")
            sys.exit(1)
            
    except ValueError as e:
        print(f"Error parsing numerical data: {e}")
        sys.exit(1)
        
    # 4. 计算指标 (Symmetry & Proj_Range)
    # 防止除以0
    up_space = high_q - ref_price
    down_space = ref_price - low_q
    
    # 计算 symmetry
    numerator = min(up_space, down_space)
    denominator = max(up_space, down_space)
    if denominator == 0:
        symmetry = 0.0 # 异常情况，上下空间都为0
    else:
        symmetry = numerator / denominator
    
    # 计算 proj_range (%)
    proj_range = (high_q - low_q) / ref_price * 100
    
    # 5. 获取信号
    abs_chg = abs(pred_chg)
    signal = get_grid_signal(abs_chg, symmetry, proj_range)
    
    # 转换信号为整数 1 或 0
    signal_int = 1 if signal else 0
    
    print("-" * 40)
    print(f"Calculation Results for {original_date_str}:")
    print(f"  Reference Price: {ref_price}")
    print(f"  High-q0.8:       {high_q}")
    print(f"  Low-q0.2:        {low_q}")
    print(f"  pred_chg%:       {pred_chg}%")
    print(f"  > Symmetry:      {symmetry:.4f}")
    print(f"  > Proj Range:    {proj_range:.4f}%")
    print(f"  > Grid Signal:   {signal_int} (1=ON, 0=OFF)")
    print("-" * 40)
    if signal_int == 1:
        print("ON")
        print("-" * 40)
    
    # 6. 回写数据
    df.at[target_idx, 'symmetry'] = symmetry
    df.at[target_idx, 'proj_range'] = proj_range
    df.at[target_idx, 'grid_signal'] = signal_int
    
    # 保存 CSV
    try:
        df.to_csv(args.data_path, index=False)
        print(f"Successfully updated {args.data_path}")
    except Exception as e:
        print(f"Error saving file: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()