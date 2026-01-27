import pandas as pd
import numpy as np
import argparse
import sys
import os

def get_grid_signal(abs_chg, sym, rng):
    """
    核心策略逻辑: 黄金网格过滤器 (预测端)
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

def calculate_ground_truth(df, target_idx):
    """
    计算实际是否震荡 (Ground Truth)
    前提: 必须有 Open, High, Low, Close 数据
    """
    req_cols = ['Open', 'High', 'Low', 'Close']
    for col in req_cols:
        if col not in df.columns or pd.isna(df.at[target_idx, col]):
            return None, None # 数据不全，无法计算

    try:
        open_val = float(df.at[target_idx, 'Open'])
        high_val = float(df.at[target_idx, 'High'])
        low_val = float(df.at[target_idx, 'Low'])
        close_val = float(df.at[target_idx, 'Close'])
        
        body_size = abs(close_val - open_val)
        # 加上极小值防止除以0
        total_range = abs(high_val - low_val)
        if total_range == 0:
            total_range = 1e-9

        real_body_ratio = body_size / total_range
        
        # 判定逻辑: 实体占比 < 0.5 视为震荡
        is_choppy = 1 if real_body_ratio < 0.5 else 0
        
        return real_body_ratio, is_choppy
        
    except Exception as e:
        print(f"Error calculating ground truth: {e}")
        return None, None

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
        # 统一转换为 datetime 对象进行比较
        target_dt = pd.to_datetime(target_date_input).normalize()
        file_dates = pd.to_datetime(df['Date'], errors='coerce').dt.normalize()
        matches = df.index[file_dates == target_dt].tolist()
        
    except Exception as e:
        print(f"Error parsing dates: {e}")
        sys.exit(1)
    
    if not matches:
        print(f"Info: Date '{target_date_input}' not found in {args.data_path}.")
        sys.exit(0)
        
    target_idx = matches[0]
    original_date_str = str(df.at[target_idx, 'Date'])
    print(f"Found record for '{target_date_input}' at index {target_idx}.")
    
    # 3. 准备数据 (Signal 计算用)
    try:
        req_cols = ['High-q0.8', 'Low-q0.2', 'pred_chg%']
        for col in req_cols:
            if col not in df.columns:
                print(f"Error: Column '{col}' missing in CSV.")
                sys.exit(1)
                
        high_q = float(df.at[target_idx, 'High-q0.8'])
        low_q = float(df.at[target_idx, 'Low-q0.2'])
        pred_chg = float(df.at[target_idx, 'pred_chg%'])
        
        ref_price = get_reference_price(df, target_idx, original_date_str)
        
        if ref_price is None:
            print(f"Error: Reference price missing for {original_date_str}.")
            sys.exit(1)
            
    except ValueError as e:
        print(f"Error parsing numerical data: {e}")
        sys.exit(1)
        
    # 4. 计算预测信号 (Signal)
    up_space = high_q - ref_price
    down_space = ref_price - low_q
    
    numerator = min(up_space, down_space)
    denominator = max(up_space, down_space)
    if denominator == 0:
        symmetry = 0.0
    else:
        symmetry = numerator / denominator
    
    proj_range = (high_q - low_q) / ref_price * 100
    
    abs_chg = abs(pred_chg)
    signal = get_grid_signal(abs_chg, symmetry, proj_range)
    signal_int = 1 if signal else 0
    
    # 回写预测数据
    df.at[target_idx, 'symmetry'] = symmetry
    df.at[target_idx, 'proj_range'] = proj_range
    df.at[target_idx, 'grid_signal'] = signal_int
    
    print("-" * 50)
    print(f"PREDICTION Results for {original_date_str}:")
    print(f"  High-q0.8:       {high_q}")
    print(f"  Low-q0.2:        {low_q}")
    print(f"  pred_chg%:       {pred_chg}%")
    print(f"  > Symmetry:      {symmetry:.4f}")
    print(f"  > Proj Range:    {proj_range:.4f}%")
    print(f"  > Grid Signal:   {signal_int} (1=ON, 0=OFF)")
    if signal_int == 1:
        print("-" * 50)
        print("ON")
    
    # 5. 计算 Ground Truth (如果有 OHLC)
    real_body_ratio, is_actual = calculate_ground_truth(df, target_idx)
    
    if is_actual is not None:
        # 回写实际数据
        df.at[target_idx, 'real_body_ratio'] = real_body_ratio
        df.at[target_idx, 'is_actual_choppy'] = is_actual
        
        print("-" * 50)
        print(f"ACTUAL (Backtest) Results:")
        print(f"  Open:            {df.at[target_idx, 'Open']}")
        print(f"  Close:           {df.at[target_idx, 'Close']}")
        print(f"  High:            {df.at[target_idx, 'High']}")
        print(f"  Low:             {df.at[target_idx, 'Low']}")
        print(f"  > Body Ratio:    {real_body_ratio:.4f}")
        print(f"  > Actual Choppy: {is_actual} (1=YES, 0=NO)")
        
        # 简单打印是否预测正确
        if signal_int == 1:
            res_str = "SUCCESS (TP)" if is_actual == 1 else "FAIL (FP)"
            print(f"  > Result:        {res_str}")
        else:
            print(f"  > Result:        No Signal (Skipped)")
    else:
        print("-" * 50)
        # print("Info: OHLC data missing or incomplete. Skipping Actual Choppy calculation.")

    # print("-" * 50)

    # 6. 保存 CSV
    try:
        df.to_csv(args.data_path, index=False)
        print(f"Successfully updated {args.data_path}")
    except Exception as e:
        print(f"Error saving file: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
