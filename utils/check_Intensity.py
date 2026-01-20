import pandas as pd
import argparse
import sys
from datetime import datetime

def get_args():
    parser = argparse.ArgumentParser(description="Calculate Intraday Intensity for Time-Based Exit")
    parser.add_argument(
        "--data_path",
        type=str,
        default="/content/data/BTC-USD.csv",
        help="Path to the OHLCV CSV file.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.3,
        help="Rebound/Correction ratio threshold for exit. Default: 0.3",
    )
    parser.add_argument(
        "--date",
        type=str,
        required=True,
        help="Target date (e.g. 2026-1-19). Code will auto-parse various formats.",
    )
    return parser.parse_args()

def calculate_intensity(row, threshold):
    """
    计算给定行的 Intensity 并判断是否需要平仓
    """
    open_price = float(row['Open'])
    high_price = float(row['High'])
    low_price = float(row['Low'])
    close_price = float(row['Close']) 
    
    # === 场景 A: 假设当前持仓是 LONG (做多) ===
    total_drop = open_price - low_price
    rebound = close_price - low_price
    
    if total_drop <= 0:
        long_intensity = 1.0 # 没跌过，视为强势
    else:
        long_intensity = rebound / total_drop

    long_action = "HOLD"
    if long_intensity < threshold:
        long_action = "CLOSE_IMMEDIATELY"

    # === 场景 B: 假设当前持仓是 SHORT (做空) ===
    total_pump = high_price - open_price
    correction = high_price - close_price
    
    if total_pump <= 0:
        short_intensity = 1.0 # 没涨过，视为弱势（对空头有利）
    else:
        short_intensity = correction / total_pump

    short_action = "HOLD"
    if short_intensity < threshold:
        short_action = "CLOSE_IMMEDIATELY"

    return {
        "LONG": {"ratio": long_intensity, "action": long_action},
        "SHORT": {"ratio": short_intensity, "action": short_action}
    }

def main():
    args = get_args()
    
    # 1. 读取数据
    try:
        df = pd.read_csv(args.data_path)
    except FileNotFoundError:
        print(f"Error: File not found at {args.data_path}")
        sys.exit(1)
        
    # 2. 智能日期匹配 (修改部分)
    try:
        # 将输入的参数转为 datetime 对象
        target_dt = pd.to_datetime(args.date)
        
        # 将 CSV 中的 Date 列也转为 datetime 对象 (处理可能存在的格式不统一)
        # errors='coerce' 会将无法解析的日期设为 NaT
        df['Date_dt'] = pd.to_datetime(df['Date'], errors='coerce')
        
        # 筛选 (比较 datetime 对象，忽略 2026-1-19 和 2026-01-19 的字符串差异)
        row = df[df['Date_dt'] == target_dt]
        
    except Exception as e:
        print(f"Error parsing dates: {e}")
        sys.exit(1)
    
    if row.empty:
        print(f"Error: Date {args.date} not found in {args.data_path}")
        print("Tip: Here are the first 3 dates in your CSV for reference:")
        print(df['Date'].head(3).tolist())
        sys.exit(1)
        
    # 取第一行
    target_row = row.iloc[0]
    
    # 3. 计算 Intensity
    result = calculate_intensity(target_row, args.threshold)
    
    # 4. 输出 Log
    # print("-" * 50)
    print(f"Intensity Check for Date: {target_row['Date']}")
    # print(f"Price Info: O={target_row['Open']}, H={target_row['High']}, L={target_row['Low']}, C(Current)={target_row['Close']}")
    print(f"Threshold: {args.threshold}")
    # print("-" * 30)
    print('')
    
    # LONG Output
    l_res = result['LONG']
    print(f"Direction: LONG")
    print(f"  Rebound Ratio (Intensity): {l_res['ratio']:.4f}")
    print(f"  Decision: {l_res['action']}")
    # if l_res['action'] == "CLOSE_IMMEDIATELY":
    #     print(f"  -> Reason: Price is sticking to the bottom. Rebound ({l_res['ratio']:.4f}) < Threshold ({args.threshold}).")
    # else:
    #     print(f"  -> Reason: Sufficient rebound or profitable.")
        
    print("-" * 30)
    
    # SHORT Output
    s_res = result['SHORT']
    print(f"Direction: SHORT")
    print(f"  Correction Ratio (Intensity): {s_res['ratio']:.4f}")
    print(f"  Decision: {s_res['action']}")
    # if s_res['action'] == "CLOSE_IMMEDIATELY":
    #     print(f"  -> Reason: Price is sticking to the top. Correction ({s_res['ratio']:.4f}) < Threshold ({args.threshold}).")
    # else:
    #     print(f"  -> Reason: Sufficient correction or profitable.")
        
    # print("-" * 50)

if __name__ == "__main__":
    main()