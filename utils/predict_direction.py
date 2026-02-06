import pandas as pd
import numpy as np
import argparse
import sys
import os

def calculate_indicators(row, ref_price):
    """
    计算用于判断方向的关键指标
    """
    # 1. Tail Ratio (尾部比率)
    # High-q0.95 距离 vs Low-q0.05 距离
    high_tail = abs(row['High-q0.95'] - ref_price)
    low_tail  = abs(ref_price - row['Low-q0.05'])
    # 防止除以0
    tail_ratio = high_tail / low_tail if low_tail > 1e-9 else 1.0
    
    # 2. Skew Ratio (整体偏度比率)
    # 所有上方分位数距离之和 vs 所有下方分位数距离之和
    up_power = 0
    down_power = 0
    for q in ['High-q0.95', 'High-q0.8', 'High-q0.6', 'High-q0.4']:
        up_power += max(0, row[q] - ref_price)
    for q in ['Low-q0.05', 'Low-q0.2', 'Low-q0.4', 'Low-q0.6']:
        down_power += max(0, ref_price - row[q])
        
    skew_ratio = up_power / down_power if down_power > 1e-9 else 1.0
    
    return tail_ratio, skew_ratio

def get_direction_signal(tail_ratio, skew_ratio, symbol):
    """
    根据统计数据判断方向
    返回: (Direction, Confidence_Level, Reason)
    """
    
    # ============================================
    # BTC 策略 (基于历史统计)
    # ============================================
    if symbol == 'BTC':
        # 1. Tail Ratio 是 BTC 最有效的指标
        if tail_ratio > 1.2:
            return "LONG", "High", "Tail_Ratio > 1.2 (WinRate 64%)"
        
        if tail_ratio < 0.8:
            return "SHORT", "Medium", "Tail_Ratio < 0.8 (WinRate 58% + High Risk Reward)"
            
        # 2. Skew Ratio 辅助
        if skew_ratio > 1.5:
            return "LONG", "High", "Skew_Ratio > 1.5 (WinRate 64%)"
            
        # 3. 弱信号区 (利用原始预测稍微倾向一下)
        # 这里比较中性，可以说 NEUTRAL，或者跟随 skew 的微弱方向
        if skew_ratio < 0.8:
            return "SHORT", "Low", "Skew_Ratio < 0.8 (Weak Signal)"
            
        return "NEUTRAL", "Low", "No strong statistical signal"

    # ============================================
    # ETH 策略 (基于历史统计 - 包含陷阱逻辑)
    # ============================================
    elif symbol == 'ETH':
        # 1. 极端的 Skew 是最强的信号
        if skew_ratio < 0.6:
            return "SHORT", "Very High", "Skew_Ratio < 0.6 (WinRate 75%)"
            
        if skew_ratio > 1.5:
            return "LONG", "Very High", "Skew_Ratio > 1.5 (WinRate 77%)"
            
        # 2. *** ETH 特有的多头陷阱 (Bull Trap) ***
        # 统计显示: 1.0-1.2 区间虽然看似看涨，实际跌幅巨大且胜率低
        if 1.0 <= skew_ratio <= 1.2:
            return "SHORT", "High", "Bull Trap: Skew 1.0-1.2 (71% Drop Rate)"
            
        # 3. Tail Ratio 辅助
        if tail_ratio > 1.2:
            return "LONG", "Medium", "Tail_Ratio > 1.2 (WinRate 64%)"
            
        # 4. Skew 中等区间
        if 1.2 < skew_ratio <= 1.5:
            return "LONG", "Medium", "Skew 1.2-1.5 (WinRate 62%)"
            
        return "NEUTRAL", "Low", "No strong statistical signal"

    return "NEUTRAL", "Low", "Unknown Symbol"

def get_reference_price(df, target_idx, date_str):
    # 优先 Open，其次 last_close
    if 'Open' in df.columns:
        val = df.at[target_idx, 'Open']
        if pd.notna(val) and float(val) > 0:
            return float(val)
    if 'last_close' in df.columns:
        val = df.at[target_idx, 'last_close']
        if pd.notna(val) and float(val) > 0:
            print(f"[{date_str}] 'Open' missing. Using 'last_close' ({val}).")
            return float(val)
    return None

def main():
    parser = argparse.ArgumentParser(description="Predict Daily Direction based on Quantile Stats.")
    parser.add_argument('--data_path', type=str, default='/content/data/Pred-BTC-USD.csv')
    parser.add_argument('--date', type=str, required=True)
    parser.add_argument('--symbol', type=str, default='BTC', choices=['BTC', 'ETH'])
    
    args = parser.parse_args()
    
    if not os.path.exists(args.data_path):
        print(f"Error: File not found at {args.data_path}")
        sys.exit(1)
    
    try:
        df = pd.read_csv(args.data_path)
    except:
        sys.exit(1)
        
    # 日期匹配
    target_date_input = args.date.strip()
    try:
        t_dt = pd.to_datetime(target_date_input).normalize()
        f_dts = pd.to_datetime(df['Date'], errors='coerce').dt.normalize()
        matches = df.index[f_dts == t_dt].tolist()
    except Exception as e:
        print(f"Date Error: {e}")
        sys.exit(1)
        
    if not matches:
        print(f"Info: Date '{target_date_input}' not found.")
        sys.exit(0)
        
    idx = matches[0]
    date_str = str(df.at[idx, 'Date'])
    
    # 准备数据
    try:
        # 检查必要的列
        req_cols = ['High-q0.95', 'Low-q0.05', 'High-q0.8', 'Low-q0.2']
        for c in req_cols:
            if c not in df.columns:
                print(f"Missing column: {c}")
                sys.exit(1)
        
        ref_price = get_reference_price(df, idx, date_str)
        if ref_price is None:
            print("Error: No reference price.")
            sys.exit(1)
            
        # 计算指标
        tail_ratio, skew_ratio = calculate_indicators(df.iloc[idx], ref_price)
        
        # 获取方向预测
        direction, confidence, reason = get_direction_signal(tail_ratio, skew_ratio, args.symbol)
        
        print("="*60)
        print(f"DIRECTION PREDICTION for {date_str} [{args.symbol}]")
        print("="*60)
        print(f"Indicators:")
        print(f"  Ref Price:  {ref_price}")
        print(f"  Tail Ratio: {tail_ratio:.4f} (High-q95 / Low-q05)")
        print(f"  Skew Ratio: {skew_ratio:.4f} (Total Up / Total Down)")
        print("-" * 60)
        print(f"PREDICTION:   [{direction}]")
        print(f"CONFIDENCE:   {confidence}")
        print(f"LOGIC:        {reason}")
        print("="*60)
        
        # 回写结果到 CSV
        df.at[idx, 'stat_direction'] = direction
        df.at[idx, 'stat_confidence'] = confidence
        df.at[idx, 'tail_ratio'] = tail_ratio
        df.at[idx, 'skew_ratio'] = skew_ratio
        
        df.to_csv(args.data_path, index=False)
        print(f"Result saved to {args.data_path}")
        
    except Exception as e:
        print(f"Runtime Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()