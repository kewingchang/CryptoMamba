import pandas as pd
import numpy as np
import talib #pip install TA-Lib
import argparse
import sys

def process_patterns(input_file, output_file):
    print(f"Reading data from {input_file}...")
    try:
        df = pd.read_csv(input_file)
    except FileNotFoundError:
        print(f"Error: File {input_file} not found.")
        sys.exit(1)

    # ==========================================
    # 1. 基础检查与数据准备
    # ==========================================
    required_inputs = ['Open', 'High', 'Low', 'Close']
    
    fixed_features = [
        'Date', 'Open', 'High', 'Low', 'Close', 'Volume']

    # 检查必需列
    missing_inputs = [c for c in required_inputs if c not in df.columns]
    if missing_inputs:
        raise ValueError(f"CRITICAL ERROR: Missing columns required for calculation: {missing_inputs}")

    # 初始化输出 DataFrame
    available_fixed = [c for c in fixed_features if c in df.columns]
    df_out = df[available_fixed].copy()

    # 准备 numpy 数组
    op = df['Open'].astype(float).values
    hi = df['High'].astype(float).values
    lo = df['Low'].astype(float).values
    cl = df['Close'].astype(float).values

    # ==========================================
    # 2. 动态获取并计算所有 K 线形态
    # ==========================================
    print("Scanning TA-Lib for candlestick patterns...")
    
    all_talib_functions = dir(talib)
    pattern_functions = [func for func in all_talib_functions if func.startswith('CDL')]
    
    print(f"Found {len(pattern_functions)} candlestick patterns.")

    # 初始化计数器数组 (全0)
    bullish_counts = np.zeros(len(df))
    bearish_counts = np.zeros(len(df))
    
    count = 0
    for pattern_name in pattern_functions:
        func = getattr(talib, pattern_name)
        
        try:
            # 1. 计算原始值 (-100, 0, 100)
            result = func(op, hi, lo, cl)
            
            # 2. 缩放到 [-1, 1] 用于单独特征
            result_scaled = result / 100.0
            df_out[pattern_name] = result_scaled
            
            # 3. 统计汇总特征
            # 如果 result > 0 (100)，是看涨，bullish_count + 1
            bullish_counts += (result > 0).astype(int)
            
            # 如果 result < 0 (-100)，是看跌，bearish_count + 1
            # 注意这里我们统计的是“数量”，所以加 1，而不是减 1
            bearish_counts += (result < 0).astype(int)

            count += 1
            
        except Exception as e:
            print(f"Warning: Failed to calculate {pattern_name}: {e}")

    # ==========================================
    # 3. 添加汇总统计特征
    # ==========================================
    # 这两个特征的值域通常是 0, 1, 2, 3... 
    # 建议在模型训练时 Skip RevIN
    df_out['bullish_pattern_count'] = bullish_counts
    df_out['bearish_pattern_count'] = bearish_counts

    # 还可以加一个净情绪特征 (看涨数 - 看跌数)
    df_out['pattern_sentiment_score'] = bullish_counts - bearish_counts

    # ==========================================
    # 4. 保存结果
    # ==========================================
    # df_out.fillna(0, inplace=True)

    if output_file is None:
        output_file = input_file
        # output_file = input_file.replace('.csv', '_with_patterns_final.csv')

    print(f"Successfully generated {count} individual patterns.")
    print(f"Added summary features: bullish_pattern_count, bearish_pattern_count, pattern_sentiment_score")
    print(f"Saving to {output_file}...")
    df_out.to_csv(output_file, index=False)
    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate TA-Lib patterns and summary counts.')
    parser.add_argument('--filename', type=str, required=True, help='Input CSV file')
    args = parser.parse_args()

    process_patterns(args.filename, None)