import pandas as pd
import numpy as np
import lightgbm as lgb
import argparse
import sys
import os
import datetime
import csv


# ==========================================
# 核心逻辑
# ==========================================
def load_data_row(data_path, target_date):
    """
    加载指定日期(的前一天)的数据行
    """
    input_date = target_date - pd.Timedelta(days=1)
    
    try:
        # 预读取 Header 检查 Date 列
        header = pd.read_csv(data_path, nrows=0)
        if 'Date' not in header.columns:
            print("[Fatal] Input CSV missing 'Date' column.")
            sys.exit(1)
            
        # 读取数据 (解析日期)
        df = pd.read_csv(data_path)
        df['Date'] = pd.to_datetime(df['Date'])
        
        # 查找 Input Date
        row = df[df['Date'] == input_date]
        
        if row.empty:
            print(f"[Fatal] Input data for {input_date.date()} (T-1) not found in {data_path}!")
            sys.exit(1)
            
        return row
    except Exception as e:
        print(f"[Fatal] Data loading error: {e}")
        sys.exit(1)

def run_inference(model_path, data_row, direction, threshold):
    """
    单个模型的推理逻辑
    """
    try:
        if not os.path.exists(model_path):
            print(f"[Fatal] Model file not found: {model_path}")
            sys.exit(1)
            
        # 1. 加载模型
        bst = lgb.Booster(model_file=model_path)
        
        # 2. 获取该模型需要的特征
        needed_features = bst.feature_name()
        
        # 3. 检查特征是否存在
        missing = [f for f in needed_features if f not in data_row.columns]
        if missing:
            print(f"[Fatal] Model [{direction}] missing features in CSV: {missing}")
            sys.exit(1)
            
        # 4. 提取特征数据
        X_input = data_row[needed_features]
        
        # 5. 预测
        prob = bst.predict(X_input)[0]
        
        # 6. 判断信号
        is_signal = False
        if direction == 'Long':
            # Long 逻辑: 概率 > 阈值
            if prob > threshold:
                is_signal = True
        else:
            # Short 逻辑: 概率 < 阈值
            if prob < threshold:
                is_signal = True
                
        return prob, is_signal
        
    except Exception as e:
        print(f"[Fatal] Inference failed for {direction}: {e}")
        sys.exit(1)

# ==========================================
# 主程序
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="Combined Long/Short Prediction")

    # Models & Thresholds
    parser.add_argument("--long_model", required=True, help="Path to Long LightGBM model")
    parser.add_argument("--short_model", required=True, help="Path to Short LightGBM model")
    parser.add_argument("--long_threshold", type=float, required=True, help="Threshold for Long (Prob > Th)")
    parser.add_argument("--short_threshold", type=float, required=True, help="Threshold for Short (Prob < Th)")
    
    # Data
    parser.add_argument("--data_path", required=True, help="Path to input features CSV")
    parser.add_argument("--pred_path", required=True, help="Path to output prediction CSV")
    parser.add_argument("--date", required=True, help="Target prediction date (YYYY-MM-DD)")
    
    args = parser.parse_args()

    # 1. 时间处理
    try:
        target_date_obj = pd.to_datetime(args.date)
    except:
        print("[Fatal] Invalid date format.")
        sys.exit(1)

    print("=" * 60)
    print(f"Prediction Job: {target_date_obj.date()}")
    print("=" * 60)

    # 2. 获取数据行
    # 只需读取一次 CSV，包含所有可能用到的特征
    data_row = load_data_row(args.data_path, target_date_obj)

    # 3. 分别运行推理
    # Long Model
    prob_long, sig_long = run_inference(args.long_model, data_row, 'Long', args.long_threshold)
    
    # Short Model
    prob_short, sig_short = run_inference(args.short_model, data_row, 'Short', args.short_threshold)

    # 4. 最终决策逻辑 (Conflict Handling)
    final_decision = "WAIT"
    final_signal_code = 0 # 1=Long, -1=Short, 0=Wait
    reason = "No Signal"
    
    if sig_long and not sig_short:
        final_decision = "LONG"
        final_signal_code = 1
        reason = "Long Signal Triggered"
        
    elif sig_short and not sig_long:
        final_decision = "SHORT"
        final_signal_code = -1
        reason = "Short Signal Triggered"
        
    elif sig_long and sig_short:
        # 冲突：既看多又看空
        final_decision = "WAIT"
        final_signal_code = 0
        reason = "CONFLICT (Double Signal) - High Volatility Risk"
        
    else:
        # 都没信号
        final_decision = "WAIT"
        final_signal_code = 0
        reason = "Both Models Silent (Flat)"

    # 5. 打印日志
    print(f"{'Model':<10} | {'Prob':<10} | {'Thresh':<10} | {'Signal'}")
    print("-" * 50)
    print(f"{'Long':<10} | {prob_long:.5f}    | > {args.long_threshold:<8} | {'YES' if sig_long else 'NO'}")
    print(f"{'Short':<10} | {prob_short:.5f}    | < {args.short_threshold:<8} | {'YES' if sig_short else 'NO'}")
    print("-" * 50)
    print(f"FINAL DECISION : [{final_decision}]")
    print(f"Reason         : {reason}")
    print("=" * 60)

    # 6. 保存结果到 CSV
    # 准备数据
    row_data = {
        'Date': target_date_obj,
        'Long_Prob': round(prob_long, 5),
        'Short_Prob': round(prob_short, 5),
        'Long_Signal_Raw': 1 if sig_long else 0,
        'Short_Signal_Raw': 1 if sig_short else 0,
        'Final_Decision': final_decision,
        'Final_Signal_Code': final_signal_code,
        'Comment': reason
    }
    
    # 读/写 CSV
    if os.path.exists(args.pred_path):
        try:
            res_df = pd.read_csv(args.pred_path)
            res_df['Date'] = pd.to_datetime(res_df['Date'])
        except:
            res_df = pd.DataFrame()
    else:
        res_df = pd.DataFrame()
        
    # Upsert 逻辑
    if not res_df.empty and (res_df['Date'] == target_date_obj).any():
        # 更新
        idx = res_df.index[res_df['Date'] == target_date_obj][0]
        for k, v in row_data.items():
            res_df.at[idx, k] = v
        print(f"[Info] Updated existing record for {target_date_obj.date()}")
    else:
        # 新增
        res_df = pd.concat([res_df, pd.DataFrame([row_data])], ignore_index=True)
        print(f"[Info] Created new record for {target_date_obj.date()}")
        
    # 排序与保存
    res_df = res_df.sort_values('Date', ascending=False)
    try:
        res_df.to_csv(args.pred_path, index=False)
        print(f"[Success] Saved result to {args.pred_path}")
    except Exception as e:
        print(f"[Error] Failed to save CSV: {e}")

if __name__ == "__main__":
    main()
