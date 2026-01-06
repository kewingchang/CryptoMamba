import argparse
import subprocess
import os
import re
import pandas as pd
from datetime import datetime, timedelta
import sys

def parse_args():
    parser = argparse.ArgumentParser(description="Automate trading inference and parse High Quantile logs.")
    
    # 1. 命令行参数定义
    parser.add_argument('--data_path', type=str, default='BTC-USD.csv', help='Input CSV file path')
    parser.add_argument('--start_date', type=str, default='2025-12-1', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end_date', type=str, default='2026-1-5', help='End date (YYYY-MM-DD)')
    parser.add_argument('--log', type=str, default='/content/data/log_pred.txt', help='Log file path')
    parser.add_argument('--config', type=str, default='cmamba_btc', help='Config name')
    parser.add_argument('--ckpt_path', type=str, default='./checkpoints/BTC_251230.ckpt', help='Checkpoint path')
    parser.add_argument('--risk', type=float, default=2.0, help='Risk parameter')
    parser.add_argument('--model', type=str, default='v2_btc_high', help='mode name')
    # 清理旧日志选项
    parser.add_argument('--clear_log', action='store_true', help='Clear log file before starting')

    return parser.parse_args()

def run_inference(args):
    """
    2. 循环执行推理命令
    """
    start = datetime.strptime(args.start_date, "%Y-%m-%d")
    end = datetime.strptime(args.end_date, "%Y-%m-%d")
    
    # 确保日志目录存在
    log_dir = os.path.dirname(args.log)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # 如果需要清理日志
    if args.clear_log and os.path.exists(args.log):
        os.remove(args.log)

    current_date = start
    while current_date <= end:
        date_str = current_date.strftime("%Y-%m-%d")
        print(f"[*] Running inference for date: {date_str}...")
        
        # 构造命令
        cmd = (
            f"python scripts/gotrade.py "
            f"--config {args.config} "
            f"--model {args.model} "
            f"--ckpt_path {args.ckpt_path} "
            f"--data_path {args.data_path} "
            f"--risk {args.risk} "
            f"--date {date_str} >> {args.log} 2>&1"
        )
        
        try:
            subprocess.run(cmd, shell=True, check=True)
        except subprocess.CalledProcessError as e:
            print(f"[!] Error running inference for {date_str}: {e}")
        
        current_date += timedelta(days=1)
    
    print("[*] Inference loop completed.")

def parse_log_and_extract_data(log_path):
    """
    3. 解析日志文件提取 q 值
    """
    if not os.path.exists(log_path):
        print(f"[!] Log file {log_path} not found.")
        return {}

    with open(log_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 按 "Prediction date:" 分块
    blocks = re.split(r'(Prediction date:\s+\d{4}-\d{2}-\d{2})', content)
    
    extracted_data = {}

    for i in range(1, len(blocks), 2):
        header = blocks[i]
        body = blocks[i+1] if i+1 < len(blocks) else ""
        full_block = header + body

        # --- 提取 Date ---
        date_match = re.search(r'Prediction date:\s+(\d{4}-\d{2}-\d{2})', full_block)
        if not date_match:
            continue
        p_date = date_match.group(1)

        # --- 提取 q 值 ---
        # 逻辑：查找 q=X... : 后面的数字
        # 正则解释：
        # q=0.05       匹配字面量
        # .*?          非贪婪匹配中间的文本 (如 "(Low Stop)")
        # :\s+         匹配冒号和空格
        # ([\d.]+)     捕获数字和小数点
        
        # q=0.05
        q05_match = re.search(r'q=0\.95.*?: \s*([\d.]+)', full_block)
        val_05 = float(q05_match.group(1)) if q05_match else None

        # q=0.2 (注意这里可能有空格，或者直接跟冒号)
        q02_match = re.search(r'q=0\.8\s.*?: \s*([\d.]+)', full_block)
        val_02 = float(q02_match.group(1)) if q02_match else None

        # q=0.4
        q04_match = re.search(r'q=0\.6\s.*?: \s*([\d.]+)', full_block)
        val_04 = float(q04_match.group(1)) if q04_match else None

        # q=0.6
        q06_match = re.search(r'q=0\.4.*?: \s*([\d.]+)', full_block)
        val_06 = float(q06_match.group(1)) if q06_match else None

        extracted_data[p_date] = {
            'High-q0.95': val_05,
            'High-q0.8': val_02,
            'High-q0.6': val_04,
            'High-q0.4': val_06
        }

    return extracted_data

def update_csv_file(args, data_map):
    """
    更新 CSV 文件，添加 Low-q 列
    """
    if not os.path.exists(args.data_path):
        print(f"[!] Input CSV {args.data_path} not found.")
        return

    print(f"[*] Updating {args.data_path} with parsed quantile data...")
    try:
        df = pd.read_csv(args.data_path)
        
        # 处理日期列
        if 'Date' not in df.columns:
            # 简单查找
            date_col = next((c for c in df.columns if 'date' in c.lower()), None)
            if date_col:
                df.rename(columns={date_col: 'Date'}, inplace=True)
            else:
                print("[!] Could not find a 'Date' column in CSV.")
                return

        df['Date_dt'] = pd.to_datetime(df['Date'])
        
        # 初始化新列
        new_cols = ['High-q0.95', 'High-q0.8', 'High-q0.6', 'High-q0.4']
        for col in new_cols:
            if col not in df.columns:
                df[col] = None

        count = 0
        for date_str, info in data_map.items():
            target_date = pd.to_datetime(date_str)
            mask = df['Date_dt'] == target_date
            
            if mask.any():
                idx = df.index[mask]
                df.loc[idx, 'High-q0.95'] = info['High-q0.95']
                df.loc[idx, 'High-q0.8'] = info['High-q0.8']
                df.loc[idx, 'High-q0.6'] = info['High-q0.6']
                df.loc[idx, 'High-q0.4'] = info['High-q0.4']
                count += 1
        
        # 清理辅助列并保存
        df.drop(columns=['Date_dt'], inplace=True)
        df.to_csv(args.data_path, index=False)
        print(f"[*] Successfully updated {count} records in {args.data_path}.")

    except Exception as e:
        print(f"[!] Error updating CSV: {e}")

def main():
    args = parse_args()
    
    # 步骤 2: 运行
    run_inference(args)
    
    # 步骤 3: 解析
    data_map = parse_log_and_extract_data(args.log)
    
    if not data_map:
        print("[!] No data extracted from logs. Check log format.")
        return

    # 步骤 3: 写入
    update_csv_file(args, data_map)

if __name__ == "__main__":
    main()