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
    parser.add_argument('--clear_log', action='store_true', help='Clear log file before starting')
    parser.add_argument('--start_date', type=str, default='2025-12-1', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end_date', type=str, default='2026-1-5', help='End date (YYYY-MM-DD)')
    parser.add_argument('--log', type=str, default='/content/data/log_pred.txt', help='Log file path')
    
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
            f"python utils/calculate_grid_signal.py "
            f"--data_path {args.data_path} "
            f"--date {date_str} >> {args.log} 2>&1"
        )
        
        try:
            subprocess.run(cmd, shell=True, check=True)
        except subprocess.CalledProcessError as e:
            print(f"[!] Error running inference for {date_str}: {e}")
        
        current_date += timedelta(days=1)
    
    print("[*] Inference loop completed.")


def main():
    args = parse_args()
    # 运行
    run_inference(args)
    
if __name__ == "__main__":
    main()
