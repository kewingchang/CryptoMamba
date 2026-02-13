import argparse
import subprocess
import os
import re
import pandas as pd
from datetime import datetime, timedelta
import sys

def parse_args():
    parser = argparse.ArgumentParser(description="Automate trading inference and log parsing.")
    
    # 1. 定义命令行参数及默认值
    parser.add_argument('--start_date', type=str, default='2025-12-1', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end_date', type=str, default='2026-1-5', help='End date (YYYY-MM-DD)')
    parser.add_argument('--log', type=str, default='/content/data/log_pred.txt', help='Log file path')

    # Models & Thresholds
    parser.add_argument("--long_model", required=True, help="Path to Long LightGBM model")
    parser.add_argument("--short_model", required=True, help="Path to Short LightGBM model")
    parser.add_argument("--long_threshold", type=float, required=True, help="Threshold for Long (Prob > Th)")
    parser.add_argument("--short_threshold", type=float, required=True, help="Threshold for Short (Prob < Th)")
    
    # Data
    parser.add_argument("--data_path", required=True, help="Path to input features CSV")
    parser.add_argument("--pred_path", required=True, help="Path to output prediction CSV")

    return parser.parse_args()

def run_inference(args):
    """
    运行推理循环
    """
    start = datetime.strptime(args.start_date, "%Y-%m-%d")
    end = datetime.strptime(args.end_date, "%Y-%m-%d")
    
    # 确保日志目录存在
    log_dir = os.path.dirname(args.log)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # 如果需要清理日志
    # if args.clear_log and os.path.exists(args.log):
    #     os.remove(args.log)

    current_date = start
    while current_date <= end:
        date_str = current_date.strftime("%Y-%m-%d")
        print(f"[*] Running inference for date: {date_str}...")
        
        # 2. 构建并执行命令
        # 注意：这里使用了 f-string 构造命令，并将 output 重定向到日志文件
        cmd = (
            f"python scripts/one_day_pred.py "
            f"--long_model {args.long_model} "
            f"--long_threshold {args.long_threshold} "
            f"--short_model {args.short_model} "
            f"--short_threshold {args.short_threshold} "
            f"--pred_path {args.pred_path} "
            f"--data_path {args.data_path} "
            f"--date {date_str} >> {args.log} 2>&1"
        )
        
        try:
            subprocess.run(cmd, shell=True, check=True)
        except subprocess.CalledProcessError as e:
            print(f"[!] Error running inference for {date_str}: {e}")
            # 可以选择 continue 或 exit，这里选择继续
        
        current_date += timedelta(days=1)
    
    print("[*] Inference loop completed.")


def main():
    args = parse_args()
    # 运行循环训练/推理
    run_inference(args)
    
if __name__ == "__main__":
    main()
