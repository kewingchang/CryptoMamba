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
    parser.add_argument('--data_path', type=str, default='BTC-USD.csv', help='Input CSV file path')
    parser.add_argument('--start_date', type=str, default='2025-12-1', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end_date', type=str, default='2026-1-5', help='End date (YYYY-MM-DD)')
    parser.add_argument('--log', type=str, default='/content/data/log_pred.txt', help='Log file path')
    parser.add_argument('--config', type=str, default='cmamba_btc', help='Config name')
    parser.add_argument('--ckpt_path', type=str, default='./checkpoints/BTC_251230.ckpt', help='Checkpoint path')
    parser.add_argument('--risk', type=float, default=2.0, help='Risk parameter')
    parser.add_argument( "--model", required=False, type=str, default='v2', help="Path to model config file.")
    parser.add_argument( "--symbol", required=True, type=str, default='BTC', help="The coin to process")
    parser.add_argument( "--pred_path", required=True, type=str, default='Pred-BTC-USD.csv', help="Path to save prediction")    
    # 添加一个标志来决定是否清理旧日志，默认追加
    parser.add_argument('--clear_log', action='store_true', help='Clear log file before starting')

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
    if args.clear_log and os.path.exists(args.log):
        os.remove(args.log)

    current_date = start
    while current_date <= end:
        date_str = current_date.strftime("%Y-%m-%d")
        print(f"[*] Running inference for date: {date_str}...")
        
        # 2. 构建并执行命令
        # 注意：这里使用了 f-string 构造命令，并将 output 重定向到日志文件
        cmd = (
            f"python scripts/gotrade.py "
            f"--config {args.config} "
            f"--model {args.model} "
            f"--ckpt_path {args.ckpt_path} "
            f"--data_path {args.data_path} "
            f"--pred_path {args.pred_path} "
            f"--symbol {args.symbol} "
            f"--risk {args.risk} "
            f"--paper_trading "
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
