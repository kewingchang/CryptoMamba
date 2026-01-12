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
            f"--model {args.model}"
            f"--ckpt_path {args.ckpt_path} "
            f"--data_path {args.data_path} "
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

def parse_log_and_extract_data(log_path):
    """
    3. 解析日志文件提取信息
    """
    if not os.path.exists(log_path):
        print(f"[!] Log file {log_path} not found.")
        return {}

    with open(log_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 将日志按 "Prediction date:" 分割成块，每一块代表一天的预测
    # 使用正则表达式分割，保留分割符以便知道每一块的开始
    blocks = re.split(r'(Prediction date:\s+\d{4}-\d{2}-\d{2})', content)
    
    extracted_data = {}

    # blocks[0] 通常是开头的一些杂乱信息，从 1 开始遍历
    # split 后，blocks[i] 是 "Prediction date: ...", blocks[i+1] 是该日期的后续内容
    for i in range(1, len(blocks), 2):
        header = blocks[i]
        body = blocks[i+1] if i+1 < len(blocks) else ""
        full_block = header + body

        # --- 提取 Date ---
        date_match = re.search(r'Prediction date:\s+(\d{4}-\d{2}-\d{2})', full_block)
        if not date_match:
            continue
        p_date = date_match.group(1)

        # --- 提取 Prediction Close ---
        close_match = re.search(r'Prediction Close:\s+([\d.]+)', full_block)
        p_close = float(close_match.group(1)) if close_match else None

        # --- 提取 Predicted Change ---
        # 提取数字，去掉百分号
        change_match = re.search(r'Predicted Change:\s+([-\d.]+)%', full_block)
        p_change = float(change_match.group(1)) if change_match else None

        # --- 提取 Decision 和 Strength ---
        # 格式可能为:
        # [DECISION]: WAIT / NO TRADE
        # [DECISION]: OPEN LONG (STRONG SIGNAL)
        # [DECISION]: OPEN SHORT (WEAK SIGNAL - SNIPER MODE)
        decision_match = re.search(r'\[DECISION\]:\s+(.*)', full_block)
        decision_text = decision_match.group(1).strip() if decision_match else ""
        
        strength = "Wait" # 默认
        if "WAIT" in decision_text:
            strength = "Wait"
        elif "STRONG" in decision_text:
            strength = "Strong"
        elif "WEAK" in decision_text:
            strength = "Weak"

        # --- 提取 Direction (LONG/SHORT) ---
        # 方向通常在 Decision 下方，或者就在 Decision 文本里。
        # 既然日志中有单独的一行 LONG 或 SHORT，我们优先找单独行
        # 也要结合 strength 判断，如果是 Wait，方向通常不重要或者没给，
        # 但为了完整性，我们尝试提取。
        direction_code = None
        
        # 查找独立的 LONG 或 SHORT 行
        dir_match = re.search(r'\n(LONG|SHORT)\s*\n', full_block)
        raw_direction = dir_match.group(1) if dir_match else None
        
        # 如果没找到单独行，尝试从 decision 文本中提取 (e.g. OPEN LONG)
        if not raw_direction:
            if "LONG" in decision_text:
                raw_direction = "LONG"
            elif "SHORT" in decision_text:
                raw_direction = "SHORT"
        
        if raw_direction == "LONG":
            direction_code = "B"
        elif raw_direction == "SHORT":
            direction_code = "S"
        else:
            # 如果是 Wait 且没找到方向，可以留空或者设为默认
            direction_code = None 

        extracted_data[p_date] = {
            'pred': p_close,
            'pred_chg': p_change,
            'direction': direction_code,
            'strength': strength
        }

    return extracted_data

def update_csv_file(args, data_map):
    """
    将提取的数据写入 CSV
    """
    if not os.path.exists(args.data_path):
        print(f"[!] Input CSV {args.data_path} not found.")
        return

    print(f"[*] Updating {args.data_path} with parsed data...")
    try:
        df = pd.read_csv(args.data_path)
        
        # 假设 CSV 中有一列叫 'Date'，将其转为 datetime 对象以便匹配
        # 如果你的日期列名不是 Date，请修改这里
        if 'Date' not in df.columns:
            # 尝试查找可能是日期的列
            date_col = None
            for col in df.columns:
                if 'date' in col.lower():
                    date_col = col
                    break
            if date_col:
                df.rename(columns={date_col: 'Date'}, inplace=True)
            else:
                print("[!] Could not find a 'Date' column in CSV.")
                return

        # 标准化日期格式
        df['Date_dt'] = pd.to_datetime(df['Date'])
        
        # ---------------------------------------------------------
        # 修改需求 2: 计算 change% = (Close - Open) / Open * 100
        # ---------------------------------------------------------
        if 'Open' in df.columns and 'Close' in df.columns:
            # 批量计算全量数据的 change%，保留4位小数（可选）
            df['change%'] = (df['Close'] - df['Open']) / df['Open'] * 100
        else:
            print("[!] 'Open' or 'Close' column missing, cannot calculate 'change%'")

        # ---------------------------------------------------------
        # 修改需求 1: 列名调整 pred_chg -> pred_chg%
        # ---------------------------------------------------------
        new_cols = ['pred', 'pred_chg%', 'direction', 'strength']
        for col in new_cols:
            if col not in df.columns:
                df[col] = None

        # 更新数据
        # 遍历 data_map 效率比遍历 dataframe 高
        count = 0
        for date_str, info in data_map.items():
            # 查找匹配日期的行索引
            target_date = pd.to_datetime(date_str)
            mask = df['Date_dt'] == target_date
            
            if mask.any():
                idx = df.index[mask]
                df.loc[idx, 'pred'] = info['pred']
                # 这里将解析出来的数值写入带 % 的新列名
                df.loc[idx, 'pred_chg%'] = info['pred_chg'] 
                df.loc[idx, 'direction'] = info['direction']
                df.loc[idx, 'strength'] = info['strength']
                count += 1
        
        # 删除辅助列
        df.drop(columns=['Date_dt'], inplace=True)
        
        # 保存文件
        df.to_csv(args.data_path, index=False)
        print(f"[*] Successfully updated {count} records in {args.data_path}.")

    except Exception as e:
        print(f"[!] Error updating CSV: {e}")

def main():
    args = parse_args()
    
    # 步骤 2: 运行循环训练/推理
    run_inference(args)
    
    # 步骤 3: 解析日志
    data_map = parse_log_and_extract_data(args.log)
    
    if not data_map:
        print("[!] No data extracted from logs.")
        return

    # 步骤 3 (写入): 更新 CSV
    update_csv_file(args, data_map)

if __name__ == "__main__":
    main()