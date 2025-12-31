# extract_metrics_dynamic.py
import os
import csv
import argparse
import re

def extract_additional_features(lines):
    pattern = r'<<<<<<<<<<<<>>>>>>>>>>>>'
    features = None
    found_first = False
    for line in lines:
        if pattern in line:
            if found_first: break
            found_first = True
        elif found_first:
            features = line.strip()
    return features

def extract_metrics(lines, start_marker, prefix_mapping):
    metrics = {}
    started = False
    in_table = False
    for line in lines:
        if start_marker in line:
            started = True
            continue
        if not started: continue
        if '┏' in line:
            in_table = True
            continue
        if in_table and '└' in line:
            break
        if in_table and '│' in line:
            parts = re.split(r'│\s*', line.strip('┃┡╇┩'))
            if len(parts) >= 3:
                key = parts[1].strip()
                value_str = parts[2].strip()
                if not key: continue
                
                # Prefix mapping (val/ -> train/)
                mapped_key = key
                for original_prefix, target_prefix in prefix_mapping.items():
                    if key.startswith(original_prefix):
                        mapped_key = key.replace(original_prefix, target_prefix, 1)
                        break
                try:
                    metrics[mapped_key] = float(value_str)
                except ValueError:
                    pass
    return metrics

def calculate_diffs(metrics_dict, prefix):
    # 自动查找该 prefix 下所有的 cover 指标
    # key 格式例如: test/q0.95_cover
    pattern = re.compile(rf"{prefix}/q([\d\.]+)_cover")
    
    updates = {}
    for key, val in metrics_dict.items():
        match = pattern.match(key)
        if match:
            q_val = float(match.group(1)) # 提取 0.95
            diff_key = f"{prefix}/q{q_val}_diff"
            # Diff = |Actual_Cover - Target_Quantile|
            updates[diff_key] = abs(val - q_val)
    
    metrics_dict.update(updates)
    return metrics_dict

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--output', default='output.csv')
    args = parser.parse_args()

    data_dir = args.data_dir
    rows = []
    
    # 动态收集所有列名
    all_keys = set()

    for subdir in sorted(os.listdir(data_dir)):
        subdir_path = os.path.join(data_dir, subdir)
        if not os.path.isdir(subdir_path): continue
        log_path = os.path.join(subdir_path, 'log.txt')
        if not os.path.exists(log_path): continue

        with open(log_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        add_feats = extract_additional_features(lines)
        
        # Extract
        test_metrics = calculate_diffs(extract_metrics(lines, 'Test Set Validate', {'test/': 'test/'}), 'test')
        val_metrics = calculate_diffs(extract_metrics(lines, 'Val Set Validate', {'val/': 'val/'}), 'val')
        train_metrics = calculate_diffs(extract_metrics(lines, 'Train Set Validate', {'val/': 'train/'}), 'train')

        row = {'dir-name': subdir, 'additional_features': add_feats}
        row.update(test_metrics)
        row.update(val_metrics)
        row.update(train_metrics)
        
        rows.append(row)
        all_keys.update(row.keys())

    # 排序并写入 CSV
    fixed_cols = ['dir-name', 'additional_features']
    metric_cols = sorted([k for k in all_keys if k not in fixed_cols])
    # 简单的排序优化：把 loss 放在前面
    metric_cols.sort(key=lambda x: (x.split('/')[0], 'loss' not in x, x)) # loss first within split
    
    columns = fixed_cols + metric_cols

    with open(args.output, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, '') for k in columns})

    print(f'CSV saved to {args.output}')

if __name__ == '__main__':
    main()