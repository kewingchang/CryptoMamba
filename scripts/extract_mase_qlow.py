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
            if found_first:
                # Second occurrence, features is the previous line
                break
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
        if not started:
            continue
        if '┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┓' in line:
            in_table = True
            continue
        if in_table and '└───────────────────────────┴───────────────────────────┘' in line:
            break
        if in_table and '│' in line:
            # Modified to be slightly more robust for the new log format
            # New format typically looks like: │ metric_name │ value │
            parts = re.split(r'│\s*', line.strip('┃┡╇┩'))
            
            # parts[0] might be empty if line starts with │
            # parts[1] is usually the key
            # parts[2] is usually the value
            if len(parts) >= 3:
                key = parts[1].strip()
                value_str = parts[2].strip()
                if not key: continue # Skip empty keys
                
                # Map to the correct prefix
                mapped_key = key
                for original_prefix, target_prefix in prefix_mapping.items():
                    if key.startswith(original_prefix):
                        mapped_key = key.replace(original_prefix, target_prefix, 1)
                        break
                
                try:
                    metrics[mapped_key] = float(value_str)
                except ValueError:
                    pass  # Skip if not float
    return metrics

def calculate_diffs(metrics_dict, prefix):
    """
    Calculates the diff metrics based on the extracted cover metrics.
    Formula: diff = abs(cover - quantile)
    Target Quantiles: 0.05, 0.2, 0.4, 0.6
    """
    quantiles = {
        'q0.05': 0.05,
        'q0.2': 0.2,
        'q0.4': 0.4,
        'q0.6': 0.6
    }

    for q_name, target_val in quantiles.items():
        cover_key = f"{prefix}/{q_name}_cover"
        diff_key = f"{prefix}/{q_name}_diff"
        
        # Only calculate if the cover metric was successfully extracted
        if cover_key in metrics_dict:
            try:
                cover_val = metrics_dict[cover_key]
                metrics_dict[diff_key] = abs(cover_val - target_val)
            except (TypeError, ValueError):
                metrics_dict[diff_key] = ''
    
    return metrics_dict

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True, help='Main directory containing subdirectories with log.txt')
    args = parser.parse_args()

    data_dir = args.data_dir
    output_csv = 'output.csv'

    # Updated columns based on the new requirements
    columns = [
        'dir-name', 'additional_features',
        
        # Test Set
        'test/loss',
        'test/q0.05_cover', 'test/q0.05_mae', 'test/q0.05_diff',
        'test/q0.2_cover',  'test/q0.2_mae',  'test/q0.2_diff',
        'test/q0.4_cover',  'test/q0.4_mae',  'test/q0.4_diff',
        'test/q0.6_cover',  'test/q0.6_mae',  'test/q0.6_diff',
        
        # Val Set
        'val/loss',
        'val/q0.05_cover', 'val/q0.05_mae', 'val/q0.05_diff',
        'val/q0.2_cover',  'val/q0.2_mae',  'val/q0.2_diff',
        'val/q0.4_cover',  'val/q0.4_mae',  'val/q0.4_diff',
        'val/q0.6_cover',  'val/q0.6_mae',  'val/q0.6_diff',
        
        # Train Set
        'train/loss',
        'train/q0.05_cover', 'train/q0.05_mae', 'train/q0.05_diff',
        'train/q0.2_cover',  'train/q0.2_mae',  'train/q0.2_diff',
        'train/q0.4_cover',  'train/q0.4_mae',  'train/q0.4_diff',
        'train/q0.6_cover',  'train/q0.6_mae',  'train/q0.6_diff',
        
        'note'
    ]

    rows = []

    # Traverse subdirectories
    for subdir in sorted(os.listdir(data_dir)):
        subdir_path = os.path.join(data_dir, subdir)
        if not os.path.isdir(subdir_path):
            continue
        log_path = os.path.join(subdir_path, 'log.txt')
        if not os.path.exists(log_path):
            continue

        with open(log_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        # Extract additional_features (Note: existing logic kept, though pattern might have changed)
        additional_features = extract_additional_features(lines)

        # 1. Extract and Calculate Test Metrics
        test_metrics = extract_metrics(lines, '>>>>>>>>>>> Test Set Validate <<<<<<<<<<<<<<', {'test/': 'test/'})
        test_metrics = calculate_diffs(test_metrics, 'test')

        # 2. Extract and Calculate Val Metrics
        val_metrics = extract_metrics(lines, '>>>>>>>>>>> Val Set Validate <<<<<<<<<<<<<<', {'val/': 'val/'})
        val_metrics = calculate_diffs(val_metrics, 'val')

        # 3. Extract and Calculate Train Metrics 
        # Note: Input has 'val/' prefix in log, mapped to 'train/' in output dict
        train_metrics = extract_metrics(lines, '>>>>>>>>>>> Train Set Validate <<<<<<<<<<<<<<', {'val/': 'train/'})
        train_metrics = calculate_diffs(train_metrics, 'train')

        # Combine all
        row = {
            'dir-name': subdir,
            'additional_features': additional_features,
            'note': ''
        }
        row.update(train_metrics)
        row.update(val_metrics)
        row.update(test_metrics)

        rows.append(row)

    # Write to CSV
    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            # Only write keys that are in the columns list to avoid errors or clutter
            filtered_row = {key: row.get(key, '') for key in columns}
            writer.writerow(filtered_row)

    print(f'CSV saved to {output_csv}')

if __name__ == '__main__':
    main()