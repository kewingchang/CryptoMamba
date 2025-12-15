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
            parts = re.split(r'│\s+', line.strip('┃┡╇┩'))
            if len(parts) >= 2:
                key = parts[1].strip()
                value = parts[2].strip() if len(parts) > 2 else ''
                # Map to the correct prefix
                for original_prefix, target_prefix in prefix_mapping.items():
                    if key.startswith(original_prefix):
                        new_key = key.replace(original_prefix, target_prefix, 1)
                        try:
                            metrics[new_key] = float(value)
                        except ValueError:
                            pass  # Skip if not float
    return metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True, help='Main directory containing subdirectories with log.txt')
    args = parser.parse_args()

    data_dir = args.data_dir
    output_csv = 'output.csv'  # Assuming output filename

    # Define CSV columns
    columns = [
        'dir-name', 'additional_features',
        'train/mae', 'train/mape', 'train/mse', 'train/rmse', 'train/rmse_price', 'train/acc', 'train/smooth_l1',
        'val/mae', 'val/mape', 'val/mse', 'val/rmse', 'val/rmse_price', 'val/acc', 'val/smooth_l1',
        'test/mae', 'test/mape', 'test/mse', 'test/rmse', 'test/acc', 'test/smooth_l1',
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

        # Extract additional_features
        additional_features = extract_additional_features(lines)

        # Extract test metrics
        test_metrics = extract_metrics(lines, '>>>>>>>>>>> Test Set Validate <<<<<<<<<<<<<<', {'test/': 'test/'})

        # Extract val metrics
        val_metrics = extract_metrics(lines, '>>>>>>>>>>> Val Set Validate <<<<<<<<<<<<<<', {'val/': 'val/'})

        # Extract train metrics (mapped from val/ to train/)
        train_metrics = extract_metrics(lines, '>>>>>>>>>>> Train Set Validate <<<<<<<<<<<<<<', {'val/': 'train/'})

        # Combine all
        row = {
            'dir-name': subdir,
            'additional_features': additional_features,
            'note': ''  # Empty note
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
            filtered_row = {key: row.get(key, '') for key in columns}
            writer.writerow(filtered_row)

    print(f'CSV saved to {output_csv}')

if __name__ == '__main__':
    main()
