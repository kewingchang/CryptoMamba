import os, sys, pathlib
sys.path.insert(0, os.path.dirname(pathlib.Path(__file__).parent.absolute()))

import pandas as pd
import numpy as np
import lightgbm as lgb
import argparse
import datetime
import csv
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, precision_score, recall_score
from utils.io_tools import load_yaml, load_data


def validate_data_integrity(df, data_cfg, train_cfg):
    fixed_feats = data_cfg.get('fixed_features', [])
    add_feats = data_cfg.get('additional_features', []) or []
    target_col = train_cfg.get('target')

    if not fixed_feats: sys.exit(1)
    if not target_col: sys.exit(1)

    all_feats = fixed_feats + add_feats
    if target_col in all_feats:
        all_feats = [f for f in all_feats if f != target_col]
        
    return all_feats, target_col

def filter_noise(df, close_chg_col, threshold=0.001):
    if close_chg_col not in df.columns:
        print("[ATTENTION!] Skip filter_noise...")
        return df
    print(f"[Preprocessing] Filtering noise (|Next_Day_Chg| < {threshold*100}%) using {close_chg_col}...")
    next_day_chg = df[close_chg_col].shift(-1)
    mask = (next_day_chg.abs() >= threshold) | (next_day_chg.isna())
    df_filtered = df.loc[mask].reset_index(drop=True)
    return df_filtered

def split_data(df, data_config):
    def get_slice(d_df, start, end):
        mask = (d_df['Date'] >= start) & (d_df['Date'] < end)
        return d_df.loc[mask]

    train_df = get_slice(df, data_config['train_interval'][0], data_config['train_interval'][1])
    val_df   = get_slice(df, data_config['val_interval'][0], data_config['val_interval'][1])
    test_df  = get_slice(df, data_config['test_interval'][0], data_config['test_interval'][1])
    return train_df, val_df, test_df

# ==========================================
# 核心逻辑：提取最佳阈值指标
# ==========================================
def find_best_threshold_metrics(y_true, y_prob, direction='Long', min_trades=5):
    """
    寻找在满足最小交易次数限制下的最高 Precision
    """
    best_prec = 0.0
    best_thresh = 0.0
    best_recall = 0.0
    best_trades = 0
    
    # 扫描范围
    if direction == 'Long':
        # 从均值向上扫描到最大值
        start = max(0.3, float(y_prob.mean()))
        end = min(0.99, float(y_prob.max()))
        scan_range = np.arange(start, end + 0.05, 0.01)
        pos_label = 1
    else: # Short
        # 从均值向下扫描到最小值
        start = float(y_prob.mean())
        end = float(y_prob.min())
        if end >= start: end = start - 0.1
        scan_range = np.arange(start, end - 0.05, -0.01)
        pos_label = 0

    for thresh in scan_range:
        if direction == 'Long':
            y_tmp = (y_prob > thresh).astype(int)
        else: # Short: 概率小于阈值才算预测为0
            y_tmp = np.where(y_prob < thresh, 0, 1)
        
        # 计算指标
        # 注意：对于 Short，我们关注 label 0 的 Precision
        prec = precision_score(y_true, y_tmp, pos_label=pos_label, zero_division=0)
        rec = recall_score(y_true, y_tmp, pos_label=pos_label, zero_division=0)
        
        if direction == 'Long':
            count = y_tmp.sum() # 预测为1的数量
        else:
            count = (y_tmp == 0).sum() # 预测为0的数量
            
        if count >= min_trades:
            if prec > best_prec:
                best_prec = prec
                best_thresh = thresh
                best_recall = rec
                best_trades = count
    
    return best_prec, best_thresh, best_recall, best_trades

# ==========================================
# 主程序
# ==========================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_config", required=True)
    parser.add_argument("--params_config", required=True)
    parser.add_argument("--training_config", required=True)
    parser.add_argument("--close_chg_col", type=str, default="PF_Close_Chg")
    parser.add_argument("--output", type=str, default="experiment_results.csv", help="CSV file to save metrics")
    
    try: args = parser.parse_args()
    except: args, _ = parser.parse_known_args()

    data_cfg = load_yaml(args.data_config)
    params_cfg = load_yaml(args.params_config)
    train_cfg = load_yaml(args.training_config)

    df = load_data(path=data_cfg['data_path'], date_format=data_cfg.get('date_format'))
    df = filter_noise(df, args.close_chg_col, threshold=0.001)
    
    feature_cols, target_col = validate_data_integrity(df, data_cfg, train_cfg)
    train_df, val_df, test_df = split_data(df, data_cfg)

    X_train = train_df[feature_cols]
    y_train = train_df[target_col].astype(int)
    X_val = val_df[feature_cols]
    y_val = val_df[target_col].astype(int)
    X_test = test_df[feature_cols]
    y_test = test_df[target_col].astype(int)

    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_val = lgb.Dataset(X_val, y_val, reference=lgb_train)

    lgbm_params = params_cfg.copy()
    lgbm_params['objective'] = 'binary'
    lgbm_params['metric'] = 'auc'
    lgbm_params['verbose'] = -1
    lgbm_params.pop('num_class', None)
    lgbm_params.pop('class_weight', None)

    # 训练
    print(f"[Training] Features: {len(feature_cols)}")
    print("[Training] Starting...")

    callbacks = [lgb.log_evaluation(period=100)]
    if train_cfg.get('early_stop', False):
        callbacks.append(lgb.early_stopping(stopping_rounds=train_cfg.get('stopping_rounds', 50)))

    model = lgb.train(
        lgbm_params,
        lgb_train,
        num_boost_round=lgbm_params.get('n_estimators', 10000),
        valid_sets=[lgb_train, lgb_val],
        valid_names=['train', 'valid'],
        callbacks=callbacks
    )

    # ==========================================
    # 评估与数据收集
    # ==========================================
    print("\n" + "-" * 30)
    print("[Evaluation] Collecting Metrics...")
    
    # 2. 获取训练过程中的 AUC
    # model.best_score 结构通常为 {'train': {'auc': 0.8}, 'valid': {'auc': 0.7}}
    train_auc = model.best_score.get('train', {}).get('auc', 0)
    valid_auc = model.best_score.get('valid', {}).get('auc', 0)
    
    # 预测测试集
    y_prob = model.predict(X_test, num_iteration=model.best_iteration)
    y_pred_default = (y_prob > 0.5).astype(int)
    
    test_acc = accuracy_score(y_test, y_pred_default)
    test_auc = roc_auc_score(y_test, y_prob)
    
    # 3. 计算 Prob Stats (Long & Short)
    # Long Prob = y_prob
    # Short Prob = 1 - y_prob
    prob_stats_long_min = y_prob.min()
    prob_stats_long_max = y_prob.max()
    prob_stats_long_mean = y_prob.mean()
    
    short_prob = 1 - y_prob
    prob_stats_short_min = short_prob.min()
    prob_stats_short_max = short_prob.max()
    prob_stats_short_mean = short_prob.mean()
    
    print(f"Test Accuracy : {test_acc:.4f}")
    print(f"Test AUC      : {test_auc:.4f}")
    print(f"Prob Stats (Long): Min={prob_stats_long_min:.4f}, Max={prob_stats_long_max:.4f}, Mean={prob_stats_long_mean:.4f}")

    # 4. 提取 Threshold Scan 关键指标
    # 我们寻找在至少有 5 笔交易的情况下，能达到的最高 Precision
    long_max_prec, long_thresh_at_max, long_rec_at_max, long_trades_at_max = find_best_threshold_metrics(y_test, y_prob, 'Long', min_trades=5)
    short_max_prec, short_thresh_at_max, short_rec_at_max, short_trades_at_max = find_best_threshold_metrics(y_test, y_prob, 'Short', min_trades=5)

    # 打印部分 Threshold Scan 以便肉眼检查 (Long)
    print("\n>>> Threshold Scan Preview (Long):")
    print(f"{'Threshold':<10} | {'Precision':<10} | {'Recall':<10} | {'Trades':<10}")
    start_scan = max(0.3, float(prob_stats_long_mean))
    end_scan = min(0.95, float(prob_stats_long_max))
    for thresh in np.arange(start_scan, end_scan + 0.05, 0.01):
        y_tmp = (y_prob > thresh).astype(int)
        prec = precision_score(y_test, y_tmp, pos_label=1, zero_division=0)
        rec = recall_score(y_test, y_tmp, pos_label=1, zero_division=0)
        count = y_tmp.sum()
        if count > 0:
            print(f"{thresh:.4f}     | {prec:.4f}     | {rec:.4f}     | {count:<10}")
    print("-" * 50)

    # ==========================================
    # 下跌 (Down/0) 的高置信度扫描
    # ==========================================
    print("\n>>> Threshold Scan (For Down/0 Precision - Lower Prob is Better):")
    print(f"{'Threshold':<10} | {'Precision':<10} | {'Recall':<10} | {'Trades':<10}")
    print("-" * 50)

    # 扫描区间：从均值(Mean) 向下扫描到 最小值(Min)
    # 比如从 0.4 扫描到 0.1
    start_scan_down = float(y_prob.mean())
    end_scan_down = float(y_prob.min())
    
    # 防止区间倒挂
    if end_scan_down >= start_scan_down: 
        end_scan_down = start_scan_down - 0.1

    # 生成递减的 range, 步长 -0.01
    scan_range_down = np.arange(start_scan_down, end_scan_down - 0.05, -0.01)

    for thresh in scan_range_down:
        # 逻辑：概率小于阈值，才判定为做空(0)，否则判定为非空(1)
        # 这里的 1 只是为了占位，表示"Rest"，不影响 pos_label=0 的计算
        y_tmp = np.where(y_prob < thresh, 0, 1)
        
        # 计算 Label 0 的指标
        prec = precision_score(y_test, y_tmp, pos_label=0, zero_division=0)
        rec = recall_score(y_test, y_tmp, pos_label=0, zero_division=0)
        
        # 统计预测为 0 的数量
        count = (y_tmp == 0).sum()
        
        if count > 0:
            print(f"<{thresh:.4f}    | {prec:.4f}     | {rec:.4f}     | {count:<10}")
    print("-" * 50)

    # ==========================================
    # 保存 CSV
    # ==========================================
    result_row = {
        'Timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'Model_Config': args.data_config, # 记录使用的参数文件
        'Features': len(feature_cols),
        'Train_AUC': round(train_auc, 5),
        'Valid_AUC': round(valid_auc, 5),
        'Test_Accuracy': round(test_acc, 5),
        'Test_AUC': round(test_auc, 5),
        
        # Prob Stats
        'Long_Prob_Min': round(prob_stats_long_min, 4),
        'Long_Prob_Max': round(prob_stats_long_max, 4),
        'Long_Prob_Mean': round(prob_stats_long_mean, 4),
        'Short_Prob_Min': round(prob_stats_short_min, 4),
        'Short_Prob_Max': round(prob_stats_short_max, 4),
        'Short_Prob_Mean': round(prob_stats_short_mean, 4),
        
        # Threshold Scan Summary (Long)
        'Long_Max_Precision': round(long_max_prec, 4),
        'Long_Best_Threshold': round(long_thresh_at_max, 4),
        'Long_Recall_at_Best': round(long_rec_at_max,4),
        'Long_Trades_at_Best': int(long_trades_at_max),
        
        # Threshold Scan Summary (Short)
        'Short_Max_Precision': round(short_max_prec, 4),
        'Short_Best_Threshold': round(short_thresh_at_max, 4),
        'Short_Recall_at_Best': round(short_rec_at_max,4),
        'Short_Trades_at_Best': int(short_trades_at_max),
    }
    
    # 写入文件
    file_exists = os.path.isfile(args.output)
    fieldnames = list(result_row.keys())
    
    try:
        with open(args.output, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            
            if not file_exists:
                writer.writeheader()
            
            writer.writerow(result_row)
            
        print(f"[Success] Evaluation metrics appended to {args.output}")
    except Exception as e:
        print(f"[Error] Failed to write to CSV: {e}")

    # 保存模型 (保持不变)
    save_dir = os.path.dirname(data_cfg.get('root', '.'))
    if 'checkpoints' in data_cfg: save_dir = data_cfg['checkpoints']
    elif 'root' in data_cfg: save_dir = data_cfg['root']
    if not os.path.exists(save_dir): os.makedirs(save_dir, exist_ok=True)
    
    save_path = os.path.join(save_dir, 'lgbm_model_updown.txt')
    model.save_model(save_path)
    print(f"[Output] Model saved to: {save_path}")

if __name__ == "__main__":
    main()