# training_oneway.py
import os, sys, pathlib
sys.path.insert(0, os.path.dirname(pathlib.Path(__file__).parent.absolute()))

import pandas as pd
import numpy as np
import lightgbm as lgb
import yaml
import argparse
import sys
import os
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, precision_score, recall_score
from sklearn.metrics import f1_score


# ==========================================
# 辅助函数
# ==========================================
def load_yaml(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = yaml.safe_load(f)
            return content if content else {}
    except Exception as e:
        print(f"[Fatal] Error loading config {filepath}: {e}")
        sys.exit(1)

def load_data(path, date_format):
    print(f"[Data] Loading CSV from {path}...")
    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"[Fatal] Failed to read CSV: {e}")
        sys.exit(1)
        
    if 'Date' in df.columns:
        try:
            df['Date'] = pd.to_datetime(df['Date'], format=date_format)
            df = df.sort_values('Date').reset_index(drop=True)
        except Exception as e:
            print(f"[Fatal] Date parsing failed: {e}")
            sys.exit(1)
    else:
        print("[Fatal] CSV must contain a 'Date' column.")
        sys.exit(1)
    return df

def validate_data_integrity(df, data_cfg, train_cfg):
    """严格校验"""
    print("[Validation] Starting pre-flight checks...")

    fixed_feats = data_cfg.get('fixed_features', [])
    if not fixed_feats:
        print("[Fatal] 'fixed_features' list cannot be empty!")
        sys.exit(1)
    
    add_feats = data_cfg.get('additional_features', []) or []
    all_feats = fixed_feats + add_feats
    missing_cols = [c for c in all_feats if c not in df.columns]
    
    if missing_cols:
        print(f"[Fatal] Missing features in CSV: {missing_cols}")
        sys.exit(1)

    target_col = train_cfg.get('target')
    if not target_col:
        print("[Fatal] 'target' must be defined in training_config!")
        sys.exit(1)
    
    if target_col not in df.columns:
        print(f"[Fatal] Target column '{target_col}' not found in CSV!")
        sys.exit(1)

    # 检查原始数据的 Label 是否合法 (0, 1, 2)
    unique_labels = df[target_col].unique()
    valid_labels = {0, 1, 2}
    invalid_labels = [x for x in unique_labels if x not in valid_labels]
    
    if df[target_col].isnull().any():
        print(f"[Fatal] Target column '{target_col}' contains NaN values!")
        sys.exit(1)

    if invalid_labels:
        print(f"[Fatal] Invalid labels found in source data: {invalid_labels}. Must be 0, 1, 2.")
        sys.exit(1)

    print(f"[Validation] Passed. Features: {len(all_feats)}, Target: '{target_col}'")
    return all_feats, target_col

def split_data(df, data_config):
    def get_slice(d_df, start, end):
        mask = (d_df['Date'] >= start) & (d_df['Date'] < end)
        return d_df.loc[mask]

    train_df = get_slice(df, data_config['train_interval'][0], data_config['train_interval'][1])
    val_df   = get_slice(df, data_config['val_interval'][0], data_config['val_interval'][1])
    test_df  = get_slice(df, data_config['test_interval'][0], data_config['test_interval'][1])
    
    print(f"[Split] Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
    
    if len(train_df) == 0:
        print("[Fatal] Training set is empty!")
        sys.exit(1)
    return train_df, val_df, test_df

# ==========================================
# 主程序
# ==========================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_config", required=True)
    parser.add_argument("--params_config", required=True)
    parser.add_argument("--training_config", required=True)
    # 新增方向参数
    parser.add_argument("--direction", required=True, choices=['Long', 'Short'], 
                        help="Train a binary model for 'Long' or 'Short' direction")
    
    try: args = parser.parse_args()
    except: args, _ = parser.parse_known_args()

    print("-" * 30)
    print(f"Training Mode: Binary Classification for [{args.direction}]")
    print("-" * 30)
    
    data_cfg = load_yaml(args.data_config)
    params_cfg = load_yaml(args.params_config)
    train_cfg = load_yaml(args.training_config)

    # 1. 加载数据
    df = load_data(
        path=data_cfg['data_path'], 
        date_format=data_cfg.get('date_format')
    )
    feature_cols, target_col = validate_data_integrity(df, data_cfg, train_cfg)
    train_df, val_df, test_df = split_data(df, data_cfg)

    # 2. Label 映射逻辑 (One-vs-Rest)
    # Long Mode:  1 -> 1 (Positive), 0/2 -> 0 (Negative)
    # Short Mode: 2 -> 1 (Positive), 0/1 -> 0 (Negative)
    
    target_val = 1 if args.direction == 'Long' else 2
    print(f"[Preprocessing] Mapping Label {target_val} to 1 (Positive), others to 0 (Negative).")

    def transform_label(y_series, target_val):
        return (y_series == target_val).astype(int)

    X_train = train_df[feature_cols]
    y_train = transform_label(train_df[target_col], target_val)
    
    X_val = val_df[feature_cols]
    y_val = transform_label(val_df[target_col], target_val)
    
    X_test = test_df[feature_cols]
    y_test = transform_label(test_df[target_col], target_val)

    # 3. 构建 Dataset
    # 注意：二分类不需要传递 custom sample_weights 列表给 dataset
    # 平衡问题通过 params.yaml 中的 scale_pos_weight 参数自动处理
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_val = lgb.Dataset(X_val, y_val, reference=lgb_train)

    # 4. 参数配置
    lgbm_params = params_cfg.copy()
    
    # 强制覆盖为二分类设置
    lgbm_params['objective'] = 'binary'
    lgbm_params['metric'] = lgbm_params.get('metric', 'auc') # 默认为 auc，除非 yaml 里指定了 binary_logloss
    lgbm_params['verbose'] = -1
    
    # 清理可能残留的多分类参数
    lgbm_params.pop('num_class', None)
    lgbm_params.pop('class_weight', None) # 二分类使用 scale_pos_weight，清理掉多分类的字典配置

    # 5. 训练
    print(f"[Training] Features: {len(feature_cols)}")
    print(f"[Training] Params: {lgbm_params}")
    print("[Training] Starting...")
    
    callbacks = []
    if train_cfg.get('early_stop', False):
        rounds = train_cfg.get('stopping_rounds', 50)
        callbacks.append(lgb.early_stopping(stopping_rounds=rounds))
        print(f"[Config] Early stopping: {rounds} rounds")
    
    callbacks.append(lgb.log_evaluation(period=100))

    model = lgb.train(
        lgbm_params,
        lgb_train,
        num_boost_round=lgbm_params.get('n_estimators', 1000),
        valid_sets=[lgb_train, lgb_val],
        valid_names=['train', 'valid'],
        callbacks=callbacks
    )

    # ------------------------------------------------------------------
    # 修改后的评估模块：打印阈值阶梯表 (针对交易员优化)
    # ------------------------------------------------------------------
    print("\n" + "-" * 30)
    print("[Evaluation] Testing model...")
    
    # 获取预测概率
    y_prob = model.predict(X_test, num_iteration=model.best_iteration)
    
    print(f"Prediction Probabilities stats:")
    print(f"  Min: {y_prob.min():.4f} | Max: {y_prob.max():.4f} | Mean: {y_prob.mean():.4f}")
    
    print("\n>>> Threshold Scan (Focus on Precision):")
    print(f"{'Threshold':<10} | {'Precision':<10} | {'Recall':<10} | {'Trades (Count)':<15}")
    print("-" * 55)
    
    # 从 Mean 开始扫描到 Max
    start_scan = float(y_prob.mean())
    end_scan = float(y_prob.max())
    # 如果 Max 太小，强行扫描到 0.6
    if end_scan < 0.5: end_scan = 0.6
    
    scan_range = np.arange(start_scan, end_scan + 0.05, 0.01)
    
    for thresh in scan_range:
        y_tmp = (y_prob > thresh).astype(int)
        prec = precision_score(y_test, y_tmp, zero_division=0)
        rec = recall_score(y_test, y_tmp, zero_division=0)
        count = y_tmp.sum()
        
        # 只显示有交易的阈值
        if count > 0:
            print(f"{thresh:.4f}     | {prec:.4f}     | {rec:.4f}     | {count:<15}")
            
    print("-" * 55)

    # 保存模型 (保持不变)
    save_dir = os.path.dirname(data_cfg.get('root', '.'))
    if 'checkpoints' in data_cfg: save_dir = data_cfg['checkpoints']
    elif 'root' in data_cfg: save_dir = data_cfg['root']
    if not os.path.exists(save_dir): os.makedirs(save_dir, exist_ok=True)
    
    filename = f'lgbm_model_{args.direction}.txt'
    save_path = os.path.join(save_dir, filename)
    model.save_model(save_path)
    print(f"[Output] Model saved to: {save_path}")
    print("-" * 30)

if __name__ == "__main__":
    main()