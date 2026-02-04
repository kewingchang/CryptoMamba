# training.py
import os, sys, pathlib
sys.path.insert(0, os.path.dirname(pathlib.Path(__file__).parent.absolute()))

import pandas as pd
import numpy as np
import lightgbm as lgb
import yaml
import argparse
import sys
import os
from utils.io_tools import load_yaml, load_data
from sklearn.metrics import classification_report, accuracy_score


# ==========================================
# 辅助函数
# ==========================================

def validate_data_integrity(df, data_cfg, train_cfg):
    """
    严格校验：特征存在性 + Label 合法性
    """
    print("[Validation] Starting pre-flight checks...")

    # 1. 检查特征列是否存在
    fixed_feats = data_cfg.get('fixed_features', [])
    if not fixed_feats:
        print("[Fatal] 'fixed_features' list cannot be empty!")
        sys.exit(1)
    
    add_feats = data_cfg.get('additional_features', []) or [] # 允许为空
    
    all_feats = fixed_feats + add_feats
    missing_cols = [c for c in all_feats if c not in df.columns]
    
    if missing_cols:
        print(f"[Fatal] Missing features in CSV: {missing_cols}")
        sys.exit(1)

    # 2. 检查 Target 列是否存在
    target_col = train_cfg.get('target')
    if not target_col:
        print("[Fatal] 'target' must be defined in training_config!")
        sys.exit(1)
    
    if target_col not in df.columns:
        print(f"[Fatal] Target column '{target_col}' not found in CSV!")
        sys.exit(1)

    # 3. 检查 Label 值域是否合法 (必须是 0, 1, 2)
    # LightGBM multiclass num_class=3 要求 label 必须是 [0, num_class-1]
    unique_labels = df[target_col].unique()
    valid_labels = {0, 1, 2}
    
    # 检查是否有非法值
    invalid_labels = [x for x in unique_labels if x not in valid_labels]
    
    # 检查是否包含 NaN
    if df[target_col].isnull().any():
        print(f"[Fatal] Target column '{target_col}' contains NaN values! Please clean your data.")
        sys.exit(1)

    if invalid_labels:
        print(f"[Fatal] Invalid labels found in '{target_col}': {invalid_labels}")
        print("LightGBM requires labels to be integers: 0 (Wait), 1 (Long), 2 (Short).")
        print("Please fix your data generation pipeline.")
        sys.exit(1)

    print(f"[Validation] Passed. Features: {len(all_feats)}, Target: '{target_col}' (Values: {sorted(unique_labels)})")
    
    return all_feats, target_col

def split_data(df, data_config):
    """按时间切分"""
    def get_slice(d_df, start, end):
        mask = (d_df['Date'] >= start) & (d_df['Date'] < end)
        return d_df.loc[mask]

    train_df = get_slice(df, data_config['train_interval'][0], data_config['train_interval'][1])
    val_df   = get_slice(df, data_config['val_interval'][0], data_config['val_interval'][1])
    test_df  = get_slice(df, data_config['test_interval'][0], data_config['test_interval'][1])
    
    print(f"[Split] Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
    
    if len(train_df) == 0:
        print("[Fatal] Training set is empty! Check date intervals.")
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
    
    try: args = parser.parse_args()
    except: args, _ = parser.parse_known_args()

    # 1. 加载配置
    print("-" * 30)
    data_cfg = load_yaml(args.data_config)
    params_cfg = load_yaml(args.params_config)
    train_cfg = load_yaml(args.training_config)

    # 2. 加载数据
    # 直接传入 path 字符串
    df = load_data(
        path=data_cfg['data_path'], 
        date_format=data_cfg.get('date_format')
    )

    # 3. 严格校验 (包含 Label 值域检查)
    feature_cols, target_col = validate_data_integrity(df, data_cfg, train_cfg)

    # 4. 数据切分
    train_df, val_df, test_df = split_data(df, data_cfg)

    # 5. 准备 Dataset
    # 直接使用原始 Label (假设已清洗为 0,1,2)
    X_train = train_df[feature_cols]
    y_train = train_df[target_col].astype(int)
    
    X_val = val_df[feature_cols]
    y_val = val_df[target_col].astype(int)
    
    X_test = test_df[feature_cols]
    y_test = test_df[target_col].astype(int)

    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_val = lgb.Dataset(X_val, y_val, reference=lgb_train)

    # 6. 准备参数
    lgbm_params = params_cfg.copy()
    lgbm_params['objective'] = 'multiclass'
    lgbm_params['num_class'] = 3  # 0, 1, 2
    lgbm_params['metric'] = 'multi_logloss'
    lgbm_params['verbose'] = -1

    # 7. 训练
    print(f"[Training] Features: {len(feature_cols)}")
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

    # 8. 评估
    print("\n" + "-" * 30)
    print("[Evaluation] Testing model...")
    
    y_prob = model.predict(X_test, num_iteration=model.best_iteration)
    y_pred = [np.argmax(x) for x in y_prob]
    
    acc = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {acc:.4f}")
    
    # 打印报告
    # 0=Wait, 1=Long, 2=Short
    target_names = ['Wait (0)', 'Long (1)', 'Short (2)']
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=target_names, zero_division=0))

    # 9. 保存
    save_dir = data_cfg.get('checkpoints', '.')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    save_path = os.path.join(save_dir, 'lgbm_model.txt')
    model.save_model(save_path)
    print(f"[Output] Model saved to: {save_path}")
    print("-" * 30)

if __name__ == "__main__":
    main()