# training_binary.py
import os, sys, pathlib
sys.path.insert(0, os.path.dirname(pathlib.Path(__file__).parent.absolute()))

import pandas as pd
import numpy as np
import lightgbm as lgb
import yaml
import argparse
import sys
import os
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
from utils.io_tools import load_yaml, load_data

# ==========================================
# 辅助函数
# ==========================================
def validate_data_integrity(df, data_cfg, train_cfg):
    print("[Validation] Starting pre-flight checks...")

    fixed_feats = data_cfg.get('fixed_features', [])
    add_feats = data_cfg.get('additional_features', []) or []
    target_col = train_cfg.get('target')
    
    if not target_col:
        print("[Fatal] 'target' not defined in training_config.")
        sys.exit(1)

    # 【防止泄露】从特征列表中剔除 target 列
    all_feats = fixed_feats + add_feats
    if target_col in all_feats:
        print(f"[Warning] Target '{target_col}' found in features! Removing it automatically to prevent leakage.")
        all_feats = [f for f in all_feats if f != target_col]

    missing_cols = [c for c in all_feats if c not in df.columns]
    if missing_cols:
        print(f"[Fatal] Missing features in CSV: {missing_cols}")
        sys.exit(1)
        
    if target_col not in df.columns:
        print(f"[Fatal] Target '{target_col}' not found in CSV!")
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

    print("-" * 30)
    data_cfg = load_yaml(args.data_config)
    params_cfg = load_yaml(args.params_config)
    train_cfg = load_yaml(args.training_config)

    # 加载数据
    df = load_data(path=data_cfg['data_path'], date_format=data_cfg.get('date_format'))
    
    # 校验并获取特征列表 (会自动剔除 target)
    feature_cols, target_col = validate_data_integrity(df, data_cfg, train_cfg)
    
    # 切分数据
    train_df, val_df, test_df = split_data(df, data_cfg)

    X_train = train_df[feature_cols]
    y_train = train_df[target_col].astype(int)
    X_val = val_df[feature_cols]
    y_val = val_df[target_col].astype(int)
    X_test = test_df[feature_cols]
    y_test = test_df[target_col].astype(int)

    # 构建 Dataset
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_val = lgb.Dataset(X_val, y_val, reference=lgb_train)

    # 准备参数
    lgbm_params = params_cfg.copy()
    lgbm_params['objective'] = 'binary'   # 强制二分类
    lgbm_params['metric'] = 'auc'         # 强制监控 AUC
    lgbm_params['verbose'] = -1
    
    # 清理多分类参数
    lgbm_params.pop('num_class', None)
    lgbm_params.pop('class_weight', None)

    # 训练
    print(f"[Training] Starting (Target: {target_col})...")
    
    callbacks = []
    if train_cfg.get('early_stop', False):
        rounds = train_cfg.get('stopping_rounds', 50)
        callbacks.append(lgb.early_stopping(stopping_rounds=rounds))
    
    callbacks.append(lgb.log_evaluation(period=100))

    model = lgb.train(
        lgbm_params,
        lgb_train,
        num_boost_round=lgbm_params.get('n_estimators', 10000),
        valid_sets=[lgb_train, lgb_val],
        valid_names=['train', 'valid'],
        callbacks=callbacks
    )

    # ==========================================
    # 【修正】二分类评估逻辑
    # ==========================================
    print("\n" + "-" * 30)
    print("[Evaluation] Testing model...")
    
    # 1. 获取概率 (1D array)
    y_prob = model.predict(X_test, num_iteration=model.best_iteration)
    
    # 2. 转换为类别 (默认阈值 0.5)
    y_pred = (y_prob > 0.5).astype(int)
    
    # 3. 计算指标
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    
    print(f"Test Accuracy : {acc:.4f}")
    print(f"Test AUC      : {auc:.4f}")
    
    print("\nClassification Report (Threshold 0.5):")
    # 这里的 names 对应 0 和 1
    print(classification_report(y_test, y_pred, target_names=['Down (0)', 'Up (1)'], zero_division=0))

    # 保存
    save_dir = os.path.dirname(data_cfg.get('root', '.'))
    if not os.path.exists(save_dir): os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'lgbm_model.txt')
    model.save_model(save_path)
    print(f"[Output] Model saved to: {save_path}")
    print("-" * 30)

if __name__ == "__main__":
    main()