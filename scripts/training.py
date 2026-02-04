import pandas as pd
import numpy as np
import lightgbm as lgb
import yaml
import argparse
import sys
import os
from sklearn.metrics import classification_report, accuracy_score
from sklearn.utils.class_weight import compute_sample_weight # 新增：用于处理 'balanced' 字符串

# ==========================================
# 辅助函数
# ==========================================

def load_yaml(filepath):
    """读取YAML配置文件"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = yaml.safe_load(f)
            if content is None:
                return {}
            return content
    except Exception as e:
        print(f"[Fatal] Error loading config {filepath}: {e}")
        sys.exit(1)

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

    unique_labels = df[target_col].unique()
    valid_labels = {0, 1, 2}
    invalid_labels = [x for x in unique_labels if x not in valid_labels]
    
    if df[target_col].isnull().any():
        print(f"[Fatal] Target column '{target_col}' contains NaN values!")
        sys.exit(1)

    if invalid_labels:
        print(f"[Fatal] Invalid labels found: {invalid_labels}. Must be 0, 1, 2.")
        sys.exit(1)

    print(f"[Validation] Passed. Features: {len(all_feats)}, Target: '{target_col}'")
    return all_feats, target_col

def load_data(data_config):
    path = data_config['data_path']
    print(f"[Data] Loading CSV from {path}...")
    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"[Fatal] Failed to read CSV: {e}")
        sys.exit(1)
        
    if 'Date' in df.columns:
        try:
            df['Date'] = pd.to_datetime(df['Date'], format=data_config.get('date_format'))
            df = df.sort_values('Date').reset_index(drop=True)
        except Exception as e:
            print(f"[Fatal] Date parsing failed: {e}")
            sys.exit(1)
    else:
        print("[Fatal] CSV must contain a 'Date' column.")
        sys.exit(1)
    return df

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
    
    try: args = parser.parse_args()
    except: args, _ = parser.parse_known_args()

    print("-" * 30)
    data_cfg = load_yaml(args.data_config)
    params_cfg = load_yaml(args.params_config)
    train_cfg = load_yaml(args.training_config)

    df = load_data(data_cfg)
    feature_cols, target_col = validate_data_integrity(df, data_cfg, train_cfg)
    train_df, val_df, test_df = split_data(df, data_cfg)

    # --- 准备数据 ---
    X_train = train_df[feature_cols]
    y_train = train_df[target_col].astype(int)
    
    X_val = val_df[feature_cols]
    y_val = val_df[target_col].astype(int)
    
    X_test = test_df[feature_cols]
    y_test = test_df[target_col].astype(int)

    # =========================================================
    # 【核心修改】处理 class_weight
    # =========================================================
    lgbm_params = params_cfg.copy()
    
    # 1. 从参数字典中 "偷走" class_weight，因为原生 lgb.train 不认识它
    custom_weight_config = lgbm_params.pop('class_weight', None)
    
    sample_weights = None
    
    if custom_weight_config:
        print(f"[Config] Applying custom class weights: {custom_weight_config}")
        
        if custom_weight_config == 'balanced':
            # 自动计算平衡权重
            sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)
        elif isinstance(custom_weight_config, dict):
            # 手动指定权重字典 {0:1, 1:3, 2:3}
            # map 将 Series 中的 label 替换为对应的 weight
            sample_weights = y_train.map(custom_weight_config).values
            
            # 处理 map 之后可能产生的 NaN (如果字典没覆盖所有label)
            if np.isnan(sample_weights).any():
                print("[Fatal] class_weight dict does not cover all labels present in data!")
                sys.exit(1)
        else:
             print(f"[Warning] Unknown format for class_weight: {custom_weight_config}. Ignoring.")

    # =========================================================

    # 构建 Dataset (将计算好的 weights 塞进去)
    lgb_train = lgb.Dataset(X_train, y_train, weight=sample_weights)
    lgb_val = lgb.Dataset(X_val, y_val, reference=lgb_train)

    # 强制核心参数
    lgbm_params['objective'] = 'multiclass'
    lgbm_params['num_class'] = 3
    lgbm_params['metric'] = 'multi_logloss' # 或 multi_error
    lgbm_params['verbose'] = -1

    # 训练
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

    # 评估
    print("\n" + "-" * 30)
    print("[Evaluation] Testing model...")
    
    y_prob = model.predict(X_test, num_iteration=model.best_iteration)
    y_pred = [np.argmax(x) for x in y_prob]
    
    acc = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {acc:.4f}")
    
    target_names = ['Wait (0)', 'Long (1)', 'Short (2)']
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=target_names, zero_division=0))

    # 保存
    save_dir = os.path.dirname(data_cfg.get('root', '.')) # 简单容错
    if 'checkpoints' in data_cfg: save_dir = data_cfg['checkpoints'] # 优先用 config 里的
    elif 'root' in data_cfg: save_dir = data_cfg['root']
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
        
    save_path = os.path.join(save_dir, 'lgbm_model.txt')
    model.save_model(save_path)
    print(f"[Output] Model saved to: {save_path}")
    print("-" * 30)

if __name__ == "__main__":
    main()