import os, sys, pathlib
sys.path.insert(0, os.path.dirname(pathlib.Path(__file__).parent.absolute()))

import pandas as pd
import numpy as np
import lightgbm as lgb
import argparse
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_config", required=True)
    parser.add_argument("--params_config", required=True)
    parser.add_argument("--training_config", required=True)
    parser.add_argument("--close_chg_col", type=str, default="PF_Close_Chg")
    
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

    print("\n" + "-" * 30)
    print("[Evaluation] Testing model...")
    
    y_prob = model.predict(X_test, num_iteration=model.best_iteration)
    y_pred_default = (y_prob > 0.5).astype(int)
    
    acc = accuracy_score(y_test, y_pred_default)
    auc = roc_auc_score(y_test, y_prob)
    
    print(f"Test Accuracy : {acc:.4f}")
    print(f"Test AUC      : {auc:.4f}")
    print(f"Prob Stats: Min={y_prob.min():.4f}, Max={y_prob.max():.4f}, Mean={y_prob.mean():.4f}")
    
    print("\n>>> Threshold Scan (For Up/1 Precision):")
    print(f"{'Threshold':<10} | {'Precision':<10} | {'Recall':<10} | {'Trades':<10}")
    print("-" * 50)
    
    start_scan = max(0.3, float(y_prob.mean()))
    end_scan = min(0.95, float(y_prob.max()))
    if end_scan <= start_scan: end_scan = start_scan + 0.1
    
    scan_range = np.arange(start_scan, end_scan + 0.05, 0.01)
    
    for thresh in scan_range:
        y_tmp = (y_prob > thresh).astype(int)
        prec = precision_score(y_test, y_tmp, pos_label=1, zero_division=0)
        rec = recall_score(y_test, y_tmp, pos_label=1, zero_division=0)
        count = y_tmp.sum()
        if count > 0:
            print(f"{thresh:.4f}     | {prec:.4f}     | {rec:.4f}     | {count:<10}")

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


    save_dir = os.path.dirname(data_cfg.get('root', '.'))
    if 'checkpoints' in data_cfg: save_dir = data_cfg['checkpoints']
    elif 'root' in data_cfg: save_dir = data_cfg['root']
    if not os.path.exists(save_dir): os.makedirs(save_dir, exist_ok=True)
    
    save_path = os.path.join(save_dir, 'lgbm_model_updown.txt')
    model.save_model(save_path)
    print(f"[Output] Model saved to: {save_path}")

if __name__ == "__main__":
    main()