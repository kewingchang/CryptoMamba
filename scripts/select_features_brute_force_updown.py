import pandas as pd
import numpy as np
import lightgbm as lgb
import yaml
import argparse
import sys
import os
import copy
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from joblib import Parallel, delayed

MIN_FEATURES_TO_KEEP = 10 
N_JOBS = -1 
CV_SPLITS = 5

def load_yaml(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        print(f"[Fatal] Error loading config {filepath}: {e}")
        sys.exit(1)

def load_data(data_config):
    df = pd.read_csv(data_config['data_path'])
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], format=data_config.get('date_format'))
        df = df.sort_values('Date').reset_index(drop=True)
    return df

def filter_noise(df, close_chg_col, threshold=0.001):
    if close_chg_col not in df.columns:
        print(f"[Warning] Column '{close_chg_col}' not found. Skipping noise filtering.")
        return df
    print(f"[Preprocessing] Filtering noise (|Next_Day_Chg| < {threshold*100}%)...")
    next_day_chg = df[close_chg_col].shift(-1)
    mask = (next_day_chg.abs() >= threshold) | (next_day_chg.isna())
    return df.loc[mask].reset_index(drop=True)

def get_dev_data(df, data_cfg):
    start = data_cfg['train_interval'][0]
    end = data_cfg['val_interval'][1]
    mask = (df['Date'] >= start) & (df['Date'] < end)
    return df.loc[mask].reset_index(drop=True)

def evaluate_feature_subset(features, X_df, y, params):
    X_subset = X_df[features]
    tscv = TimeSeriesSplit(n_splits=CV_SPLITS)
    scores = []
    
    run_params = params.copy()
    run_params['objective'] = 'binary'
    run_params['metric'] = 'auc'
    run_params['verbose'] = -1
    run_params['n_jobs'] = 1 
    run_params.pop('num_class', None)
    run_params.pop('class_weight', None)
    
    try:
        for train_idx, val_idx in tscv.split(X_subset):
            X_train, X_val = X_subset.iloc[train_idx], X_subset.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            lgb_train = lgb.Dataset(X_train, y_train)
            lgb_val = lgb.Dataset(X_val, y_val, reference=lgb_train)
            
            bst = lgb.train(
                run_params,
                lgb_train,
                num_boost_round=500, 
                valid_sets=[lgb_val],
                callbacks=[lgb.early_stopping(30, verbose=False)]
            )
            
            preds_prob = bst.predict(X_val)
            try:
                s = roc_auc_score(y_val, preds_prob)
            except ValueError:
                s = 0.5
            scores.append(s)
            
        return np.mean(scores)
    except Exception as e:
        return -1.0

def main():
    parser = argparse.ArgumentParser(description="Brute Force RFE (AUC Optimized)")
    parser.add_argument("--data_config", required=True)
    parser.add_argument("--params_config", required=True)
    parser.add_argument("--training_config", required=True)
    parser.add_argument("--close_chg_col", type=str, default="Feat_Close_Chg")
    args = parser.parse_args()

    data_cfg = load_yaml(args.data_config)
    params_cfg = load_yaml(args.params_config)
    train_cfg = load_yaml(args.training_config)

    df = load_data(data_cfg)
    df = filter_noise(df, args.close_chg_col, threshold=0.001)

    target_col = train_cfg['target']
    
    # excludes = ['Date', target_col, 'Open', 'High', 'Low', 'Close', 'Volume', 'marketCap']
    excludes = ['Date', "Label_ATR", "label", "label_0.3", "label_0.5", "label_0.8",  "label_0.5_3cat", "label_0.5_4cat", "label_0.5_Long_inclusive", "label_0.5_Short_inclusive", "close_chg_label"]
    all_candidate_cols = [c for c in df.columns if c not in excludes and not c.startswith('label_')]
    all_candidate_cols = list(set(all_candidate_cols))
    
    dev_df = get_dev_data(df, data_cfg)
    y_dev = dev_df[target_col].astype(int)
    
    current_features = all_candidate_cols.copy()
    
    print(f"[Brute Force RFE] Target: Global AUC. Features: {len(current_features)}")
    
    initial_score = evaluate_feature_subset(current_features, dev_df, y_dev, params_cfg)
    print(f"Initial Baseline AUC: {initial_score:.5f}")
    
    best_global_score = initial_score
    best_global_features = current_features.copy()
    
    while len(current_features) > MIN_FEATURES_TO_KEEP:
        print(f"\n--- Round: {len(current_features)} features -> Eliminating 1 ---")
        
        tasks = []
        for feat_to_remove in current_features:
            remaining_feats = [f for f in current_features if f != feat_to_remove]
            tasks.append((feat_to_remove, remaining_feats))
            
        results = Parallel(n_jobs=N_JOBS)(
            delayed(evaluate_feature_subset)(task[1], dev_df, y_dev, params_cfg) 
            for task in tasks
        )
        
        max_score = -1.0
        feature_to_drop = None
        
        for i, score in enumerate(results):
            if score > max_score:
                max_score = score
                feature_to_drop = tasks[i][0]
        
        current_features.remove(feature_to_drop)
        diff = max_score - best_global_score
        print(f"   > Dropped: {feature_to_drop}")
        print(f"   > New AUC: {max_score:.5f} (Diff: {diff:+.5f})")
        
        if max_score >= best_global_score:
            best_global_score = max_score
            best_global_features = current_features.copy()
            print(f"   >>> [Log] Removing '{feature_to_drop}' led to New Best AUC.")
            with open('suggested_features_brute.yaml', 'w') as f:
                yaml.dump({'fixed_features': best_global_features}, f)
        
    print("\n" + "="*40)
    print(f"Best AUC: {best_global_score:.5f}")

if __name__ == "__main__":
    main()
