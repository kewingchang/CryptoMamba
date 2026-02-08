import pandas as pd
import numpy as np
import lightgbm as lgb
import yaml
import argparse
import sys
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import TimeSeriesSplit

# ==========================================
# 辅助配置
# ==========================================
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
    """删除微小波动的样本"""
    if close_chg_col not in df.columns:
        print(f"[Warning] Column '{close_chg_col}' not found. Skipping noise filtering.")
        return df
    
    print(f"[Preprocessing] Filtering noise (|Next_Day_Chg| < {threshold*100}%)...")
    original_len = len(df)
    
    # 检查 Next Day 的波动 (Shift -1)
    next_day_chg = df[close_chg_col].shift(-1)
    
    mask = (next_day_chg.abs() >= threshold) | (next_day_chg.isna())
    df_filtered = df.loc[mask].reset_index(drop=True)
    
    print(f"  -> Removed {original_len - len(df_filtered)} noise samples. Remaining: {len(df_filtered)}")
    return df_filtered

def get_dev_data(df, data_cfg):
    start = data_cfg['train_interval'][0]
    end = data_cfg['val_interval'][1]
    mask = (df['Date'] >= start) & (df['Date'] < end)
    return df.loc[mask].reset_index(drop=True)

# ==========================================
# 核心评估逻辑
# ==========================================
def evaluate_and_rank(X, y, params):
    tscv = TimeSeriesSplit(n_splits=5)
    scores = []
    feature_importances = np.zeros(X.shape[1])
    
    run_params = params.copy()
    run_params['objective'] = 'binary'
    run_params['metric'] = 'auc'
    run_params['verbose'] = -1
    run_params['n_jobs'] = -1
    run_params.pop('num_class', None)
    run_params.pop('class_weight', None)

    for train_idx, val_idx in tscv.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        lgb_train = lgb.Dataset(X_train, y_train)
        lgb_val = lgb.Dataset(X_val, y_val, reference=lgb_train)
        
        bst = lgb.train(
            run_params,
            lgb_train,
            num_boost_round=1000,
            valid_sets=[lgb_val],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)]
        )
        
        preds_prob = bst.predict(X_val)
        try:
            s = roc_auc_score(y_val, preds_prob)
        except ValueError:
            s = 0.5
            
        scores.append(s)
        feature_importances += bst.feature_importance(importance_type='gain')
        
    return np.mean(scores), feature_importances / 5.0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_config", required=True)
    parser.add_argument("--params_config", required=True)
    parser.add_argument("--training_config", required=True)
    parser.add_argument("--step_size", type=int, default=5, required=False)
    parser.add_argument("--min_features", type=int, default=20, required=False)
    parser.add_argument("--close_chg_col", type=str, default="PF_Close_Chg", help="Column for daily return")
    args = parser.parse_args()

    data_cfg = load_yaml(args.data_config)
    params_cfg = load_yaml(args.params_config)
    train_cfg = load_yaml(args.training_config)

    df = load_data(data_cfg)
    
    # 执行噪音过滤
    df = filter_noise(df, args.close_chg_col, threshold=0.001)
    
    target_col = train_cfg['target']
    
    # excludes = ['Date', target_col, 'Open', 'High', 'Low', 'Close', 'Volume', 'marketCap']
    # excludes = ['Date', "Label_ATR", "label", "label_0.3", "label_0.5", "label_0.8",  "label_0.5_3cat", "label_0.5_4cat", "label_0.5_Long_inclusive", "label_0.5_Short_inclusive", "close_chg_label"]
    # all_features = [c for c in df.columns if c not in excludes and not c.startswith('label_')]
    # all_features = list(set(all_features))
    fixed_feats = data_cfg.get('fixed_features', []) or []
    add_feats = data_cfg.get('additional_features', []) or []
    all_features = fixed_feats + add_feats
    all_features = list(set(all_features))

    dev_df = get_dev_data(df, data_cfg)
    y = dev_df[target_col].astype(int)
    
    current_features = all_features.copy()
    history = []
    
    print(f"[Step-wise RFE] Target: Global AUC. Features: {len(current_features)}")
    print("-" * 60)
    print(f"{'Iter':<5} | {'Num Feats':<10} | {'CV AUC':<20} | {'Action'}")
    print("-" * 60)
    
    iteration = 0
    while len(current_features) >= args.min_features:
        iteration += 1
        X = dev_df[current_features]
        score, importances = evaluate_and_rank(X, y, params_cfg)
        
        history.append({'n_feats': len(current_features), 'score': score, 'features': current_features.copy()})
        print(f"{iteration:<5} | {len(current_features):<10} | {score:.4f}               | ", end="")
        
        feat_imp_df = pd.DataFrame({'feature': current_features, 'imp': importances}).sort_values('imp', ascending=True)
        
        zero_imp_feats = feat_imp_df[feat_imp_df['imp'] == 0]['feature'].tolist()
        # if len(zero_imp_feats) >= args.step_size:
        #     to_drop = zero_imp_feats
        #     msg = f"Drop {len(to_drop)} (Zero Imp)"
        # else:
        #     to_drop = feat_imp_df.head(args.step_size)['feature'].tolist()
        #     msg = f"Drop bottom {len(to_drop)}"
        # drop one by one
        to_drop = feat_imp_df.head(args.step_size)['feature'].tolist()
        msg = f"Drop bottom {len(to_drop)}"
            
        print(msg)
        current_features = [f for f in current_features if f not in to_drop]
        if len(to_drop) == 0: break

    print("-" * 60)
    best_record = max(history, key=lambda x: x['score'])
    print(f"Best CV AUC: {best_record['score']:.4f} with {best_record['n_feats']} features.")
    
    with open('suggested_features.yaml', 'w') as f:
        yaml.dump({'fixed_features': best_record['features']}, f)
    print(f"Saved to 'suggested_features.yaml'.")

if __name__ == "__main__":
    main()