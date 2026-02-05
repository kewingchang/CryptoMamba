import pandas as pd
import numpy as np
import lightgbm as lgb
import yaml
import argparse
import sys
import os
from sklearn.metrics import precision_score
from sklearn.model_selection import TimeSeriesSplit

# ==========================================
# 辅助函数
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

def get_dev_data(df, data_cfg):
    """合并 Train 和 Val 作为 Dev Set"""
    # 这里我们排除 Test 集，严防死守！
    start = data_cfg['train_interval'][0]
    end = data_cfg['val_interval'][1] # 结束于 Val 的末尾
    
    mask = (df['Date'] >= start) & (df['Date'] < end)
    dev_df = df.loc[mask].reset_index(drop=True)
    
    # 提取 Test 集仅用于 final report (用户自行决定看不看，代码逻辑里不依据它做决策)
    test_mask = (df['Date'] >= data_cfg['test_interval'][0]) & (df['Date'] < data_cfg['test_interval'][1])
    test_df = df.loc[test_mask].reset_index(drop=True)
    
    return dev_df, test_df

# ==========================================
# RFE 核心逻辑
# ==========================================

def evaluate_features(X, y, params, cv_splits=5):
    """
    使用 TimeSeriesSplit 交叉验证评估当前特征集的表现
    返回: 平均 Precision (Long), 特征重要性
    """
    tscv = TimeSeriesSplit(n_splits=cv_splits)
    scores = []
    feature_importances = np.zeros(X.shape[1])
    
    # 强制参数
    run_params = params.copy()
    run_params['objective'] = 'multiclass'
    run_params['num_class'] = 3
    run_params['verbose'] = -1
    run_params['n_jobs'] = -1
    
    for train_idx, val_idx in tscv.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        lgb_train = lgb.Dataset(X_train, y_train)
        lgb_val = lgb.Dataset(X_val, y_val, reference=lgb_train)
        
        # 训练
        bst = lgb.train(
            run_params,
            lgb_train,
            num_boost_round=1000,
            valid_sets=[lgb_val],
            callbacks=[
                lgb.early_stopping(50, verbose=False),
                lgb.log_evaluation(0) # 静默
            ]
        )
        
        # 预测
        preds_prob = bst.predict(X_val)
        preds = [np.argmax(x) for x in preds_prob]
        
        # 评分：只看 Long (Label 1) 的 Precision
        # 如果你想兼顾 Short，可以改 average='macro'
        score = precision_score(y_val, preds, labels=[1], average='micro', zero_division=0)
        scores.append(score)
        
        # 累加重要性 (Split gain is usually better than frequency)
        feature_importances += bst.feature_importance(importance_type='gain')
        
    return np.mean(scores), feature_importances / cv_splits

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_config", required=True)
    parser.add_argument("--params_config", required=True)
    parser.add_argument("--training_config", required=True)
    args = parser.parse_args()

    # 1. 加载配置
    print("-" * 30)
    data_cfg = load_yaml(args.data_config)
    params_cfg = load_yaml(args.params_config)
    train_cfg = load_yaml(args.training_config)

    # 2. 准备数据
    df = load_data(data_cfg)
    target_col = train_cfg['target']
    
    # 获取所有候选特征 (从 CSV 列中排除非特征列)
    # 这里我们不再读取配置文件里的 fixed_features，而是直接扫描 CSV
    # 假设除了 Date, target, 以及被排除的列，其他都是特征
    # excludes = ['Date', target_col, 'Open', 'High', 'Low', 'Close', 'Volume', 'marketCap'] 
    excludes = ['Date', "Label_ATR", "label", "label_0.3", "label_0.5", "label_0.8"] 
    candidate_features = [c for c in df.columns if c not in excludes]
    candidate_features = list(set(candidate_features))
    
    # 3. 获取开发集
    dev_df, _ = get_dev_data(df, data_cfg)
    y_dev = dev_df[target_col].astype(int)
    
    current_features = candidate_features.copy()
    best_score = -1
    best_features = []
    
    print(f"[RFE] Starting with {len(current_features)} features.")
    print(f"[RFE] Target Metric: Precision (Long) via 5-Fold TimeSeriesCV\n")
    
    # 4. 循环剔除
    iteration = 0
    while len(current_features) > 10: # 至少保留10个特征
        iteration += 1
        X_dev = dev_df[current_features]
        
        # 评估
        score, importances = evaluate_features(X_dev, y_dev, params_cfg)
        
        print(f"Iter {iteration}: Feats={len(current_features)}, CV_Score={score:.4f}")
        
        # 记录最佳状态
        # 允许微小的性能下降(0.005)以换取特征数量的大幅减少
        if score >= best_score - 0.005:
            if score > best_score:
                best_score = score
            best_features = current_features.copy()
        else:
            print(f"   -> Performance dropped significantly (Best: {best_score:.4f}). Stopping.")
            break
            
        # 策略：找出重要性最低的特征
        feat_imp_df = pd.DataFrame({
            'feature': current_features,
            'imp': importances
        }).sort_values('imp', ascending=True)
        
        # A. 首先删除所有重要性为 0 的特征
        zero_imp_feats = feat_imp_df[feat_imp_df['imp'] == 0]['feature'].tolist()
        
        if len(zero_imp_feats) > 0:
            print(f"   -> Removing {len(zero_imp_feats)} features with Zero Importance.")
            to_drop = zero_imp_feats
        else:
            # B. 如果没有0重要性的，删除末尾 10%
            n_drop = max(1, int(len(current_features) * 0.1)) 
            to_drop = feat_imp_df.head(n_drop)['feature'].tolist()
            print(f"   -> Removing {len(to_drop)} weakest features.")
            
        # 更新特征列表
        current_features = [f for f in current_features if f not in to_drop]

    # 5. 输出结果
    print("-" * 30)
    print(f"Done. Best CV Score: {best_score:.4f}")
    print(f"Selected Features: {len(best_features)} (Original: {len(candidate_features)})")
    
    # 保存为 YAML 格式片段，方便用户复制
    output_yaml = {
        'fixed_features': best_features
    }
    
    with open('suggested_features.yaml', 'w') as f:
        yaml.dump(output_yaml, f)
        
    print("Saved list to 'suggested_features.yaml'.")

if __name__ == "__main__":
    main()