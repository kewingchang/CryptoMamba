import pandas as pd
import numpy as np
import lightgbm as lgb
import yaml
import argparse
import sys
import os
import copy
from sklearn.metrics import precision_score
from sklearn.model_selection import TimeSeriesSplit
from joblib import Parallel, delayed

# ==========================================
# 配置区域
# ==========================================
# 最小保留特征数 (防止删光)
MIN_FEATURES_TO_KEEP = 10 

# 并行核心数 (-1 使用所有核心, -2 使用所有核心减1)
N_JOBS = -1 

# 交叉验证折数
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

def get_dev_data(df, data_cfg):
    """提取 Train + Val"""
    start = data_cfg['train_interval'][0]
    end = data_cfg['val_interval'][1]
    mask = (df['Date'] >= start) & (df['Date'] < end)
    return df.loc[mask].reset_index(drop=True)

# ==========================================
# 单次评估函数 (将被并行调用)
# ==========================================

def evaluate_feature_subset(features, X_df, y, params):
    """
    输入: 特征列表, 数据, 标签, 参数
    输出: 该特征组合下的 CV Long Precision
    """
    # 这里的 X_df 包含了所有列，我们只取需要的 features
    X_subset = X_df[features]
    
    tscv = TimeSeriesSplit(n_splits=CV_SPLITS)
    scores = []
    
    # 强制参数设置，确保训练快且静默
    run_params = params.copy()
    run_params['objective'] = 'multiclass'
    run_params['num_class'] = 3
    run_params['verbose'] = -1
    run_params['n_jobs'] = 1 # 单个模型内部不再并行，因为我们在外部已经并行了
    
    try:
        for train_idx, val_idx in tscv.split(X_subset):
            X_train, X_val = X_subset.iloc[train_idx], X_subset.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            lgb_train = lgb.Dataset(X_train, y_train)
            lgb_val = lgb.Dataset(X_val, y_val, reference=lgb_train)
            
            bst = lgb.train(
                run_params,
                lgb_train,
                num_boost_round=500, # 稍微减少轮数以加速
                valid_sets=[lgb_val],
                callbacks=[lgb.early_stopping(30, verbose=False)]
            )
            
            preds_prob = bst.predict(X_val)
            preds = [np.argmax(x) for x in preds_prob]
            
            # 核心目标: Precision of Class 1 (Long)
            s = precision_score(y_val, preds, labels=[1], average='micro', zero_division=0)
            scores.append(s)
            
        return np.mean(scores)
        
    except Exception as e:
        # 万一报错，返回 -1
        return -1.0

def main():
    parser = argparse.ArgumentParser(description="Brute Force Recursive Feature Elimination")
    parser.add_argument("--data_config", required=True)
    parser.add_argument("--params_config", required=True)
    parser.add_argument("--training_config", required=True)
    args = parser.parse_args()

    # 1. 加载
    data_cfg = load_yaml(args.data_config)
    params_cfg = load_yaml(args.params_config)
    train_cfg = load_yaml(args.training_config)

    df = load_data(data_cfg)
    target_col = train_cfg['target']
    
    # 2. 准备数据
    # 排除非特征列
    # excludes = ['Date', target_col, 'Open', 'High', 'Low', 'Close', 'Volume', 'marketCap']
    excludes = ['Date', "Label_ATR", "label", "label_0.3", "label_0.5", "label_0.8"] 
    all_candidate_cols = [c for c in df.columns if c not in excludes]
    all_candidate_cols = list( set(all_candidate_cols) )
    
    dev_df = get_dev_data(df, data_cfg)
    y_dev = dev_df[target_col].astype(int)
    
    current_features = all_candidate_cols.copy()
    
    # 3. 初始化基准
    print(f"[Brute Force RFE] Starting with {len(current_features)} features.")
    print(f"[Brute Force RFE] Using {N_JOBS} CPU cores.")
    
    # 计算初始分数
    initial_score = evaluate_feature_subset(current_features, dev_df, y_dev, params_cfg)
    print(f"Initial Baseline Score (Long Precision): {initial_score:.5f}")
    
    best_global_score = initial_score
    best_global_features = current_features.copy()
    
    # 4. 循环剔除
    while len(current_features) > MIN_FEATURES_TO_KEEP:
        n_feats = len(current_features)
        print(f"\n--- Round: {n_feats} features -> Eliminating 1 ---")
        
        # 构造这一轮的所有可能性列表
        # 每一个任务是: (尝试删除的特征名, 删除后的特征列表)
        tasks = []
        for feat_to_remove in current_features:
            remaining_feats = [f for f in current_features if f != feat_to_remove]
            tasks.append((feat_to_remove, remaining_feats))
            
        # 并行计算所有可能性的分数
        # results 是一个列表: [score_if_remove_A, score_if_remove_B, ...]
        results = Parallel(n_jobs=N_JOBS)(
            delayed(evaluate_feature_subset)(task[1], dev_df, y_dev, params_cfg) 
            for task in tasks
        )
        
        # 找到表现最好的那一次删除
        # 注意：我们要找 score 最高的，这意味着删掉该特征后模型效果最好（或下降最少）
        max_score = -1.0
        feature_to_drop = None
        
        for i, score in enumerate(results):
            if score > max_score:
                max_score = score
                feature_to_drop = tasks[i][0]
        
        # 执行删除
        current_features.remove(feature_to_drop)
        
        # 打印日志
        diff = max_score - best_global_score
        # 简单的进度条显示
        print(f"   > Dropped: {feature_to_drop}")
        print(f"   > New Score: {max_score:.5f} (Diff to Best: {diff:+.5f})")
        
        # 更新全局最佳
        # 策略: 只要分数比全局最高分没有低太多（或更高），我们都继续记录
        # 这里我们只记录绝对最高分，防止保存了局部最优
        if max_score >= best_global_score:
            best_global_score = max_score
            best_global_features = current_features.copy()
            print(f"   >>> NEW BEST FOUND! Saved to yaml.")
            
            # 实时保存，防止跑断了
            with open('suggested_features_brute.yaml', 'w') as f:
                yaml.dump({'fixed_features': best_global_features}, f)
        
    print("\n" + "="*40)
    print("Brute Force RFE Completed.")
    print(f"Best Score: {best_global_score:.5f}")
    print(f"Final Feature Count: {len(best_global_features)}")
    print("Optimization finished.")

if __name__ == "__main__":
    main()