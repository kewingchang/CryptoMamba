import pandas as pd
import numpy as np
import lightgbm as lgb
import yaml
import argparse
import sys
import os
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, accuracy_score
from sklearn.model_selection import TimeSeriesSplit

# ==========================================
# 辅助配置
# ==========================================
# 每次删除多少个特征？建议 1 或 5。
# 1 最准但最慢，5 比较平衡。
STEP_SIZE = 5
MIN_FEATURES = 20 # 删到剩多少个为止

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
    """合并 Train + Val 作为 Dev Set"""
    start = data_cfg['train_interval'][0]
    end = data_cfg['val_interval'][1]
    mask = (df['Date'] >= start) & (df['Date'] < end)
    return df.loc[mask].reset_index(drop=True)

# ==========================================
# 核心评估逻辑
# ==========================================

def evaluate_and_rank(X, y, params):
    """
    训练并返回：CV分数, 特征重要性
    """
    # 5折交叉验证
    tscv = TimeSeriesSplit(n_splits=5)
    scores = []
    
    # 记录特征重要性 (累加)
    feature_importances = np.zeros(X.shape[1])
    
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
        
        bst = lgb.train(
            run_params,
            lgb_train,
            num_boost_round=1000,
            valid_sets=[lgb_val],
            callbacks=[lgb.early_stopping(50, verbose=False)]
        )
        
        preds_prob = bst.predict(X_val)
        preds = [np.argmax(x) for x in preds_prob]
        
        # 核心指标：Long Precision
        # zero_division=0 防止除以零报错
        s = precision_score(y_val, preds, labels=[1], average='micro', zero_division=0)
        scores.append(s)
        
        # 累加 Importance (Gain)
        feature_importances += bst.feature_importance(importance_type='gain')
        
    avg_score = np.mean(scores)
    avg_importance = feature_importances / 5.0
    
    return avg_score, avg_importance

def main():
    parser = argparse.ArgumentParser()
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
    
    # 2. 准备候选特征
    # excludes = ['Date', target_col, 'Open', 'High', 'Low', 'Close', 'Volume', 'marketCap']
    excludes = ['Date', "Label_ATR", "label", "label_0.3", "label_0.5", "label_0.8"] 
    all_features = [c for c in df.columns if c not in excludes]
    all_features = list(set(all_features))
    
    dev_df = get_dev_data(df, data_cfg)
    y = dev_df[target_col].astype(int)
    
    current_features = all_features.copy()
    
    # 记录历史
    history = [] # [{num_feats, score, features_list}]
    
    print(f"[Step-wise RFE] Starting with {len(current_features)} features.")
    print(f"[Step-wise RFE] Step Size: Remove {STEP_SIZE} worst features per round.")
    print("-" * 60)
    print(f"{'Iter':<5} | {'Num Feats':<10} | {'CV Precision (Long)':<20} | {'Action'}")
    print("-" * 60)
    
    iteration = 0
    
    # 3. 循环剔除
    while len(current_features) >= MIN_FEATURES:
        iteration += 1
        X = dev_df[current_features]
        
        # 评估当前集合
        score, importances = evaluate_and_rank(X, y, params_cfg)
        
        # 记录
        history.append({
            'n_feats': len(current_features),
            'score': score,
            'features': current_features.copy()
        })
        
        print(f"{iteration:<5} | {len(current_features):<10} | {score:.4f}               | ", end="")
        
        # 找出最差的 STEP_SIZE 个特征
        feat_imp_df = pd.DataFrame({
            'feature': current_features,
            'imp': importances
        }).sort_values('imp', ascending=True)
        
        # 如果有 0 重要性的，优先删除 0 重要性的 (哪怕数量超过 STEP_SIZE)
        # 这样可以加速初期清理
        zero_imp_feats = feat_imp_df[feat_imp_df['imp'] == 0]['feature'].tolist()
        
        if len(zero_imp_feats) >= STEP_SIZE:
            to_drop = zero_imp_feats # 一次性把0的全删了，加速
            msg = f"Drop {len(to_drop)} (Zero Imp)"
        else:
            to_drop = feat_imp_df.head(STEP_SIZE)['feature'].tolist()
            msg = f"Drop bottom {len(to_drop)}"
            
        print(msg)
        
        # 更新特征列表
        current_features = [f for f in current_features if f not in to_drop]
        
        # 防止死循环 (如果没东西可删了)
        if len(to_drop) == 0:
            break

    # 4. 复盘寻找最佳点
    print("-" * 60)
    best_record = max(history, key=lambda x: x['score'])
    
    print(f"RFE Completed.")
    print(f"Best CV Score: {best_record['score']:.4f}")
    print(f"At Iteration with {best_record['n_feats']} features.")
    
    # 5. 保存结果
    output_yaml = {
        'fixed_features': best_record['features']
    }
    
    with open('suggested_features.yaml', 'w') as f:
        yaml.dump(output_yaml, f)
        
    print(f"Saved best feature set to 'suggested_features.yaml'.")
    
    # (可选) 绘制曲线
    try:
        scores = [h['score'] for h in history]
        n_feats = [h['n_feats'] for h in history]
        plt.figure(figsize=(10, 5))
        plt.plot(n_feats, scores, marker='o')
        plt.gca().invert_xaxis() # X轴反转，从多到少
        plt.xlabel('Number of Features')
        plt.ylabel('CV Precision (Long)')
        plt.title('Recursive Feature Elimination Path')
        plt.grid(True)
        plt.savefig('rfe_plot.png')
        print("Saved plot to 'rfe_plot.png'")
    except:
        pass

if __name__ == "__main__":
    main()