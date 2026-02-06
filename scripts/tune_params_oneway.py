# tune_params_oneway.py
import os, sys, pathlib
sys.path.insert(0, os.path.dirname(pathlib.Path(__file__).parent.absolute()))

import pandas as pd
import numpy as np
import lightgbm as lgb
import yaml
import argparse
import sys
import os
import optuna
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from utils.io_tools import load_yaml, load_data, save_yaml


# ==========================================
# 辅助函数
# ==========================================
def validate_and_prepare(df, data_cfg, train_cfg):
    """数据校验与特征提取"""
    fixed_feats = data_cfg.get('fixed_features', [])
    add_feats = data_cfg.get('additional_features', []) or []
    target_col = train_cfg.get('target')

    if not fixed_feats or not target_col:
        print("[Fatal] Config missing fixed_features or target.")
        sys.exit(1)

    all_cols = fixed_feats + add_feats + [target_col]
    missing = [c for c in all_cols if c not in df.columns]
    if missing:
        print(f"[Fatal] Missing columns in CSV: {missing}")
        sys.exit(1)
        
    return fixed_feats + add_feats, target_col

def get_slice(df, interval):
    mask = (df['Date'] >= interval[0]) & (df['Date'] < interval[1])
    return df.loc[mask]

# ==========================================
# Optuna 核心逻辑
# ==========================================
class ObjectiveCV:
    '''
    Nested Cross-Validation / Walk-Forward Optimization (Binary Version)
    '''
    def __init__(self, X_dev, y_dev, base_params, train_cfg, direction):
        self.X = X_dev
        self.y = y_dev
        self.base_params = base_params
        self.train_cfg = train_cfg
        self.direction = direction

    def __call__(self, trial):
        # ============================================================
        # 【核心修改】优化后的搜索空间 (更激进，更灵活)
        # ============================================================
        param_grid = {
            # 1. 树结构：允许更大的树
            "num_leaves": trial.suggest_int("num_leaves", 30, 150), 
            "max_depth": trial.suggest_int("max_depth", 5, 12),
            "min_child_samples": trial.suggest_int("min_child_samples", 20, 100),
            
            # 2. 学习率：稍微放宽
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 1000, 8000),
            
            # 3. 随机性：保持原样
            "subsample": trial.suggest_float("subsample", 0.6, 0.95),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 0.95),
            
            # 4. 正则化：大幅降低上限！防止模型"躺平"
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 2.0), # 之前是 10.0
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 2.0), # 之前是 10.0
            
            # 5. 【新增】正样本权重：这对 Recall/Precision 平衡至关重要
            # 1.0 = 平衡, >1.0 = 重视正样本(Recall↑ Precision↓), <1.0 = 重视负样本(Recall↓ Precision↑)
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 0.5, 5.0)
        }
        
        # 2. 合并参数
        params = self.base_params.copy()
        params.update(param_grid)
        
        # 强制二分类设置
        params['objective'] = 'binary'
        params['metric'] = 'auc'  # 调参时优化 AUC，因为它衡量排序能力，不受阈值影响
        params['verbose'] = -1
        params['n_jobs'] = -1
        
        # 清理多分类遗留参数
        params.pop('num_class', None)
        params.pop('class_weight', None) 
        # 注意：scale_pos_weight 会保留 base_params 中的设置，这正是我们想要的

        # 3. 5折时间序列交叉验证
        tscv = TimeSeriesSplit(n_splits=5)
        scores = []
        
        # 加上 enumerate 来获取当前是第几折
        for i, (train_index, val_index) in enumerate(tscv.split(self.X)):
            X_train_fold, X_val_fold = self.X.iloc[train_index], self.X.iloc[val_index]
            y_train_fold, y_val_fold = self.y.iloc[train_index], self.y.iloc[val_index]
            
            lgb_train = lgb.Dataset(X_train_fold, label=y_train_fold)
            lgb_val = lgb.Dataset(X_val_fold, label=y_val_fold, reference=lgb_train)
            
            callbacks = []
            
            # 只在第 0 折 (第一折) 添加 Optuna 剪枝回调
            # 这样后面的折数就不会重复报告 step 0, step 1... 导致冲突了
            if i == 0:
                callbacks.append(optuna.integration.LightGBMPruningCallback(trial, "auc"))

            if self.train_cfg.get('early_stop', False):
                rounds = self.train_cfg.get('stopping_rounds', 50)
                callbacks.append(lgb.early_stopping(stopping_rounds=rounds, verbose=False))

            bst = lgb.train(
                params,
                lgb_train,
                num_boost_round=params['n_estimators'],
                valid_sets=[lgb_val],
                callbacks=callbacks
            )
            
            # 预测验证 Fold (概率)
            preds_prob = bst.predict(X_val_fold)
            
            # 评分指标：AUC
            try:
                score = roc_auc_score(y_val_fold, preds_prob)
            except ValueError:
                score = 0.5 # 防止某折只有一个类别报错
                
            scores.append(score)
            
        # 4. 返回平均 AUC
        return np.mean(scores)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_config", required=True)
    parser.add_argument("--params_config", required=True)
    parser.add_argument("--training_config", required=True)
    # 新增方向参数
    parser.add_argument("--direction", required=True, choices=['Long', 'Short'], 
                        help="Tuning for 'Long' or 'Short' direction")
    
    try: args = parser.parse_args()
    except: args, _ = parser.parse_known_args()

    print("-" * 40)
    print(f"Starting Hyperparameter Tuning (Optuna) for [{args.direction}]")
    print("-" * 40)

    # 1. 加载配置
    data_cfg = load_yaml(args.data_config)
    params_cfg = load_yaml(args.params_config)
    train_cfg = load_yaml(args.training_config)

    # 2. 准备数据
    df = load_data(
        path=data_cfg['data_path'], 
        date_format=data_cfg.get('date_format')
    )
    feature_cols, target_col = validate_and_prepare(df, data_cfg, train_cfg)
    n_trials = train_cfg.get('n_trials', 100)

    # 3. 合并 Train 和 Val 作为 "开发集"
    dev_df = pd.concat([
        get_slice(df, data_cfg['train_interval']),
        get_slice(df, data_cfg['val_interval'])
    ]).sort_values('Date').reset_index(drop=True)
    
    # 4. Label 映射 (One-vs-Rest)
    target_val = 1 if args.direction == 'Long' else 2
    print(f"[Data] Mapping Label {target_val} to 1 (Positive), others to 0.")
    
    X_dev = dev_df[feature_cols]
    y_dev = (dev_df[target_col] == target_val).astype(int) # 转换为 0/1
    
    # 5. 启动调参
    objective = ObjectiveCV(X_dev, y_dev, params_cfg, train_cfg, args.direction)
    
    # AUC 也是越大越好
    study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner())
    
    print(f"[Optuna] Running {n_trials} trials...")
    study.optimize(objective, n_trials=n_trials)

    # 6. 结果处理
    print("\n" + "=" * 40)
    print(f"Best Trial AUC: {study.best_value:.4f}")
    print("Best Params:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")
    
    # 7. 更新并保存 params.yaml
    new_params = params_cfg.copy()
    new_params.update(study.best_params)
    
    # 强制修正一些可能被覆盖的关键结构参数
    new_params['objective'] = 'binary'
    new_params['metric'] = 'auc'
    # 移除不需要的参数
    new_params.pop('class_weight', None)
    new_params.pop('num_class', None)
    
    save_yaml(new_params, args.params_config)
    print("=" * 40)

if __name__ == "__main__":
    main()