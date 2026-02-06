import os, sys, pathlib
sys.path.insert(0, os.path.dirname(pathlib.Path(__file__).parent.absolute()))

import pandas as pd
import numpy as np
import lightgbm as lgb
import yaml
import argparse
import sys
import optuna
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from utils.io_tools import load_yaml, load_data, save_yaml

# ==========================================
# 辅助函数
# ==========================================
def validate_and_prepare(df, data_cfg, train_cfg):
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
# Optuna Objective
# ==========================================
class StepwiseObjective:
    def __init__(self, X_dev, y_dev, current_params, train_cfg, group_name):
        self.X = X_dev
        self.y = y_dev
        self.base_params = current_params.copy()
        self.train_cfg = train_cfg
        self.group_name = group_name

    def __call__(self, trial):
        # 基于当前的基础参数
        params = self.base_params.copy()
        
        # 根据不同的 Group 覆盖特定的参数
        if self.group_name == 'structure':
            params['num_leaves'] = trial.suggest_int('num_leaves', 20, 100)
            params['max_depth'] = trial.suggest_int('max_depth', 5, 12)
            params['min_child_samples'] = trial.suggest_int('min_child_samples', 10, 60)
            
        elif self.group_name == 'regularization':
            params['reg_alpha'] = trial.suggest_float('reg_alpha', 0.0, 5.0)
            params['reg_lambda'] = trial.suggest_float('reg_lambda', 0.0, 5.0)
            
        elif self.group_name == 'sampling':
            params['subsample'] = trial.suggest_float('subsample', 0.5, 1.0)
            params['colsample_bytree'] = trial.suggest_float('colsample_bytree', 0.5, 1.0)
            
        elif self.group_name == 'weight':
            # 搜索范围：从 0.5 (抑制正类) 到 5.0 (极度重视正类)
            params['scale_pos_weight'] = trial.suggest_float('scale_pos_weight', 0.5, 5.0)

        # 强制配置
        params['objective'] = 'binary'
        params['metric'] = 'auc'
        params['verbose'] = -1
        params['n_jobs'] = -1
        
        # 清理杂项
        params.pop('num_class', None)
        params.pop('class_weight', None)

        # 5折交叉验证
        tscv = TimeSeriesSplit(n_splits=5)
        scores = []
        
        for i, (train_index, val_index) in enumerate(tscv.split(self.X)):
            X_train_fold, X_val_fold = self.X.iloc[train_index], self.X.iloc[val_index]
            y_train_fold, y_val_fold = self.y.iloc[train_index], self.y.iloc[val_index]
            
            lgb_train = lgb.Dataset(X_train_fold, label=y_train_fold)
            lgb_val = lgb.Dataset(X_val_fold, label=y_val_fold, reference=lgb_train)
            
            callbacks = []
            if i == 0: # 只在第一折剪枝
                callbacks.append(optuna.integration.LightGBMPruningCallback(trial, "auc"))

            if self.train_cfg.get('early_stop', False):
                rounds = self.train_cfg.get('stopping_rounds', 50)
                callbacks.append(lgb.early_stopping(stopping_rounds=rounds, verbose=False))

            bst = lgb.train(
                params,
                lgb_train,
                num_boost_round=params.get('n_estimators', 5000),
                valid_sets=[lgb_val],
                callbacks=callbacks
            )
            
            preds_prob = bst.predict(X_val_fold)
            try:
                score = roc_auc_score(y_val_fold, preds_prob)
            except ValueError:
                score = 0.5
            scores.append(score)
            
        return np.mean(scores)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_config", required=True)
    parser.add_argument("--params_config", required=True)
    parser.add_argument("--training_config", required=True)
    parser.add_argument("--direction", required=True, choices=['Long', 'Short'])
    # 新增 group 参数
    parser.add_argument("--group", required=True, 
                        choices=['structure', 'regularization', 'sampling', 'weight'],
                        help="Which parameter group to tune")
    parser.add_argument("--n_trials", type=int, default=50)
    
    args = parser.parse_args()

    print("-" * 50)
    print(f"Step-wise Tuning: Group [{args.group}] for [{args.direction}]")
    print("-" * 50)

    # 1. 加载当前参数 (作为基准)
    params_cfg = load_yaml(args.params_config)
    print(f"Current Base Params (Before Tuning):")
    print(params_cfg)

    # 2. 准备数据
    data_cfg = load_yaml(args.data_config)
    train_cfg = load_yaml(args.training_config)
    df = load_data(data_cfg['data_path'], data_cfg.get('date_format'))
    feature_cols, target_col = validate_and_prepare(df, data_cfg, train_cfg)
    
    dev_df = pd.concat([
        get_slice(df, data_cfg['train_interval']),
        get_slice(df, data_cfg['val_interval'])
    ]).sort_values('Date').reset_index(drop=True)
    
    target_val = 1 if args.direction == 'Long' else 2
    print(f"[Data] Mapping Label {target_val} to 1 (Positive), others to 0.")
    
    X_dev = dev_df[feature_cols]
    y_dev = (dev_df[target_col] == target_val).astype(int)

    # 3. 运行 Optuna
    objective = StepwiseObjective(X_dev, y_dev, params_cfg, train_cfg, args.group)
    study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner())
    study.optimize(objective, n_trials=args.n_trials)

    print("\n" + "=" * 40)
    print(f"Best AUC: {study.best_value:.4f}")
    print(f"Best Params for Group [{args.group}]:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")
    
    # 4. 更新参数并保存
    # 这一步至关重要：把这次找到的最佳参数更新到配置文件里，供下一步使用
    new_params = params_cfg.copy()
    new_params.update(study.best_params)
    
    save_yaml(new_params, args.params_config)
    print(f"[Output] Updated params saved to {args.params_config}")
    print("=" * 40)

if __name__ == "__main__":
    main()