import os, sys, pathlib
sys.path.insert(0, os.path.dirname(pathlib.Path(__file__).parent.absolute()))

import pandas as pd
import numpy as np
import lightgbm as lgb
import yaml
import argparse
import sys
import optuna
from sklearn.metrics import average_precision_score 
from sklearn.model_selection import TimeSeriesSplit
from utils.io_tools import load_yaml, load_data, save_yaml

def validate_and_prepare(df, data_cfg, train_cfg):
    fixed_feats = data_cfg.get('fixed_features', [])
    add_feats = data_cfg.get('additional_features', []) or []
    target_col = train_cfg.get('target')
    if not fixed_feats or not target_col:
        sys.exit(1)
    all_cols = fixed_feats + add_feats + [target_col]
    return fixed_feats + add_feats, target_col

def get_slice(df, interval):
    mask = (df['Date'] >= interval[0]) & (df['Date'] < interval[1])
    return df.loc[mask]

class StepwiseObjective:
    def __init__(self, X_dev, y_dev, current_params, train_cfg, group_name):
        self.X = X_dev
        self.y = y_dev
        self.base_params = current_params.copy()
        self.train_cfg = train_cfg
        self.group_name = group_name

    def __call__(self, trial):
        params = self.base_params.copy()
        
        if self.group_name == 'structure':
            params['num_leaves'] = trial.suggest_int('num_leaves', 20, 80) # 不要太大
            params['max_depth'] = trial.suggest_int('max_depth', 4, 10)
            params['min_child_samples'] = trial.suggest_int('min_child_samples', 10, 50) # 保持较小
            
        elif self.group_name == 'regularization':
            # 【修改点 3】大幅降低正则化上限，防止模型躺平
            params['reg_alpha'] = trial.suggest_float('reg_alpha', 0.0, 1.5)
            params['reg_lambda'] = trial.suggest_float('reg_lambda', 0.0, 1.5)
            
        elif self.group_name == 'sampling':
            params['subsample'] = trial.suggest_float('subsample', 0.5, 0.9)
            params['colsample_bytree'] = trial.suggest_float('colsample_bytree', 0.5, 0.9)
            
        elif self.group_name == 'weight':
            # 【修改点 2】对于 Long 模型，我们只允许降低权重（提纯），不允许增加权重（注水）
            # 搜索范围 0.5 ~ 1.2
            params['scale_pos_weight'] = trial.suggest_float('scale_pos_weight', 0.5, 1.2)

        params['objective'] = 'binary'
        params['metric'] = 'auc' # 训练过程依然监控 AUC，这没问题
        params['verbose'] = -1
        params['n_jobs'] = -1
        
        params.pop('num_class', None)
        params.pop('class_weight', None)

        tscv = TimeSeriesSplit(n_splits=5)
        scores = []
        
        for i, (train_index, val_index) in enumerate(tscv.split(self.X)):
            X_train_fold, X_val_fold = self.X.iloc[train_index], self.X.iloc[val_index]
            y_train_fold, y_val_fold = self.y.iloc[train_index], self.y.iloc[val_index]
            
            lgb_train = lgb.Dataset(X_train_fold, label=y_train_fold)
            lgb_val = lgb.Dataset(X_val_fold, label=y_val_fold, reference=lgb_train)
            
            callbacks = []
            # Average Precision 对 Pruning 不太友好，这里暂时去掉 Pruning 或者改用 auc prune
            # 为了简单，这里去掉 pruning 以保证计算完整准确
            
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
            
            # 【修改点 1】优化目标改为 Average Precision Score
            # 这会强迫模型去优化"置信度排序"，让高分样本更准
            score = average_precision_score(y_val_fold, preds_prob)
            scores.append(score)
            
        return np.mean(scores)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_config", required=True)
    parser.add_argument("--params_config", required=True)
    parser.add_argument("--training_config", required=True)
    parser.add_argument("--direction", required=True, choices=['Long', 'Short'])
    parser.add_argument("--group", required=True, choices=['structure', 'regularization', 'sampling', 'weight'])
    parser.add_argument("--n_trials", type=int, default=50)
    
    try: args = parser.parse_args()
    except: args, _ = parser.parse_known_args()

    print(f"Step-wise Tuning V2 (AP Optimized): Group [{args.group}] for [{args.direction}]")

    params_cfg = load_yaml(args.params_config)
    data_cfg = load_yaml(args.data_config)
    train_cfg = load_yaml(args.training_config)
    df = load_data(data_cfg['data_path'], data_cfg.get('date_format'))
    feature_cols, target_col = validate_and_prepare(df, data_cfg, train_cfg)
    
    dev_df = pd.concat([
        get_slice(df, data_cfg['train_interval']),
        get_slice(df, data_cfg['val_interval'])
    ]).sort_values('Date').reset_index(drop=True)
    
    target_val = 1 if args.direction == 'Long' else 2
    print(f"[Data] Mapping Label {target_val} to 1.")
    
    X_dev = dev_df[feature_cols]
    y_dev = (dev_df[target_col] == target_val).astype(int)

    objective = StepwiseObjective(X_dev, y_dev, params_cfg, train_cfg, args.group)
    study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner())
    study.optimize(objective, n_trials=args.n_trials)

    print(f"Best Average Precision: {study.best_value:.4f}")
    print(f"Best Params for Group [{args.group}]:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")
    
    new_params = params_cfg.copy()
    new_params.update(study.best_params)
    save_yaml(new_params, args.params_config)

if __name__ == "__main__":
    main()