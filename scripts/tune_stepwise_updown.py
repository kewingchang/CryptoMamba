import os, sys, pathlib
sys.path.insert(0, os.path.dirname(pathlib.Path(__file__).parent.absolute()))

import pandas as pd
import numpy as np
import lightgbm as lgb
import argparse
import sys
import optuna
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from utils.io_tools import load_yaml, load_data, save_yaml

def validate_and_prepare(df, data_cfg, train_cfg):
    fixed_feats = data_cfg.get('fixed_features', [])
    add_feats = data_cfg.get('additional_features', []) or []
    target_col = train_cfg.get('target')

    if not fixed_feats or not target_col:
        sys.exit(1)
    
    all_cols = fixed_feats + add_feats
    if target_col in all_cols:
        all_cols = [f for f in all_cols if f != target_col]
        
    return all_cols, target_col

def filter_noise(df, close_chg_col, threshold=0.001):
    if close_chg_col not in df.columns:
        return df
    print(f"[Tuning] Filtering noise (|Next_Day_Chg| < {threshold*100}%)...")
    next_day_chg = df[close_chg_col].shift(-1)
    mask = (next_day_chg.abs() >= threshold) | (next_day_chg.isna())
    return df.loc[mask].reset_index(drop=True)

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
            # 适当放宽叶子数，加入学习率
            params['num_leaves'] = trial.suggest_int('num_leaves', 10, 50)
            params['max_depth'] = trial.suggest_int('max_depth', 3, 8)
            params['min_child_samples'] = trial.suggest_int('min_child_samples', 20, 100)
            params['learning_rate'] = trial.suggest_float('learning_rate', 0.005, 0.1, log=True)
            
        elif self.group_name == 'regularization':
            # 大幅提高正则化上限，适应高噪音数据
            params['reg_alpha'] = trial.suggest_float('reg_alpha', 0.0, 10.0)
            params['reg_lambda'] = trial.suggest_float('reg_lambda', 0.0, 10.0)
            
        elif self.group_name == 'sampling':
            params['subsample'] = trial.suggest_float('subsample', 0.5, 0.95)
            params['colsample_bytree'] = trial.suggest_float('colsample_bytree', 0.5, 0.95)
            
        elif self.group_name == 'weight':
            params['scale_pos_weight'] = trial.suggest_float('scale_pos_weight', 0.8, 1.2)

        params['objective'] = 'binary'
        params['metric'] = 'auc'
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
            lgb_val = lgb.Dataset(X_val_fold, y_val_fold, reference=lgb_train)
            
            callbacks = []
            if i == 0: 
                callbacks.append(optuna.integration.LightGBMPruningCallback(trial, "auc"))

            if self.train_cfg.get('early_stop', False):
                rounds = self.train_cfg.get('stopping_rounds', 50)
                callbacks.append(lgb.early_stopping(stopping_rounds=rounds, verbose=False))

            bst = lgb.train(
                params,
                lgb_train,
                num_boost_round=params.get('n_estimators', 10000), # 确保这里足够大
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
    parser.add_argument("--group", required=True, choices=['structure', 'regularization', 'sampling', 'weight'])
    parser.add_argument("--close_chg_col", type=str, default="Feat_Close_Chg")
    parser.add_argument("--n_trials", type=int, default=50)
    
    try: args = parser.parse_args()
    except: args, _ = parser.parse_known_args()

    print(f"Step-wise Tuning for [Up/Down Model]: Group [{args.group}]")

    data_cfg = load_yaml(args.data_config)
    params_cfg = load_yaml(args.params_config)
    train_cfg = load_yaml(args.training_config)
    # df = load_data(data_cfg['data_path'], data_cfg.get('date_format'))
    df = load_data(path=data_cfg['data_path'], date_format=data_cfg.get('date_format'))
    df = filter_noise(df, args.close_chg_col, threshold=0.001)

    feature_cols, target_col = validate_and_prepare(df, data_cfg, train_cfg)
    
    dev_df = pd.concat([
        get_slice(df, data_cfg['train_interval']),
        get_slice(df, data_cfg['val_interval'])
    ]).sort_values('Date').reset_index(drop=True)
    
    X_dev = dev_df[feature_cols]
    y_dev = dev_df[target_col].astype(int)

    objective = StepwiseObjective(X_dev, y_dev, params_cfg, train_cfg, args.group)
    study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner())
    study.optimize(objective, n_trials=args.n_trials)

    print(f"Best CV AUC: {study.best_value:.4f}")
    
    new_params = params_cfg.copy()
    new_params.update(study.best_params)
    save_yaml(new_params, args.params_config)

if __name__ == "__main__":
    main()