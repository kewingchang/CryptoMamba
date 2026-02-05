# tune_params.py
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
from sklearn.metrics import accuracy_score, log_loss
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
class Objective:
    def __init__(self, X_train, y_train, X_val, y_val, base_params, train_cfg):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.base_params = base_params
        self.train_cfg = train_cfg

    def __call__(self, trial):
        # 1. 定义搜索空间 (Search Space)
        # 这里覆盖了 params.yaml 中的核心参数
        param_grid = {
            "num_leaves": trial.suggest_int("num_leaves", 10, 100),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 500, 3000),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 10.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 10.0),
        }

        # 2. 合并参数 (优先使用搜索到的参数，其他保持默认)
        params = self.base_params.copy()
        params.update(param_grid)
        
        # 强制参数
        params['objective'] = 'multiclass'
        params['num_class'] = 3
        params['metric'] = 'multi_logloss' # 优化目标通常用 LogLoss 会比 Accuracy 更平滑
        params['verbose'] = -1
        params['n_jobs'] = -1

        # 3. 训练
        train_data = lgb.Dataset(self.X_train, label=self.y_train)
        val_data = lgb.Dataset(self.X_val, label=self.y_val, reference=train_data)

        callbacks = []
        # Optuna 自带剪枝 (Pruning) 回调，可以加速搜索，提前杀掉表现不好的 Trial
        callbacks.append(optuna.integration.LightGBMPruningCallback(trial, "multi_logloss"))
        
        if self.train_cfg.get('early_stop', False):
            rounds = self.train_cfg.get('stopping_rounds', 50)
            callbacks.append(lgb.early_stopping(stopping_rounds=rounds, verbose=False))

        bst = lgb.train(
            params,
            train_data,
            num_boost_round=params['n_estimators'],
            valid_sets=[val_data],
            callbacks=callbacks
        )

        # 4. 预测与评分 (使用 Accuracy 作为最终评价指标，虽然优化过程用LogLoss)
        preds_prob = bst.predict(self.X_val)
        preds = [np.argmax(x) for x in preds_prob]
        accuracy = accuracy_score(self.y_val, preds)
        
        return accuracy



class ObjectiveCV:
    '''
    Nested Cross-Validation / Walk-Forward Optimization
    '''
    def __init__(self, X_dev, y_dev, base_params, train_cfg):
        self.X = X_dev
        self.y = y_dev
        self.base_params = base_params
        self.train_cfg = train_cfg

    def __call__(self, trial):
        # 1. 定义搜索空间
        param_grid = {
            "num_leaves": trial.suggest_int("num_leaves", 10, 100),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 500, 3000),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 10.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 10.0),
        }
        
        # 2. 合并参数
        params = self.base_params.copy()
        params.update(param_grid)
        params['objective'] = 'multiclass'
        params['num_class'] = 3
        params['metric'] = 'multi_logloss'
        params['verbose'] = -1
        params['n_jobs'] = -1

        # 3. 5折时间序列交叉验证
        # n_splits=5 意味着数据会被切成 5 份，逐步扩大训练窗口
        tscv = TimeSeriesSplit(n_splits=5)
        scores = []
        
        # 传入 X, y 进行切分
        for train_index, val_index in tscv.split(self.X):
            # iloc 是关键，它按位置索引，不受原始 index 影响
            X_train_fold, X_val_fold = self.X.iloc[train_index], self.X.iloc[val_index]
            y_train_fold, y_val_fold = self.y.iloc[train_index], self.y.iloc[val_index]
            
            train_data = lgb.Dataset(X_train_fold, label=y_train_fold)
            val_data = lgb.Dataset(X_val_fold, label=y_val_fold, reference=train_data)
            
            callbacks = []
            
            # 在 CV 内部也可以用 Early Stopping
            if self.train_cfg.get('early_stop', False):
                rounds = self.train_cfg.get('stopping_rounds', 50)
                callbacks.append(lgb.early_stopping(stopping_rounds=rounds, verbose=False))

            bst = lgb.train(
                params,
                train_data,
                num_boost_round=params['n_estimators'],
                valid_sets=[val_data],
                callbacks=callbacks
            )
            
            # 预测验证 Fold
            preds_prob = bst.predict(X_val_fold)
            preds = [np.argmax(x) for x in preds_prob]
            acc = accuracy_score(y_val_fold, preds)
            scores.append(acc)
            
        # 4. 返回 5 次验证的平均分
        return np.mean(scores)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_config", required=True)
    parser.add_argument("--params_config", required=True)
    parser.add_argument("--training_config", required=True)
    
    try: args = parser.parse_args()
    except: args, _ = parser.parse_known_args()

    print("-" * 40)
    print("Starting Hyperparameter Tuning (Optuna)")
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
    n_trials = train_cfg.get('n_trials', 50)

    # 3. 切分 (只用 Train 和 Val 进行调参，Test 留给最终测试)
    # train_df = get_slice(df, data_cfg['train_interval'])
    # val_df = get_slice(df, data_cfg['val_interval'])
    
    # if len(train_df) == 0 or len(val_df) == 0:
    #     print("[Fatal] Train or Val set is empty.")
    #     sys.exit(1)

    # print(f"[Data] Train: {len(train_df)} | Val: {len(val_df)}")
    # print(f"[Features] {len(feature_cols)} features used.")

    # X_train = train_df[feature_cols]
    # y_train = train_df[target_col].astype(int)
    # X_val = val_df[feature_cols]
    # y_val = val_df[target_col].astype(int)

    # # 4. Optuna 搜索
    # # direction="maximize" 因为我们返回的是 Accuracy
    # study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner())
    # objective = Objective(X_train, y_train, X_val, y_val, params_cfg, train_cfg)
    # study.optimize(objective, n_trials=n_trials)

    # 3. 合并 Train 和 Val 作为 "开发集"
    # Test 集依然雷打不动，坚决不参与调参
    dev_df = pd.concat([
        get_slice(df, data_cfg['train_interval']),
        get_slice(df, data_cfg['val_interval'])
    ]).sort_values('Date').reset_index(drop=True)
    
    X_dev = dev_df[feature_cols]
    y_dev = dev_df[target_col].astype(int)
    
    # 启动 CV 调参
    objective = ObjectiveCV(X_dev, y_dev, params_cfg, train_cfg)
    study = optuna.create_study(direction="maximize")
    
    print(f"[Optuna] Running {n_trials} trials...")
    study.optimize(objective, n_trials=n_trials)

    # 5. 结果处理
    print("\n" + "=" * 40)
    print(f"Best Trial Accuracy: {study.best_value:.4f}")
    print("Best Params:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")
    
    # 6. 更新并保存 params.yaml
    # 将最优参数覆盖原来的 params_cfg
    new_params = params_cfg.copy()
    new_params.update(study.best_params)
    
    # 保存回文件
    save_yaml(new_params, args.params_config)
    print("=" * 40)

if __name__ == "__main__":
    main()