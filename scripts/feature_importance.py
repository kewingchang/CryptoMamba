# feature_importance.py
import os
import sys
import pathlib
sys.path.insert(0, os.path.dirname(pathlib.Path(__file__).parent.absolute()))

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from argparse import ArgumentParser
from utils import io_tools
from pl_modules.data_module import CMambaDataModule
from data_utils.data_transforms import DataTransform
from pl_modules.cmamba_module import CryptoMambaModule

# 忽略部分警告
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

ROOT = io_tools.get_root(__file__, num_returns=2)

def get_args():
    parser = ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to the best checkpoint (.ckpt)")
    parser.add_argument("--config", type=str, default='cmamba_nv', help="Name of the config file used for training (e.g. cmamba_v)")
    parser.add_argument("--n_repeats", type=int, default=10, help="Number of times to permute each feature")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for inference")
    parser.add_argument("--output_file", type=str, default="feature_importance_results.csv", help="Output CSV path")
    return parser.parse_args()

def get_feature_names(config):
    """
    重建输入Tensor对应的特征名称列表，必须与 DataTransform 中的逻辑完全一致
    """
    # DataTransform 初始化时: self.keys = ['Timestamp', 'Open', 'High', 'Low', 'Close']
    # 注意：你的 Dataset 代码逻辑中，Timestamp 似乎也被作为 feature 加入了 Tensor
    # base_features = ['Timestamp', 'Open', 'High', 'Low', 'Close']
    base_features = ['Open', 'High', 'Low', 'Close']
    
    use_volume = config.get('use_volume', False)
    if use_volume:
        base_features.append('Volume')
        
    additional_features = config.get('additional_features', [])
    
    # 最终特征列表
    feature_names = base_features + additional_features
    return feature_names

def calculate_rmse(y_true, y_pred):
    return torch.sqrt(torch.mean((y_true - y_pred) ** 2)).item()

def load_model_with_config(checkpoint_path, train_config, feature_names):
    """
    加载模型结构配置，并结合 checkpoint 加载权重
    """
    # 1. 读取架构映射文件 (configs/models/archs.yaml)
    arch_config = io_tools.load_config_from_yaml(f'{ROOT}/configs/models/archs.yaml')
    
    # 2. 获取模型具体的配置文件名 (例如 'CMamba_v2' -> 'CryptoMamba/v2.yaml')
    model_arch_name = train_config.get('model')
    model_config_file = arch_config.get(model_arch_name)
    
    if not model_config_file:
        raise ValueError(f"Model architecture '{model_arch_name}' not found in archs.yaml")
        
    # 3. 读取具体的模型参数配置 (configs/models/CryptoMamba/v2.yaml)
    model_config_path = f'{ROOT}/configs/models/{model_config_file}'
    model_conf = io_tools.load_config_from_yaml(model_config_path)
    model_params = model_conf.get('params')

    # 4. 处理 Hyperparams 覆盖 (与 training.py 逻辑保持一致)
    # 如果 training config 中有 hyperparams 覆盖了 model params
    train_hyperparams = train_config.get('hyperparams')
    if train_hyperparams is not None:
        for key in train_hyperparams.keys():
            if key in model_params:
                model_params[key] = train_hyperparams.get(key)

    # 5. [关键修复] 注入 feature_names 和 skip_revin
    # 这确保模型初始化时知道哪些特征需要跳过归一化，从而正确构建 RevIN 层
    model_params['feature_names'] = feature_names
    model_params['skip_revin'] = train_config.get('skip_RevIN', [])

    print(f"Initializing model with params: {model_params}")
    
    # 6. 加载 Checkpoint，传入 strict=False 以防部分非权重参数不匹配，但传入 **model_params 以构建正确结构
    model = CryptoMambaModule.load_from_checkpoint(checkpoint_path, strict=True, **model_params)
    return model

def main():
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. 加载训练配置
    config = io_tools.load_config_from_yaml(f'{ROOT}/configs/training/{args.config}.yaml')
    data_config = io_tools.load_config_from_yaml(f"{ROOT}/configs/data_configs/{config.get('data_config')}.yaml")
    
    use_volume = config.get('use_volume', False)
    print(f"Loaded training config '{args.config}'. Additional features count: {len(config.get('additional_features', []))}")

    # 2. 准备特征名称列表 (必须在加载模型前完成)
    feature_names = get_feature_names(config)
    print(f"Target Feature List ({len(feature_names)}): {feature_names}")

    # 3. 准备数据
    val_transform = DataTransform(is_train=False, use_volume=use_volume, additional_features=config.get('additional_features', []))
    
    data_module = CMambaDataModule(
        data_config,
        train_transform=None,
        val_transform=val_transform,
        test_transform=None,
        batch_size=args.batch_size,
        distributed_sampler=False,
        num_workers=4,
        normalize=config.get('normalize', False), 
        window_size=14 
    )
    
    print("Loading validation data...")
    val_loader = data_module.val_dataloader()
    
    all_features = []
    all_y = []
    all_y_old = []
    
    # 这里的 y_key 需要从配置或默认值获取，BaseModule 默认为 Close
    y_key = 'Close' 

    for batch in val_loader:
        all_features.append(batch['features'])
        all_y.append(batch[y_key])
        all_y_old.append(batch[f'{y_key}_old'])

    X_val = torch.cat(all_features).to(device)
    y_val = torch.cat(all_y).to(device)
    y_old_val = torch.cat(all_y_old).to(device)
    
    print(f"Validation set shape: {X_val.shape}")

    # 3. 加载模型 (使用修复后的函数)
    print(f"Loading model structure and weights from {args.checkpoint_path}...")
    model = load_model_with_config(args.checkpoint_path, config, feature_names)
    
    model.to(device)
    model.eval()
    
    # 设置归一化系数 (用于 denormalize)
    model.set_normalization_coeffs(data_module.factors)

    # 4. 计算基准 RMSE
    print("Calculating baseline performance...")
    with torch.no_grad():
        y_hat_base = model(X_val, y_old_val).reshape(-1)
        _, y_hat_base_denorm = model.denormalize(y_val, y_hat_base)
        y_val_denorm, _ = model.denormalize(y_val, y_val)
        baseline_rmse = calculate_rmse(y_val_denorm, y_hat_base_denorm)
    
    print(f"Baseline Validation RMSE: {baseline_rmse:.6f}")

    # 6. PFI 分析
    print("X_val.shape[1]: ", X_val.shape[1])

    if len(feature_names) != X_val.shape[1]:
        print(f"WARNING: Feature names count ({len(feature_names)}) != Tensor channels ({X_val.shape[1]}).")
        feature_names = [f"Feat_{i}" for i in range(X_val.shape[1])]
    else:
        print("Feature names aligned successfully.")

    importances = {}
    print(f"Starting PFI for {len(feature_names)} features with {args.n_repeats} repeats...")
    
    # 遍历每一个特征 (维度 1)
    for i, feat_name in tqdm(enumerate(feature_names), total=len(feature_names)):
        permute_scores = []
        
        # 锁定第 i 个特征通道 (Batch, i, Length)
        # clone() 很重要，防止修改原始数据
        original_feat_col = X_val[:, i, :].clone()
        
        for _ in range(args.n_repeats):
            # 生成随机索引，打乱 Batch 维度
            perm_idx = torch.randperm(X_val.size(0)).to(device)
            
            # 实施置换：将该特征的所有时间步(整个窗口)在样本间打乱
            # 这样破坏了该特征与 Target 的关联，但保留了该特征内部的时间结构
            X_val[:, i, :] = original_feat_col[perm_idx]
            
            with torch.no_grad():
                y_hat_perm = model(X_val, y_old_val).reshape(-1)
                _, y_hat_perm_denorm = model.denormalize(y_val, y_hat_perm)
                # 这里的 y_val_denorm 是之前计算好的，不需要重算
                rmse_perm = calculate_rmse(y_val_denorm, y_hat_perm_denorm)
                permute_scores.append(rmse_perm)
            
            # 恢复数据，准备下一个 repeat 或下一个特征
            X_val[:, i, :] = original_feat_col 

        # 计算平均重要性
        avg_perm_rmse = np.mean(permute_scores)
        importance = avg_perm_rmse - baseline_rmse
        importances[feat_name] = importance

    # 6. 保存结果
    df_imp = pd.DataFrame(list(importances.items()), columns=['Feature', 'Importance_RMSE_Increase'])
    df_imp = df_imp.sort_values(by='Importance_RMSE_Increase', ascending=False).reset_index(drop=True)
    
    print("\n========== Important Features ==========")
    # print(df_imp.head(10))
    print(df_imp)
    
    df_imp.to_csv(args.output_file, index=False)
    print(f"\nFull results saved to {args.output_file}")

if __name__ == "__main__":
    main()