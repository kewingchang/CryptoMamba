# training.py
import os, sys, pathlib
sys.path.insert(0, os.path.dirname(pathlib.Path(__file__).parent.absolute()))

import yaml
from utils import io_tools
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint # 显式导入
from argparse import ArgumentParser
from pl_modules.data_module import CMambaDataModule
from data_utils.data_transforms import DataTransform
from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning.loggers import TensorBoardLogger
import warnings
import torch
import glob

warnings.simplefilter(action='ignore', category=FutureWarning)


ROOT = io_tools.get_root(__file__, num_returns=2)

# [移除] SnapshotCallback 类已不再需要

def get_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--logdir",
        type=str,
        help="Logging directory.",
    )
    parser.add_argument(
        "--accelerator",
        type=str,
        default='gpu',
        help="The type of accelerator.",
    )
    parser.add_argument(
        "--devices",
        type=int,
        default=1,
        help="Number of computing devices.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=23,
        help="Logging directory.",
    )
    parser.add_argument(
        "--expname",
        type=str,
        default='Cmamba',
        help="Experiment name. Reconstructions will be saved under this folder.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default='cmamba_nv',
        help="Path to config file.",
    )
    parser.add_argument(
        "--logger_type",
        default='tb',
        type=str,
        help="Path to config file.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of parallel workers.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="batch_size",
    )
    parser.add_argument(
        '--save_checkpoints', 
        default=False,   
        action='store_true',          
    )
    parser.add_argument(
        '--use_volume', 
        default=False,   
        action='store_true',          
    )

    parser.add_argument(
        '--resume_from_checkpoint',
        default=None,
    )

    parser.add_argument(
        '--max_epochs',
        type=int,
        default=200,
    )

    # [新增] Ensemble 数量控制
    parser.add_argument(
        '--ensemble_k', 
        type=int, 
        default=5, 
        help="Number of top checkpoints to keep for ensemble"
    )

    args = parser.parse_args()
    return args

# ... (save_all_hparams, load_model, calculate_all_metrics 函数保持不变) ...
def save_all_hparams(log_dir, args):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    save_dict = vars(args)
    path = log_dir + '/hparams.yaml'
    if os.path.exists(path):
        return
    with open(path, 'w') as f:
        yaml.dump(save_dict, f)

def load_model(config, logger_type, max_epochs, feature_names=None, skip_revin=None):
    # 修改：添加 max_epochs 参数
    # [修改] load_model 增加 feature_names 和 skip_revin 参数
    arch_config = io_tools.load_config_from_yaml('configs/models/archs.yaml')
    model_arch = config.get('model')
    model_config_path = f'{ROOT}/configs/models/{arch_config.get(model_arch)}'
    model_config = io_tools.load_config_from_yaml(model_config_path)

    normalize = model_config.get('normalize', False)
    hyperparams = config.get('hyperparams')
    if hyperparams is not None:
        for key in hyperparams.keys():
            model_config.get('params')[key] = hyperparams.get(key)

    model_config.get('params')['logger_type'] = logger_type
    model_config.get('params')['max_epochs'] = max_epochs  # 新增：将 max_epochs 添加到 params

    # [新增] 将特征名称和跳过列表注入到模型参数中
    if feature_names is not None:
        model_config.get('params')['feature_names'] = feature_names
    if skip_revin is not None:
        model_config.get('params')['skip_revin'] = skip_revin

    model = io_tools.instantiate_from_config(model_config)
    model.cuda()
    model.train()
    return model, normalize

# 辅助函数：手动计算指标
def calculate_all_metrics(preds, targets, y_olds):
    """
    preds:   [N], 预测价格
    targets: [N], 真实价格
    y_olds:  [N], 昨日价格 (用于计算 Acc)
    """
    # 确保都在 CPU 上
    preds = preds.cpu()
    targets = targets.cpu()
    y_olds = y_olds.cpu()

    # 1. MSE / RMSE
    mse = torch.mean((preds - targets) ** 2)
    rmse = torch.sqrt(mse)

    # 2. MAE (L1)
    mae = torch.mean(torch.abs(preds - targets))

    # 3. MAPE
    # 注意处理分母为0的情况，虽然价格通常不为0
    epsilon = 1e-8
    mape = torch.mean(torch.abs((preds - targets) / (targets + epsilon)))

    # 4. Accuracy (方向预测准确率)
    diff_pred = preds - y_olds
    diff_true = targets - y_olds
    
    # 这里的逻辑和 base_module 保持一致
    true_dir = (diff_true > 0).float()
    pred_dir = (diff_pred > 0).float()
    acc = (true_dir == pred_dir).float().mean()

    return {
        "rmse": rmse.item(),
        "mse": mse.item(),
        "mae": mae.item(),
        "mape": mape.item(),
        "acc": acc.item()
    }

if __name__ == "__main__":

    args = get_args()
    pl.seed_everything(args.seed)

    logdir = args.logdir

    config = io_tools.load_config_from_yaml(f'{ROOT}/configs/training/{args.config}.yaml')
    data_config = io_tools.load_config_from_yaml(f"{ROOT}/configs/data_configs/{config.get('data_config')}.yaml")
    use_volume = args.use_volume
    if not use_volume:
        use_volume = config.get('use_volume')
    train_transform = DataTransform(is_train=True, use_volume=use_volume, additional_features=config.get('additional_features', []))
    val_transform = DataTransform(is_train=False, use_volume=use_volume, additional_features=config.get('additional_features', []))
    test_transform = DataTransform(is_train=False, use_volume=use_volume, additional_features=config.get('additional_features', []))

    # [新增] 提取特征名称 (排除 Timestamp_orig，因为它不进入 features Tensor)
    feature_names = [k for k in train_transform.keys if k != 'Timestamp_orig']
    skip_revin_list = config.get('skip_revin', []) # 从 yaml 读取

    # 修改：传入 args.max_epochs
    # model, normalize = load_model(config, args.logger_type, args.max_epochs)
    # [修改] 传入 feature_names 和 skip_revin_list
    model, normalize = load_model(config, args.logger_type, args.max_epochs, feature_names, skip_revin_list)

    tmp = vars(args)
    tmp.update(config)

    name = config.get('name', args.expname)
    if args.logger_type == 'tb':
        logger = TensorBoardLogger("logs", name=name)
        logger.log_hyperparams(args)
    elif args.logger_type == 'wandb':
        logger = pl.loggers.WandbLogger(project=args.expname, config=tmp)
    else:
        raise ValueError('Unknown logger type.')

    data_module = CMambaDataModule(data_config,
                                   train_transform=train_transform,
                                   val_transform=val_transform,
                                   test_transform=test_transform,
                                   batch_size=args.batch_size,
                                   distributed_sampler=True,
                                   num_workers=args.num_workers,
                                   normalize=normalize,
                                   window_size=model.window_size,
                                   )
    
    callbacks = []
    
    # 1. [关键修改] ModelCheckpoint: 保存 Top K 个最好的模型
    if args.save_checkpoints:
        checkpoint_callback = ModelCheckpoint(
            save_top_k=args.ensemble_k,  # 使用参数控制，例如 5
            verbose=True,
            monitor="val/rmse",
            mode="min",
            filename='epoch{epoch}-val-rmse{val/rmse:.4f}',
            auto_insert_metric_name=False,
            save_last=True # 另外保留最后一个，防止TopK都没覆盖到最后
        )
        callbacks.append(checkpoint_callback)
    
    # 2. EarlyStopping
    early_stop_callback = EarlyStopping(
        monitor="val/rmse",
        min_delta=0.001,
        patience=100,
        verbose=True,
        mode="min"
    )
    callbacks.append(early_stop_callback)

    # [移除] SnapshotCallback

    max_epochs = config.get('max_epochs', args.max_epochs)
    model.set_normalization_coeffs(data_module.factors)

    trainer = pl.Trainer(accelerator=args.accelerator, 
                         devices=args.devices,
                         max_epochs=max_epochs,
                         enable_checkpointing=args.save_checkpoints,
                         log_every_n_steps=10,
                         logger=logger,
                         callbacks=callbacks,
                         strategy = DDPStrategy(find_unused_parameters=False),
                         )

    trainer.fit(model, datamodule=data_module, weights_only=False)

    if args.save_checkpoints:
        print("\n>>>>>>>>>>> Test Set Validate <<<<<<<<<<<<<<")
        trainer.test(model, datamodule=data_module, ckpt_path=checkpoint_callback.best_model_path, weights_only=False)
        # Validate on Val set
        print("\n>>>>>>>>>>> Val Set Validate <<<<<<<<<<<<<<")
        trainer.validate(model, dataloaders=data_module.val_dataloader(), ckpt_path=checkpoint_callback.best_model_path, weights_only=False)
        # "Validate" on Train set (模拟 Train 评测)
        print("\n>>>>>>>>>>> Train Set Validate <<<<<<<<<<<<<<")
        trainer.validate(model, dataloaders=data_module.train_dataloader(), ckpt_path=checkpoint_callback.best_model_path, weights_only=False)

    # =========================================================
    # Top-K Ensemble 评估部分
    # =========================================================
    
    # 获取最好的 K 个模型路径
    # checkpoint_callback.best_k_models 是一个字典 {path: score}
    snapshot_paths = list(checkpoint_callback.best_k_models.keys())

    # 1. 确定 Checkpoint 目录
    ckpt_dir = checkpoint_callback.dirpath
    if ckpt_dir is None and trainer.logger:
        # 如果 callback 还没来得及设置 dirpath，尝试从 logger 推断
        ckpt_dir = os.path.join(trainer.logger.log_dir, "checkpoints")
    
    snapshot_paths = []
    if ckpt_dir and os.path.exists(ckpt_dir):
        # 2. 扫描所有 .ckpt 文件
        all_ckpts = glob.glob(os.path.join(ckpt_dir, "*.ckpt"))
        # 3. 过滤掉 'last.ckpt' (它通常不是指标最好的，只是最新的)
        snapshot_paths = [p for p in all_ckpts if "last.ckpt" not in os.path.basename(p)]        
        # 4. (可选) 按文件名排序，方便观察
        snapshot_paths.sort()

    if len(snapshot_paths) == 0:
        print("\n[Warning] No checkpoints found for ensemble.")
    else:
        print(f"\n>>>>>>>>>>> Starting Top-{len(snapshot_paths)} Ensemble Evaluation <<<<<<<<<<<<<<")
        print(f"Models used: {snapshot_paths}")
        
        # 定义要评估的数据集
        eval_stages = []
        if args.save_checkpoints: 
             eval_stages.append(("Validation Set", data_module.val_dataloader()))
        eval_stages.append(("Test Set", data_module.test_dataloader()))

        for stage_name, dataloader in eval_stages:
            print(f"\n=== Evaluating on {stage_name} ===")
            
            # --- A. 收集 Ground Truth ---
            all_targets = []
            all_y_olds = []
            
            for batch in dataloader:
                y = batch[model.y_key]           
                y_old = batch[f'{model.y_key}_old'] 
                all_targets.append(y.cpu())
                all_y_olds.append(y_old.cpu())
            
            all_targets = torch.cat(all_targets, dim=0)
            all_y_olds = torch.cat(all_y_olds, dim=0)

            # --- B. 对每个 Top-K 模型进行预测 ---
            ensemble_preds_denorm = [] 

            for i, path in enumerate(snapshot_paths):
                print(f"[{i+1}/{len(snapshot_paths)}] Predicting: {os.path.basename(path)}")
                
                checkpoint = torch.load(path, weights_only=False)
                model.load_state_dict(checkpoint['state_dict'])
                model.eval()
                model.cuda()

                cycle_preds = []
                with torch.no_grad():
                    for batch in dataloader:
                        x = batch['features'].cuda()
                        y_old_batch = batch[f'{model.y_key}_old'].cuda()
                        
                        y_hat_raw = model(x, y_old_batch).reshape(-1)
                        _, y_hat_denorm = model.denormalize(None, y_hat_raw)
                        cycle_preds.append(y_hat_denorm.cpu())

                model_full_pred = torch.cat(cycle_preds, dim=0)
                ensemble_preds_denorm.append(model_full_pred)
                
                # [可选] 打印单个模型的性能，方便排查谁是“烂苹果”
                single_metrics = calculate_all_metrics(model_full_pred, all_targets, all_y_olds)
                print(f"    -> RMSE: {single_metrics['rmse']:.4f}, -> MAE: {single_metrics['mae']:.4f}, -> MAPE: {single_metrics['mape']:.4f}, ACC: {single_metrics['acc']:.4f}")

            # --- C. 计算平均 (Ensemble) ---
            stacked_preds = torch.stack(ensemble_preds_denorm)
            avg_preds = torch.mean(stacked_preds, dim=0)

            # --- D. 计算 Ensemble 指标 ---
            metrics = calculate_all_metrics(avg_preds, all_targets, all_y_olds)
            
            print(f"--- Top-{len(snapshot_paths)} Ensemble Results ({stage_name}) ---")
            print(f"RMSE : {metrics['rmse']:.6f}")
            print(f"MAE  : {metrics['mae']:.6f}")
            print(f"MAPE  : {metrics['mape']:.6f}")
            print(f"ACC  : {metrics['acc']:.6f}")
            
            # 与最好的单个模型对比（best_model_path 通常是 Top-1）
            if checkpoint_callback.best_model_path:
                print(f"Best Single Model path: {checkpoint_callback.best_model_path}")