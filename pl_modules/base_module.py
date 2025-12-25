# base_module.py
import copy
import torch
import torch.nn as nn
import pytorch_lightning as pl
from models.cmamba import CMamba
from torchmetrics.regression import MeanAbsolutePercentageError as MAPE
from torch.optim.lr_scheduler import CosineAnnealingLR  # 已存在
from torch.nn import SmoothL1Loss  # 新增
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts  # 修改：导入 CosineAnnealingWarmRestarts 


# [新增] 分位数损失函数
class QuantileLoss(nn.Module):
    def __init__(self, quantiles):
        super().__init__()
        self.quantiles = quantiles # list e.g., [0.05, 0.30]

    def forward(self, preds, target):
        # preds: (B, num_quantiles)
        # target: (B) or (B, 1)
        loss = 0
        target = target.view(-1, 1) # Ensure (B, 1)
        for i, q in enumerate(self.quantiles):
            error = target - preds[:, i:i+1]
            loss += torch.max(q * error, (q - 1) * error).mean()
        return loss

class WarmupCosineScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_epochs, max_epochs, eta_min=1e-6, warmup_start_ratio=0.1):  # 新增 warmup_start_ratio，避免从 0 开始
        self.warmup_epochs = warmup_epochs
        self.cosine_scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=eta_min)  # 修改：用 CosineAnnealingWarmRestarts 替换 CosineAnnealingLR
        self.warmup_start_ratio = warmup_start_ratio  # 新增：起始 lr 比例 (e.g., 0.1 * base_lr)
        super().__init__(optimizer)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            return [base_lr * self.warmup_start_ratio + (base_lr - base_lr * self.warmup_start_ratio) * (self.last_epoch + 1) / self.warmup_epochs for base_lr in self.base_lrs]  # 修改：从 base_lr * ratio 开始线性升
        else:
            return self.cosine_scheduler.get_lr()

    def step(self, epoch=None):
        super().step(epoch)
        if self.last_epoch >= self.warmup_epochs:
            self.cosine_scheduler.step()  # 新增：调用 cosine_scheduler.step() 更新内部状态


class BaseModule(pl.LightningModule):

    def __init__(
        self,
        lr=0.0002, 
        lr_step_size=50,
        lr_gamma=0.1,
        weight_decay=0.0, 
        logger_type=None,
        window_size=14,
        y_key='Close',
        optimizer='adam',
        mode='default',
        loss_type='rmse',
        max_epochs=1000,
        alpha=0.7,  # 新增：混合损失的权重，alpha for RMSE, (1-alpha) for Focal loss
        feature_names=None,  # [新增] 接收特征名称列表
        quantiles=None, # [新增] 接收分位数列表，例如 [0.05, 0.30]
        **kwargs
    ):
        super().__init__()

        self.lr = lr
        self.lr_step_size = lr_step_size
        self.lr_gamma = lr_gamma
        self.weight_decay = weight_decay
        self.logger_type = logger_type 
        self.y_key = y_key
        self.optimizer = optimizer
        self.batch_size = None   
        self.mode = mode
        self.window_size = window_size
        self.loss_type = loss_type
        self.max_epochs = max_epochs
        self.alpha = alpha  # 新增权重
        self.quantiles = quantiles

        # only quantile supported
        assert(self.loss_type == 'quantile')

        # 初始化 Loss
        if self.loss_type == 'quantile':
            assert self.quantiles is not None, "Quantiles must be provided for quantile loss"
            self.criterion = QuantileLoss(self.quantiles)
        else:
            self.criterion = nn.MSELoss() # fallback default

        self.smooth_l1 = SmoothL1Loss(beta=0.5)  # 新增

        # Index for 'Close' in keys
        self.target_channel = 3
        self.feature_names = feature_names

        if self.feature_names is not None and self.y_key in self.feature_names:
            self.target_channel = self.feature_names.index(self.y_key)
            print(f"[BaseModule] Target '{self.y_key}' found at channel index: {self.target_channel}")
        else:
            print(f"[BaseModule] Warning: '{self.y_key}' not found in features, defaulting target_channel to 3.")

        self.mse = nn.MSELoss()
        self.l1 = nn.L1Loss()
        self.mape = MAPE()

        self.normalization_coeffs = None

    def forward(self, x, y_old=None):
        # [修改] 移除 reshape(-1)，因为 cmamba 内部已经处理了 output_dim=1 的情况
        # 且如果是 quantile，我们需要保留 (B, 2) 形状
        out = self.model(x)
        if self.mode == 'diff':
             # 注意：如果是多头输出，diff 模式可能需要广播 y_old
             # 暂时假设 quantile 模型不使用 'diff' 模式
             if out.dim() > 1:
                 return out + y_old.unsqueeze(1)
             return out + y_old
        return out
        
    def set_normalization_coeffs(self, factors):
        if factors is None:
            return
        scale = factors.get(self.y_key).get('max') - factors.get(self.y_key).get('min')
        shift = factors.get(self.y_key).get('min')
        self.normalization_coeffs = (scale, shift)

    def denormalize(self, y, y_hat):
            # 1. 如果使用了 Dataset 级别的归一化 (min-max 等)
            if self.normalization_coeffs is not None:
                scale, shift = self.normalization_coeffs
                if y is not None:
                    y = y * scale + shift
                y_hat = y_hat * scale + shift
            
            # 2. 如果使用了 RevIN (模型级别的归一化)
            elif hasattr(self.model, 'revin') and self.model.revin is not None:
                re = self.model.revin
                
                # 我们需要查找当前的 target (log_return) 在 RevIN 内部对应的索引
                # 如果 target 被 skip 了，就找不到索引，那么就不需要反归一化
                
                target_global_idx = self.target_channel # 全局索引 (例如 4)
                effective_idx = None # RevIN 内部的有效索引
                
                # 检查模型是否有 norm_indices (记录了哪些特征被归一化了)
                if hasattr(self.model, 'norm_indices') and self.model.norm_indices is not None:
                    # 在 norm_indices 中查找 target_global_idx
                    # (self.model.norm_indices == target_global_idx) 会返回一个布尔掩码
                    matches = (self.model.norm_indices == target_global_idx).nonzero(as_tuple=True)[0]
                    
                    if len(matches) > 0:
                        # 找到了！说明 target 被归一化了
                        effective_idx = matches.item()
                else:
                    # 兼容旧模型或无 selective RevIN 的情况，默认索引一致
                    effective_idx = target_global_idx

                # 只有当找到了有效索引，且该索引在 RevIN 参数范围内时，才执行反归一化
                if effective_idx is not None and effective_idx < re.num_features:
                    # 获取统计量
                    stdev = re.stdev[:, 0, effective_idx]
                    mean = re.mean[:, 0, effective_idx] if not re.subtract_last else re.last[:, 0, effective_idx]
                    
                    # [修改] 支持多头输出的反归一化
                    # y_hat 可能是 (B, 2)，stdev 是 (B,)
                    # 需要 unsqueeze stdev 变成 (B, 1) 以进行广播
                    if y_hat.dim() > 1:
                         stdev = stdev.unsqueeze(1)
                         mean = mean.unsqueeze(1)

                    if re.affine:
                        # Affine 比较麻烦，通常 affine_weight 是 (C,)。
                        # 如果 output 是多头但对应同一个 feature (Low)，我们应用相同的 affine 参数
                        y_hat = y_hat - re.affine_bias[effective_idx]
                        y_hat = y_hat / (re.affine_weight[effective_idx] + re.eps)
                    
                    y_hat = y_hat * stdev
                    y_hat = y_hat + mean
    
            return y, y_hat

    def training_step(self, batch, batch_idx):
        x = batch['features']
        y = batch[self.y_key]
        y_old = batch[f'{self.y_key}_old']
        if self.batch_size is None:
            self.batch_size = x.shape[0]
            
        y_hat = self.forward(x, y_old) # (B, output_dim)
        
        # [修改] 损失计算前，保持 normalized 状态计算 Quantile Loss 会更稳定
        # 但为了指标统一，这里演示 denormalize 后计算 (你可以根据效果调整)
        y_denorm, y_hat_denorm = self.denormalize(y, y_hat)
        
        if self.loss_type == 'quantile':
            # Quantile Loss
            loss = self.criterion(y_hat_denorm, y_denorm)
            self.log("train/loss", loss, batch_size=self.batch_size, sync_dist=True, prog_bar=True)
            # 核心评价指标
            target = y_denorm.view(-1, 1)
            for i, q in enumerate(self.quantiles):
                # 1. 计算覆盖率 (Coverage Rate)
                # 统计有多少比例的真实值小于预测值
                # 对于 q=0.05，我们期望这个比例接近 0.05
                coverage = (target < y_hat_denorm[:, i:i+1]).float().mean()
                
                # 2. 计算 MAE (作为参考)
                mae = self.l1(y_hat_denorm[:, i], y_denorm)
                
                # Log 出来，格式例如: val/q0.05_cover, val/q0.05_mae
                self.log(f"train/q{q}_cover", coverage, batch_size=self.batch_size, sync_dist=True, prog_bar=False)
                self.log(f"train/q{q}_mae", mae, batch_size=self.batch_size, sync_dist=True, prog_bar=False)
        else:
            raise ValueError(f"Unsupported loss_type: {self.loss_type!r}. Only 'quantile' supported.")

        return loss

    def validation_step(self, batch, batch_idx):
        x = batch['features']
        y = batch[self.y_key]
        y_old = batch[f'{self.y_key}_old']
        y_hat = self.forward(x, y_old)
        y_denorm, y_hat_denorm = self.denormalize(y, y_hat)

        if self.loss_type == 'quantile':
            loss = self.criterion(y_hat_denorm, y_denorm)
            self.log("val/loss", loss, batch_size=self.batch_size, sync_dist=True, prog_bar=True)
            
            # 核心评价指标
            # y_denorm: (B,)  y_hat_denorm: (B, num_quantiles)
            # 确保 y_denorm 是 (B, 1) 以便广播对比
            target = y_denorm.view(-1, 1)
            
            for i, q in enumerate(self.quantiles):
                # 1. 计算覆盖率 (Coverage Rate)
                # 统计有多少比例的真实值小于预测值
                # 对于 q=0.05，我们期望这个比例接近 0.05
                coverage = (target < y_hat_denorm[:, i:i+1]).float().mean()
                
                # 2. 计算 MAE (作为参考)
                mae = self.l1(y_hat_denorm[:, i], y_denorm)
                
                # Log 出来，格式例如: val/q0.05_cover, val/q0.05_mae
                self.log(f"val/q{q}_cover", coverage, batch_size=self.batch_size, sync_dist=True, prog_bar=False)
                self.log(f"val/q{q}_mae", mae, batch_size=self.batch_size, sync_dist=True, prog_bar=False)

            return {"val_loss": loss}
        else:
            raise ValueError(f"Unsupported loss_type: {self.loss_type!r}. Only 'quantile' supported.")

    def test_step(self, batch, batch_idx):
        x = batch['features']
        y = batch[self.y_key]
        y_old = batch[f'{self.y_key}_old']
        y_hat = self.forward(x, y_old)
        y_denorm, y_hat_denorm = self.denormalize(y, y_hat)

        if self.loss_type == 'quantile':
            loss = self.criterion(y_hat_denorm, y_denorm)
            self.log("test/loss", loss, batch_size=self.batch_size, sync_dist=True, prog_bar=True)
            
            # === [新增] 同上 ===
            target = y_denorm.view(-1, 1)
            for i, q in enumerate(self.quantiles):
                coverage = (target < y_hat_denorm[:, i:i+1]).float().mean()
                mae = self.l1(y_hat_denorm[:, i], y_denorm)
                
                self.log(f"test/q{q}_cover", coverage, batch_size=self.batch_size, sync_dist=True, prog_bar=True)
                self.log(f"test/q{q}_mae", mae, batch_size=self.batch_size, sync_dist=True, prog_bar=True)
            
            return {"test_loss": loss}
        else:
            raise ValueError(f"Unsupported loss_type: {self.loss_type!r}. Only 'quantile' supported.")
    
    def configure_optimizers(self):
        # 1. 准备参数分组：分离出不需要 Weight Decay 的参数
        # Mamba/Transformer 的最佳实践：Bias, LayerNorm, 以及特定的 SSM 参数不应衰减
        decay_params = []
        no_decay_params = []
        
        # 遍历模型所有参数
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
                
            # 检查 cmamba.py 中定义的特殊属性 _no_weight_decay
            if hasattr(param, "_no_weight_decay") and param._no_weight_decay:
                no_decay_params.append(param)
            # 常见的不衰减类型：bias (偏置) 和 norm (归一化层权重)
            elif "bias" in name or "norm" in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        # 构建参数组
        optim_groups = [
            {"params": decay_params, "weight_decay": self.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]

        # 2. 初始化优化器
        if self.optimizer == 'adamw':
            # AdamW 使用我们构建的分组
            optim = torch.optim.AdamW(
                optim_groups, 
                lr=self.lr, 
                betas=(0.9, 0.95) # Mamba 论文常用的 beta2 通常较低，也可以保持默认 (0.9, 0.999)
            )
        elif self.optimizer == 'adam':
            # 旧的 Adam 逻辑 (也可以用分组，但 Adam 的 WD 实现本身就有问题，这里保持原样)
            optim = torch.optim.Adam(
                self.parameters(), lr=self.lr, weight_decay=self.weight_decay
            )
        elif self.optimizer == 'sgd':
            optim = torch.optim.SGD(
                self.parameters(), lr=self.lr, weight_decay=self.weight_decay
            )
        else:
            raise ValueError(f'Unimplemented optimizer {self.optimizer}')

        # 3. Scheduler (保持你原有的逻辑不变)
        scheduler = WarmupCosineScheduler(optim, warmup_epochs=10, max_epochs=self.max_epochs)
        return [optim], [scheduler]

    def lr_scheduler_step(self, scheduler, *args, **kwargs):
        scheduler.step()
