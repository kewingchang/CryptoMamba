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
        self.loss_type = loss_type  # 改名为 loss_type，以区分混合
        self.max_epochs = max_epochs
        self.alpha = alpha  # 新增权重

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
        # 新增 BCE 损失（用 WithLogits 以处理原始 logit 输出）
        # self.focal = nn.BCEWithLogitsLoss()
        # 新增 Focal 损失
        # self.focal = FocalLoss(gamma=2.0, alpha=0.5)
        self.normalization_coeffs = None

    def forward(self, x, y_old=None):
        if self.mode == 'default':
            return self.model(x).reshape(-1)
        elif self.mode == 'diff':
            return self.model(x).reshape(-1) + y_old
        
    def set_normalization_coeffs(self, factors):
        if factors is None:
            return
        scale = factors.get(self.y_key).get('max') - factors.get(self.y_key).get('min')
        shift = factors.get(self.y_key).get('min')
        self.normalization_coeffs = (scale, shift)

    def denormalize(self, y, y_hat):
        if self.normalization_coeffs is not None:
            scale, shift = self.normalization_coeffs
            if y is not None:
                y = y * scale + shift
            y_hat = y_hat * scale + shift
        elif hasattr(self.model, 'revin') and self.model.revin is not None:
            re = self.model.revin
            # print("Before denorm: y_hat=", y_hat.mean(), "stdev=", re.stdev.mean())  # 检查stats是否从norm继承
            target_idx = self.target_channel
            if re.affine:
                y_hat = y_hat - re.affine_bias[target_idx]
                y_hat = y_hat / (re.affine_weight[target_idx] + re.eps)
            y_hat = y_hat * re.stdev[:, 0, target_idx]
            if re.subtract_last:
                y_hat = y_hat + re.last[:, 0, target_idx]
            else:
                y_hat = y_hat + re.mean[:, 0, target_idx]
            # print("After denorm: y_hat=", y_hat.mean())  # 应恢复原始尺度
        return y, y_hat

    def training_step(self, batch, batch_idx):
        x = batch['features']
        y = batch[self.y_key]
        y_old = batch[f'{self.y_key}_old'] # 注意：如果预测 log_return，这里 y_old 是上一日的 log_return
        if self.batch_size is None:
            self.batch_size = x.shape[0]
        y_hat = self.forward(x, y_old).reshape(-1)
        y, y_hat = self.denormalize(y, y_hat)
        mse = self.mse(y_hat, y)
        rmse = torch.sqrt(mse)
        mape = self.mape(y_hat, y)
        l1 = self.l1(y_hat, y)

        # SmoothL1
        smooth_l1_loss = self.smooth_l1(y_hat, y)

        # 混合损失：根据 loss_type
        if self.loss_type == 'hybrid':
            # 1126: 暂时只使用SmoothL1
            # loss = self.alpha * smooth_l1_loss + (1 - self.alpha) * focal_loss
            # loss = smooth_l1_loss + 0.3 * direction_loss 
            loss = smooth_l1_loss
        elif self.loss_type == 'rmse':
            loss = rmse
        elif self.loss_type == 'mse':
            loss = mse
        # ... 其他 elif 如 'mse' 等保持不变

        # [修改点 3]：准确率计算逻辑调整
        # 如果是预测 Log Return，y 本身的正负就代表涨跌，不需要减去 y_old
        # y > 0 表示涨，y < 0 表示跌
        if self.y_key == 'log_return':
            true_dir = (y > 0).float()
            pred_dir = (y_hat > 0).float()
        else:
            # 旧逻辑：预测价格时，需要看 (Current - Old)
            diff_pred = y_hat - y_old
            diff_true = y - y_old
            true_dir = (diff_true > 0).float()
            pred_dir = (diff_pred > 0).float()   
        acc = (true_dir == pred_dir).float().mean()

        # ==========================================
        # 3. [新增] 价格还原与指标计算 (仅用于监控，不参与反向传播)
        # ==========================================
        if self.y_key == 'log_return':
            # 这里的 Close 和 Close_old 是真实价格（未归一化，或者你需要确认是否被DataTransform归一化了）
            # 根据你的 dataset.py 实现，batch 里应该有原始的 'Close' 和 'Close_old'
            # 只要 DataTransform 里 keys 包含 'Close' 即可
            
            # 注意：DataTransform 输出的 tensor 可能是 float32
            price_true = batch['Close']       # 今天的真实价格
            price_old = batch['Close_old']    # 昨天的真实价格
            
            # 还原预测价格: P_t = P_{t-1} * exp(r_t)
            price_pred = price_old * torch.exp(y_hat)
            
            # 计算“美元单位”的误差
            mse = self.mse(price_pred, price_true)
            rmse = torch.sqrt(mse)
            mape = self.mape(price_pred, price_true)
            l1 = self.l1(price_pred, price_true)
            smooth_l1_loss = self.smooth_l1(price_pred, price_true)

        # 日志记录
        self.log("train/mse", mse.detach(), batch_size=self.batch_size, sync_dist=True, prog_bar=False)
        self.log("train/rmse", rmse.detach(), batch_size=self.batch_size, sync_dist=True, prog_bar=True)
        # 注意：预测 Log Return 时，MAPE 可能非常大或不稳定（因为真实值接近0），建议看 MAE/RMSE
        self.log("train/mape", mape.detach(), batch_size=self.batch_size, sync_dist=True, prog_bar=True)
        self.log("train/mae", l1.detach(), batch_size=self.batch_size, sync_dist=True, prog_bar=False)
        self.log("train/acc", acc, batch_size=self.batch_size, prog_bar=True)
        self.log("train/smooth_l1", smooth_l1_loss.detach(), batch_size=self.batch_size, sync_dist=True, prog_bar=True)  # 新增 smooth_l1_loss log

        return loss  # 返回总损失

    def validation_step(self, batch, batch_idx):
        x = batch['features']
        y = batch[self.y_key]
        y_old = batch[f'{self.y_key}_old']
        if self.batch_size is None:
            self.batch_size = x.shape[0]
        y_hat = self.forward(x, y_old).reshape(-1)
        y, y_hat = self.denormalize(y, y_hat)
        mse = self.mse(y_hat, y)
        mse_ret = self.mse(y_hat, y)
        rmse_monitor = torch.sqrt(mse)
        rmse = torch.sqrt(mse)
        mape = self.mape(y_hat, y)
        l1 = self.l1(y_hat, y)
        smooth_l1_loss = self.smooth_l1(y_hat, y)

        # [修改点 3]：准确率计算逻辑调整 (同上)
        if self.y_key == 'log_return':
            true_dir = (y > 0).float()
            pred_dir = (y_hat > 0).float()
        else:
            diff_pred = y_hat - y_old
            diff_true = y - y_old
            true_dir = (diff_true > 0).float()
            pred_dir = (diff_pred > 0).float()  
        acc = (true_dir == pred_dir).float().mean()

        # ==========================================
        # 3. [新增] 价格还原与指标计算 (仅用于监控，不参与反向传播)
        # ==========================================
        if self.y_key == 'log_return':
            # 这里的 Close 和 Close_old 是真实价格（未归一化，或者你需要确认是否被DataTransform归一化了）
            # 根据你的 dataset.py 实现，batch 里应该有原始的 'Close' 和 'Close_old'
            # 只要 DataTransform 里 keys 包含 'Close' 即可
            
            # 注意：DataTransform 输出的 tensor 可能是 float32
            price_true = batch['Close']       # 今天的真实价格
            price_old = batch['Close_old']    # 昨天的真实价格
            
            # 还原预测价格: P_t = P_{t-1} * exp(r_t)
            price_pred = price_old * torch.exp(y_hat)
            
            # 计算“美元单位”的误差
            mse = self.mse(price_pred, price_true)
            rmse = torch.sqrt(mse)
            mape = self.mape(price_pred, price_true)
            l1 = self.l1(price_pred, price_true)
            smooth_l1_loss = self.smooth_l1(price_pred, price_true)

        self.log("val/mse", mse.detach(), sync_dist=True, batch_size=self.batch_size, prog_bar=False)
        self.log("val/rmse", rmse.detach(), batch_size=self.batch_size, sync_dist=True, prog_bar=True)
        self.log("val/rmse_monitor", rmse.detach(), batch_size=self.batch_size, sync_dist=True, prog_bar=True)
        self.log("val/mape", mape.detach(), batch_size=self.batch_size, sync_dist=True, prog_bar=True)
        self.log("val/mae", l1.detach(), batch_size=self.batch_size, sync_dist=True, prog_bar=False)
        self.log("val/acc", acc, batch_size=self.batch_size, prog_bar=True)
        self.log("val/smooth_l1", smooth_l1_loss.detach(), batch_size=self.batch_size, sync_dist=True, prog_bar=True)  # 新增 smooth_l1_loss log
        
        return {
            "val_loss": mse_ret,
        }
    
    def test_step(self, batch, batch_idx):
        x = batch['features']
        y = batch[self.y_key]
        y_old = batch[f'{self.y_key}_old']
        if self.batch_size is None:
            self.batch_size = x.shape[0]
        y_hat = self.forward(x, y_old).reshape(-1)
        y, y_hat = self.denormalize(y, y_hat)
        mse = self.mse(y_hat, y)
        mse_ret = self.mse(y_hat, y)
        rmse = torch.sqrt(mse)
        mape = self.mape(y_hat, y)
        l1 = self.l1(y_hat, y)

        smooth_l1_loss = self.smooth_l1(y_hat, y)  # 新增smooth_l1_loss
        if self.y_key == 'log_return':
            true_dir = (y > 0).float()
            pred_dir = (y_hat > 0).float()
        else:
            diff_pred = y_hat - y_old
            diff_true = y - y_old
            true_dir = (diff_true > 0).float()
            pred_dir = (diff_pred > 0).float()   
        acc = (true_dir == pred_dir).float().mean()

        # ==========================================
        # 3. [新增] 价格还原与指标计算 (仅用于监控，不参与反向传播)
        # ==========================================
        if self.y_key == 'log_return':
            # 这里的 Close 和 Close_old 是真实价格（未归一化，或者你需要确认是否被DataTransform归一化了）
            # 根据你的 dataset.py 实现，batch 里应该有原始的 'Close' 和 'Close_old'
            # 只要 DataTransform 里 keys 包含 'Close' 即可
            
            # 注意：DataTransform 输出的 tensor 可能是 float32
            price_true = batch['Close']       # 今天的真实价格
            price_old = batch['Close_old']    # 昨天的真实价格
            
            # 还原预测价格: P_t = P_{t-1} * exp(r_t)
            price_pred = price_old * torch.exp(y_hat)
            
            # 计算“美元单位”的误差
            mse = self.mse(price_pred, price_true)
            rmse = torch.sqrt(mse)
            mape = self.mape(price_pred, price_true)
            l1 = self.l1(price_pred, price_true)
            smooth_l1_loss = self.smooth_l1(price_pred, price_true)

        self.log("test/mse", mse.detach(), sync_dist=True, batch_size=self.batch_size, prog_bar=False)
        self.log("test/rmse", rmse.detach(), batch_size=self.batch_size, sync_dist=True, prog_bar=True)
        self.log("test/mape", mape.detach(), batch_size=self.batch_size, sync_dist=True, prog_bar=True)
        self.log("test/mae", l1.detach(), batch_size=self.batch_size, sync_dist=True, prog_bar=False)
        self.log("test/acc", acc, batch_size=self.batch_size, prog_bar=True)
        self.log("test/smooth_l1", smooth_l1_loss.detach(), batch_size=self.batch_size, sync_dist=True, prog_bar=True)  # 新增 smooth_l1_loss log
        return {
            "test_loss": mse_ret,
        }
    
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
