# base_module.py
import copy
import torch
import torch.nn as nn
import pytorch_lightning as pl
from models.cmamba import CMamba
from torchmetrics.regression import MeanAbsolutePercentageError as MAPE
from torch.optim.lr_scheduler import CosineAnnealingLR  # 已存在
from torch.nn import SmoothL1Loss  # 新增

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
        alpha=0.7,  # 新增：混合损失的权重，alpha for RMSE, (1-alpha) for BCE
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

        self.target_channel = 4  # Index for 'Close' in keys

        self.mse = nn.MSELoss()
        self.l1 = nn.L1Loss()
        self.mape = MAPE()
        self.bce = nn.BCEWithLogitsLoss()  # 新增 BCE 损失（用 WithLogits 以处理原始 logit 输出）
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
        y_old = batch[f'{self.y_key}_old']
        if self.batch_size is None:
            self.batch_size = x.shape[0]
        y_hat = self.forward(x, y_old).reshape(-1)
        y, y_hat = self.denormalize(y, y_hat)
        mse = self.mse(y_hat, y)
        rmse = torch.sqrt(mse)
        mape = self.mape(y_hat, y)
        l1 = self.l1(y_hat, y)

        # 新增：计算方向损失 (BCE logit)
        pred_logit = y_hat - y_old # 预测涨跌
        true_direction = (y > y_old).float()      # 真实涨跌
        bce_loss = self.bce(pred_logit, true_direction) # BCE 计算
        smooth_l1_loss = self.smooth_l1(y_hat, y)  # 新增smooth_l1_loss

        # 混合损失：根据 loss_type
        if self.loss_type == 'hybrid':
            loss = self.alpha * smooth_l1_loss + (1 - self.alpha) * bce_loss
            # loss = self.alpha * rmse + (1 - self.alpha) * bce_loss  # 加权和（用 rmse 以匹配原损失）
        elif self.loss_type == 'rmse':
            loss = rmse  # 原损失
        # ... 其他 elif 如 'mse' 等保持不变

        # 日志记录（新增 BCE log）
        self.log("train/mse", mse.detach(), batch_size=self.batch_size, sync_dist=True, prog_bar=False)
        self.log("train/rmse", rmse.detach(), batch_size=self.batch_size, sync_dist=True, prog_bar=True)
        self.log("train/mape", mape.detach(), batch_size=self.batch_size, sync_dist=True, prog_bar=True)
        self.log("train/mae", l1.detach(), batch_size=self.batch_size, sync_dist=True, prog_bar=False)
        self.log("train/bce", bce_loss.detach(), batch_size=self.batch_size, sync_dist=True, prog_bar=True)  # 新增 BCE log
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
        rmse = torch.sqrt(mse)
        mape = self.mape(y_hat, y)
        l1 = self.l1(y_hat, y)

        # BCE logit 计算
        pred_logit = y_hat - y_old
        true_direction = (y > y_old).float()      # 真实涨跌
        bce_loss = self.bce(pred_logit, true_direction)
        smooth_l1_loss = self.smooth_l1(y_hat, y)  # 新增smooth_l1_loss

        self.log("val/mse", mse.detach(), sync_dist=True, batch_size=self.batch_size, prog_bar=False)
        self.log("val/rmse", rmse.detach(), batch_size=self.batch_size, sync_dist=True, prog_bar=True)
        self.log("val/mape", mape.detach(), batch_size=self.batch_size, sync_dist=True, prog_bar=True)
        self.log("val/mae", l1.detach(), batch_size=self.batch_size, sync_dist=True, prog_bar=False)
        self.log("val/bce", bce_loss.detach(), sync_dist=True, batch_size=self.batch_size, prog_bar=True) 
        self.log("val/smooth_l1", smooth_l1_loss.detach(), batch_size=self.batch_size, sync_dist=True, prog_bar=True)  # 新增 smooth_l1_loss log
        return {
            "val_loss": mse,
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
        rmse = torch.sqrt(mse)
        mape = self.mape(y_hat, y)
        l1 = self.l1(y_hat, y)

        # BCE logit计算
        pred_logit = y_hat - y_old
        true_direction = (y > y_old).float()      # 真实涨跌
        bce_loss = self.bce(pred_logit, true_direction)
        smooth_l1_loss = self.smooth_l1(y_hat, y)  # 新增smooth_l1_loss

        self.log("test/mse", mse.detach(), sync_dist=True, batch_size=self.batch_size, prog_bar=False)
        self.log("test/rmse", rmse.detach(), batch_size=self.batch_size, sync_dist=True, prog_bar=True)
        self.log("test/mape", mape.detach(), batch_size=self.batch_size, sync_dist=True, prog_bar=True)
        self.log("test/mae", l1.detach(), batch_size=self.batch_size, sync_dist=True, prog_bar=False)
        self.log("test/bce", bce_loss.detach(), sync_dist=True, batch_size=self.batch_size, prog_bar=True) 
        self.log("test/smooth_l1", smooth_l1_loss.detach(), batch_size=self.batch_size, sync_dist=True, prog_bar=True)  # 新增 smooth_l1_loss log
        return {
            "test_loss": mse,
        }
    
    def configure_optimizers(self):
        if self.optimizer == 'adam':
            optim = torch.optim.Adam(
                self.parameters(), lr=self.lr, weight_decay=self.weight_decay
            )
        elif self.optimizer == 'sgd':
            optim = torch.optim.SGD(
                self.parameters(), lr=self.lr, weight_decay=self.weight_decay
            )
        else:
            raise ValueError(f'Unimplemented optimizer {self.optimizer}')
        scheduler = CosineAnnealingLR(optim, T_max=self.max_epochs, eta_min=1e-6)  # 新增 CosineAnnealingLR
        return [optim], [scheduler]

    def lr_scheduler_step(self, scheduler, *args, **kwargs):
        scheduler.step()
