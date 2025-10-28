import copy
import torch
import torch.nn as nn
import pytorch_lightning as pl
from models.cmamba import CMamba
from torchmetrics.regression import MeanAbsolutePercentageError as MAPE
from einops import rearrange  # ADDED FOR REVIN: import rearrange

# ADDED FOR REVIN: RevIN class definition
class RevIN(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True):
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x, mode: str, target_idx=None):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
            return x
        elif mode == 'denorm':
            x = self._denormalize(x, target_idx)
            return x
        else:
            raise ValueError("Mode must be either 'norm' or 'denorm'")

    def _get_statistics(self, x):
        # x shape: (b, s, f)
        self.mean = torch.mean(x, dim=1, keepdim=True).detach()
        var = torch.var(x, dim=1, keepdim=True, unbiased=False)
        self.stdev = torch.sqrt(var + self.eps).detach()

    def _normalize(self, x):
        x = (x - self.mean) / self.stdev
        if self.affine:
            x = x * self.weight[None, None, :] + self.bias[None, None, :]
        return x

    def _denormalize(self, x, target_idx=None):
        if x.ndim == 1:
            x = x.unsqueeze(-1).unsqueeze(-1)  # (b,) -> (b,1,1)
        elif x.ndim == 2:
            x = x.unsqueeze(-1)  # (b,1) -> (b,1,1)
        if self.affine:
            x = (x - self.bias[None, None, :]) / (self.weight[None, None, :] + self.eps)
        if target_idx is None:
            x = x * self.stdev + self.mean
        else:
            mean = self.mean[:, :, target_idx:target_idx+1]
            stdev = self.stdev[:, :, target_idx:target_idx+1]
            x = x * stdev + mean
        return x.squeeze()
# END ADDED FOR REVIN

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
        loss='rmse',
        use_revin=False,  # ADDED FOR REVIN: add use_revin parameter
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
        self.loss = loss
        self.use_revin = use_revin  # ADDED FOR REVIN
        self.target_idx = None  # ADDED FOR REVIN: set later in training/eval/pred
        self.mse = nn.MSELoss()
        self.l1 = nn.L1Loss()
        self.mape = MAPE()
        self.normalization_coeffs = None

    def forward(self, x, y_old=None):
        # ADDED FOR REVIN: apply RevIN normalization if enabled
        if self.use_revin:
            x = rearrange(x, 'b f s -> b s f')
            x = self.revin(x, 'norm')
            x = rearrange(x, 'b s f -> b f s')
        # END ADDED FOR REVIN

        if self.mode == 'default':
            y_hat_norm = self.model(x).reshape(-1)
        elif self.mode == 'diff':
            y_hat_norm = self.model(x).reshape(-1) + y_old

        # ADDED FOR REVIN: denormalize if use_revin
        if self.use_revin:
            y_hat = self.revin(y_hat_norm, 'denorm', target_idx=self.target_idx)
        else:
            y_hat = y_hat_norm
        # For training, return both for loss calculation
        if self.training and self.use_revin:
            return y_hat, y_hat_norm
        return y_hat
        # END ADDED FOR REVIN

    def set_normalization_coeffs(self, factors):
        if factors is None:
            return
        scale = factors.get(self.y_key).get('max') - factors.get(self.y_key).get('min')
        shift = factors.get(self.y_key).get('min')
        self.normalization_coeffs = (scale, shift)

    def denormalize(self, y, y_hat):
        if self.normalization_coeffs is not None:
            scale, shift = self.normalization_coeffs
            y = y * scale + shift
            y_hat = y_hat * scale + shift
        return y, y_hat

    def training_step(self, batch, batch_idx):
        x = batch['features']
        y = batch[self.y_key]
        y_old = batch[f'{self.y_key}_old']
        if self.batch_size is None:
            self.batch_size = x.shape[0]

        # MODIFIED FOR REVIN: handle loss on normalized space if use_revin
        if self.use_revin:
            y_hat, y_hat_norm = self.forward(x, y_old)
            mean = self.revin.mean[:, 0, self.target_idx]
            stdev = self.revin.stdev[:, 0, self.target_idx]
            y_trans = (y - mean) / (stdev + self.revin.eps)
            if self.revin.affine:
                w = self.revin.weight[self.target_idx]
                b = self.revin.bias[self.target_idx]
                y_norm = y_trans * w + b
            else:
                y_norm = y_trans
            mse_norm = self.mse(y_hat_norm, y_norm)
            mse = self.mse(y_hat, y)
            rmse = torch.sqrt(mse)
            mape = self.mape(y_hat, y)
            l1 = self.l1(y_hat, y)
            loss = mse_norm
        else:
            y_hat = self.forward(x, y_old).reshape(-1)
            y, y_hat = self.denormalize(y, y_hat)
            mse = self.mse(y_hat, y)
            rmse = torch.sqrt(mse)
            mape = self.mape(y_hat, y)
            l1 = self.l1(y_hat, y)
            loss = mse
        # END MODIFIED FOR REVIN

        self.log("train/mse", mse.detach(), batch_size=self.batch_size, sync_dist=True, prog_bar=False)
        self.log("train/rmse", rmse.detach(), batch_size=self.batch_size, sync_dist=True, prog_bar=True)
        self.log("train/mape", mape.detach(), batch_size=self.batch_size, sync_dist=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch['features']
        y = batch[self.y_key]
        y_old = batch[f'{self.y_key}_old']
        if self.batch_size is None:
            self.batch_size = x.shape[0]

        # MODIFIED FOR REVIN: compute denorm metrics, return denorm mse for val_loss
        if self.use_revin:
            y_hat = self.forward(x, y_old)
            mse = self.mse(y_hat, y)
            rmse = torch.sqrt(mse)
            mape = self.mape(y_hat, y)
            l1 = self.l1(y_hat, y)
        else:
            y_hat = self.forward(x, y_old).reshape(-1)
            y, y_hat = self.denormalize(y, y_hat)
            mse = self.mse(y_hat, y)
            rmse = torch.sqrt(mse)
            mape = self.mape(y_hat, y)
            l1 = self.l1(y_hat, y)
        # END MODIFIED FOR REVIN

        self.log("val/mse", mse.detach(), sync_dist=True, batch_size=self.batch_size, prog_bar=False)
        self.log("val/rmse", rmse.detach(), sync_dist=True, batch_size=self.batch_size, prog_bar=True)
        self.log("val/mape", mape.detach(), sync_dist=True, batch_size=self.batch_size, prog_bar=True)
        self.log("val/mae", l1.detach(), batch_size=self.batch_size, sync_dist=True, prog_bar=False)
        return {"val_loss": mse}

    def test_step(self, batch, batch_idx):
        x = batch['features']
        y = batch[self.y_key]
        y_old = batch[f'{self.y_key}_old']
        if self.batch_size is None:
            self.batch_size = x.shape[0]

        # MODIFIED FOR REVIN: compute denorm metrics, return denorm mse for test_loss
        if self.use_revin:
            y_hat = self.forward(x, y_old)
            mse = self.mse(y_hat, y)
            rmse = torch.sqrt(mse)
            mape = self.mape(y_hat, y)
            l1 = self.l1(y_hat, y)
        else:
            y_hat = self.forward(x, y_old).reshape(-1)
            y, y_hat = self.denormalize(y, y_hat)
            mse = self.mse(y_hat, y)
            rmse = torch.sqrt(mse)
            mape = self.mape(y_hat, y)
            l1 = self.l1(y_hat, y)
        # END MODIFIED FOR REVIN

        self.log("test/mse", mse.detach(), sync_dist=True, batch_size=self.batch_size, prog_bar=False)
        self.log("test/rmse", rmse.detach(), sync_dist=True, batch_size=self.batch_size, prog_bar=True)
        self.log("test/mape", mape.detach(), sync_dist=True, batch_size=self.batch_size, prog_bar=True)
        self.log("test/mae", l1.detach(), batch_size=self.batch_size, sync_dist=True, prog_bar=False)
        return {"test_loss": mse}

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
        scheduler = torch.optim.lr_scheduler.StepLR(optim, 
                                                   self.lr_step_size, 
                                                   self.lr_gamma)
        return [optim], [scheduler]

    def lr_scheduler_step(self, scheduler, *args, **kwargs):
        scheduler.step()
