This is the model training log. The training model epoch967-val-rmse46.1179.ckpt is here. To use it, copy the model to ./checkpoints/cmamba_v.ckpt 

The last lines of the training log are as follows: 
Epoch 999: 100% 46/46 [00:04<00:00, 10.57it/s, v_num=0, train/rmse=24.00, train/mape=0.0363, val/rmse=178.0, val/mape=0.109]
Restoring states from the checkpoint path at logs/CMamba/version_0/checkpoints/epoch967-val-rmse46.1179.ckpt 
LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES. [0] 
Loaded model weights from the checkpoint at logs/CMamba/version_0/checkpoints/epoch967-val-rmse46.1179.ckpt 
350 data points loaded as test split.
Testing DataLoader 0: 100% 11/11 [00:00<00:00, 52.75it/s] 
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┓ 
┃        Test metric        ┃       DataLoader 0        ┃ 
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━┩ 
│         test/mae          │ 69. 83438110351562        │ 
│ test/mape                 │ 0.024765657261013985      │ 
│ test/mse                  │ 10060.7314453125          │ 
│ test/rmse                 │ 98.6056137084961          │
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━┩ 
