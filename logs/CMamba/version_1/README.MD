# 🚀 模型一训练备注

## 概述

本项目采用与 `version_0` 相同的训练配置和数据，但训练结果略有不同（稍差）。  
训练配置与数据文件存储在 `config & data` 目录中。以下是关键文件的介绍：

1. **ETH_Dataset_daily.csv**：包含以太坊（ETH）的完整历史数据，原始格式，未经处理。  
   来源：[Yahoo Finance ETH-USD 历史数据](https://finance.yahoo.com/quote/ETH-USD/history/)
2. **mode_1.yaml**：数据配置文件，位于 `configs/data_configs/mode_1.yaml`。
   注意：jumps = 86400；未指定windows_size

训练完成后，以下文件将生成在 `data/2018-09-17_2024-09-16_86400` 目录中：
- `config.pkl`
- `test.csv`
- `train.csv`
- `val.csv`

## 特别注意

1. **ETH_Dataset_daily.csv**：原始数据，价格保留多达5位小数。
2. **one_day_pred.csv**：用于预测价格的文件也应为原始数据（保留多位小数）。

## 训练前准备

1. 下载 `ETH_Dataset_daily.csv` 并放置在 `data` 目录中。
2. 根据需要修改配置文件 `configs/data_configs/mode_1.yaml`。

## 训练后模型

1. **训练模型**：位于 `logs/CMamba/version_1/checkpoints/epoch932-val-rmse46.8029.ckpt`。
2. **最后保存的模型**（不一定是最终模型）：位于 `logs/CMamba/version_1/checkpoints/last.ckpt`。

## 预测步骤

1. 将模型检查点 `logs/CMamba/version_1/checkpoints/epoch932-val-rmse46.8029.ckpt` 复制到 `./checkpoints` 目录，并重命名为 `cmamba_v.ckpt`。
2. 从 [Yahoo Finance](https://finance.yahoo.com/quote/ETH-USD/history/) 下载预测日期前15天的以太坊历史数据，重命名为 `one_day_pred.csv`，并放置在 `./data` 目录中。

## 📈 结果

<div align="center">

| 模型 | RMSE | MAPE | MAE | 参数量 |
| :--: | :--: | :--: | :--: |  :--: |
| CryptoMamba-v | **101.09** | **2.59** | **72.82** | **136k** |

</div>
