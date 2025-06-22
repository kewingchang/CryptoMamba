# 🚀 Version_0: 当前使用模型

## 概述

`version_0` 是当前所使用模型。

训练配置文件存储在 `config` 和 `data` 目录中。以下是相关文件描述：
1. **ETH_Dataset_daily.csv**：包含以太坊（ETH）的部分历史数据（仅模型训练时段数据），原始数据格式，未经处理。  
   来源：[Yahoo Finance ETH-USD 历史数据](https://finance.yahoo.com/quote/ETH-USDdatasethistory/)。
2. **mode_1.yaml**：数据配置文件，位于 `configs/data_configs/mode_1.yaml`。

训练完成后，以下文件将生成于在 `data/2018-09-17_2024-09-16_86400` 目录中：
- `config.pkl`
- `test.csv`
- `train.csv`
- `val.csv`

## 特别注意事项

ETH_Dataset_daily.csv 应与 one_day_pred.csv 保持一致。  
1. **ETH_Dataset_daily.csv**：原始数据，价格保留多达 5 位小数。  
2. **one_day_pred.csv**：用于预测价格的文件也应为原始数据（保留多位小数）。。

## 训练前准备工作

1. 下载 `ETH_Dataset_daily.csv` 并放置在 `data` 目录中。  
2. 根据需要修改配置文件 `configs/data_configs/mode_1.yaml`。

## 训练后模型

1. **训练模型**：位于 `logs/CMamba/version_0/checkpoints/epoch967-val-rmse46.1179.ckpt`。  
2. **最后保存的模型**（不一定是最终模型）：位于 `logs/CMamba/version_0/checkpoints/last.ckpt`。

## 预测步骤

1. 将模型检查点 `logs/CMamba/version_0/checkpoints/epoch967-val-rmse46.1179.ckpt` 复制到 `./checkpoints` 目录，并重命名为 `cmamba_v.ckpt`。  
2. 从 [Yahoo Finance](https://finance.yahoo.com/quote/ETH-USD/history/) 下载预测日期前 15 天的以太坊历史数据，重命名为 `one_day_pred.csv`，并放置在 `./data` 目录中。

## 📈 Results

<div align="center">

| Model | RMSE | MAPE | MAE | Parameters |
| :--: | :--: | :--: | :--: |  :--: |
| CryptoMamba-v | **98.6** | **2.47** | **69.8** | **136k** |

</div>
