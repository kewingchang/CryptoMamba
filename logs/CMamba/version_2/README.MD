# 🚀 训练备注

## 概述

本项目主要特征：
1. 训练数据仅保留2位小数。
2. 设置jumps=1440，window_size=14

训练配置文件存储在 `config & data` 目录中。以下是相关文件描述：
1. **ETH_Dataset_daily.csv**：包含以太坊（ETH）的完整历史数据，原始数据格式，未经处理。  
   来源：[Yahoo Finance ETH-USD 历史数据](https://finance.yahoo.com/quote/ETH-USDdatasethistory/)。
2. **mode_1.yaml**：数据配置文件，位于 `configs/data_configs/mode_1.yaml`。

训练完成后，以下文件将生成于在 `data/2018-09-17_2024-09-16_1440` 目录中：
- `config.pkl`
- `test.csv`
- `train.csv`
- `val.csv`

## 特别注意事项

ETH_Dataset_daily.csv 应与 one_day_pred.csv 保持一致。  
1. **ETH_Dataset_daily.csv**：原始数据，价格仅保留2位小数。  
2. **one_day_pred.csv**：用于预测价格的文件也应为原始数据（保留2位小数）。

## 训练前准备工作

1. 下载 `ETH_Dataset_daily.csv`,数据预处理仅保留2位小时，放置在 `data` 目录中。  
2. 根据需要修改配置文件 `configs/data_configs/mode_1.yaml`。

## 训练后模型

1. **训练模型**：位于 `logs/CMamba/version_2/checkpoints/epoch972-val-rmse51.6859.ckpt`。  
2. **最后保存的模型**（不一定是最终模型）：位于 `logs/CMamba/version_2/checkpoints/last.ckpt`。

## 预测步骤

1. 将模型检查点 `logs/CMamba/version_2/checkpoints/epoch972-val-rmse51.6859.ckpt` 复制到 `./checkpoints` 目录，并重命名为 `cmamba_v.ckpt`。  
2. 从 [Yahoo Finance](https://finance.yahoo.com/quote/ETH-USD/history/) 下载预测日期前 15 天的以太坊历史数据，重命名为 `one_day_pred.csv`，并放置在 `./data` 目录中。

## 📈 结果

<div align="center">

| 模型 | RMSE | MAPE | MAE | 参数量 |
| :--: | :--: | :--: | :--: |  :--: |
| CryptoMamba-v | **109.89** | **2.73** | **76.47** | **136k** |

</div>
