<img width="71" height="8192" alt="image" src="https://github.com/user-attachments/assets/d25d70c1-4468-4795-a85c-48e183b64727" /><p align="center">
 <img src="assets/logo.png" alt="drawing" width="200" style="float: center;"/> 
</p>

<h1 align="center">🚀 CryptoMamba: Leveraging State Space Models for Accurate Bitcoin Price Prediction</h1>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License"/></a>
  <a href="https://github.com/MShahabSepehri/CryptoMamba/stargazers"><img src="https://img.shields.io/github/stars/MShahabSepehri/CryptoMamba?style=social" alt="GitHub Stars"/></a>
</p>

> 📣 **Announcements**  
> CryptoMamba paper has been published at [IEEE International Conference on Blockchain and Cryptocurrency (ICBC) 2025](https://icbc2025.ieee-icbc.org/)! 🎉  
> We presented CryptoMamba at the [Advances in Financial AI Workshop @ ICLR 2025](https://sites.google.com/view/financialaiiclr25/).

<p align="justify" > 
<strong>CryptoMamba</strong> is a novel Mamba-based architecture designed for accurate and efficient time-series forecasting, with a focus on cryptocurrency price prediction. Leveraging the capabilities of SSMs, CryptoMamba excels at capturing long-range dependencies and adapting to highly volatile market conditions.
</p>

Repository Includes:
- **Implementation** of CryptoMamba and baseline models (LSTM, Bi-LSTM, GRU, and S-Mamba).  
- **Two Trading Algorithms**: Vanilla and Smart, for evaluating real-world performance.  
- **Code** for data preprocessing, model training, evaluation metrics, and trading simulations.  

<p align="justify"> 
CryptoMamba’s robust performance and generalizability make it a promising solution for various sequential forecasting tasks, including financial markets, commodities, and other time-series applications.
</p>

## 📖 Table of Contents

  * [Requirements](#-requirements)
  * [Usage](#-usage)
    * [Data](#data)
    * [Config](#config)
    * [Train New Model](#train-new-model)
    * [Evaluate Model](#evaluate-model)
    * [Predict Next Day Price](#predict-next-day-price)
  * [Results](#-results)
  * [Citation](#-citation)
  * [Where to Ask for Help](#-where-to-ask-for-help)

## 🔧 Requirements

To install the requirements, you can use:

```
pip install -r requirements.txt
```

<p align="justify" > 
If you have difficulties installing <code>mamba_ssm</code>, please follow the instructions in <a href="https://github.com/state-spaces/mamba">its GitHub repository</a>.
</p>

## 💡 Usage

### Data

<p align="justify" > 
You can find the processed data that we use in <a href="./data/2018-09-17_2024-09-16_86400">here</a>. If you want to use another data configuration, you should change the configuration in <a href="./configs/data_configs/mode_1.yaml">the data config file</a>. Note that the <code>data_path</code> should point to the raw data file with a similar format to <a href="./data/one_day_pred.csv">this</a>.
</p>


### Config
<p align="justify" > 
If you want to use additional features other than Open, Close, High, Low, Timestamp, and Volume, you should specify a list called <code>additional_features</code> in your data and training configuration files. Note that your raw data file should have these features. 
</br>
If you want to change the time resolution, you should change the <code>date_format</code> in your data configuration file and also set <code>jumps</code> to your desired resolution in seconds. Note that your raw data dates and your start and end dates in the data configuration should follow the new date format.
</p>


### Train New Model
To train a model, use the following: 

```
python3 scripts/training.py --config CONFIG_NAME
```
<p align="justify" > 
Here, <code>CONFIG_NAME</code> is the name of a config file in <a href="configs/training/">the training config folder</a> without its extension. For example, to train CryptoMamba with volume you can run the following command:
</p>

```
python3 scripts/training.py --config cmamba_v
```

### Evaluate Model

To evaluate a model, run this command:

```
python scripts/evaluation.py --config cmamba_v --ckpt_path ./checkpoints/ETH_cmamba_v.ckpt
```

To run the trading simulation, you can use the following command:

```
python scripts/simulate_trade.py --config CONFIG_NAME --ckpt_path PATH_TO_CHECKPOINT --split SPLIT_NAME --trade_mode TRADE_ALGORITHM
```

Where `SPLIT_NAME` is `train`, `val`, or `test` and `TRADE_ALGORITHM is` `smart` or `vanilla` or `smart_w_short`.

### Predict Next Day Price
<p align="justify" > 
We also have a script to predict the next day's price and trade suggestions by providing the prices of its previous days. You have to create a <code>csv</code> file similar to [this](data/one_day_trade.csv) and use this command:
</p>

```
python scripts/one_day_pred.py --config CONFIG_NAME --ckpt_path PATH_TO_CHECKPOINT --date DATE
```

If you don't provide a value for `DATE`, the code automatically predicts one day after the last day that is in the file.

<!-- ### Trading Suggestion -->
 

## 📈 Results

<div align="center">

| Model | RMSE | MAPE | MAE | Parameters |
| :--: | :--: | :--: | :--: |  :--: |
| LSTM | 2672.7 | 3.609 | 2094.3 | 204K |
| Bi-LSTM | 2325.6 | 3.072 | 1778.8 | 569k |
| GRU | 1892.4 | 2.385 | 1371.2 | 153k |
| iTransformer | 1826.9 | 2.4260 | 1334.3 | 201k |
| S-Mamba | 1717.4 | 2.248 | 1239.9 | 330k |
| CryptoMamba | **1713.0** | **2.171** | **1200.9** | **136k** |
| LSTM-v | 2202.1 | 2.896 | 1668.9 | 204K |
| Bi-LSTM-v | 2080.2 | 2.738 | 1562.5 | 569k |
| GRU-v | 1978.0 | 2.526 | 1454.3  | 153k |
| iTransformer-v | 1779.9 | 2.427 | 1322.1 | 201k |
| S-Mamba-v | 1651.6 | 2.215 | 1209.7 | 330k |
| CryptoMamba-v | **1598.1** | **2.034** | **1120.7** | **136k** |

</div>

<!---
## 📚 Paper

-->
## 🎯 Citation 

If you use CryptoMamba in a research paper, please cite our [paper](https://arxiv.org/abs/2501.01010):

```bibtex
@article{Sepehri2025CryptoMamba,
      title={CryptoMamba: Leveraging State Space Models for Accurate Bitcoin Price Prediction}, 
      author={Mohammad Shahab Sepehri and Asal Mehradfar and Mahdi Soltanolkotabi and Salman Avestimehr},
      year={2025},
      url={https://arxiv.org/abs/2501.01010}
}
```

## ❓ Where to Ask for Help

<p align="justify" > 
If you have any questions, feel free to open a <a href="https://github.com/MShahabSepehri/CryptoMamba/discussions">Discussion</a> and ask your question. You can also email <a href="mailto:sepehri@usc.edu">sepehri@usc.edu</a> (Mohammad Shahab Sepehri) or <a href="mailto:mehradfa@usc.edu">mehradfa@usc.edu</a> (Asal Mehradfar).
</p>
