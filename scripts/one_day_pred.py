# one_day_pred.py
import os, sys, pathlib
sys.path.insert(0, os.path.dirname(pathlib.Path(__file__).parent.absolute()))

import time
import yaml
import torch
import numpy as np
import pandas as pd
from utils import io_tools
from datetime import datetime
from argparse import ArgumentParser
from pl_modules.data_module import CMambaDataModule
from data_utils.data_transforms import DataTransform
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

ROOT = io_tools.get_root(__file__, num_returns=2)

def get_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--accelerator",
        type=str,
        default='gpu',
        help="The type of accelerator.",
    )
    parser.add_argument(
        "--date",
        type=str,
        default=None,
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
        "--config",
        type=str,
        default='cmamba_v',
        help="Path to config file.",
    )
    parser.add_argument(
        '--use_volume', 
        default=False,   
        action='store_true',          
    )
    parser.add_argument(
        "--data_path",
        default='data/one_day_pred.csv',
        type=str,
        help="Path to config file.",
    )
    parser.add_argument(
        "--ckpt_path",
        required=True,
        type=str,
    )
    parser.add_argument(
        "--risk",
        default=2,
        type=float,
    )

    args = parser.parse_args()
    return args

def print_and_write(file, txt, add_new_line=True):
    print(txt)
    if add_new_line:
        file.write(f'{txt}\n')
    else:
        file.write(txt)

def init_dirs(args, date):
    path = f'{ROOT}/Predictions/{args.config}/'
    if not os.path.isdir(path):
        os.makedirs(path)
    txt_file = open(f'{path}/{date}.txt', 'w')
    return txt_file

def load_model(config, ckpt_path, feature_names=None, skip_revin=None):
    arch_config = io_tools.load_config_from_yaml('configs/models/archs.yaml')
    model_arch = config.get('model')
    model_config_path = f'{ROOT}/configs/models/{arch_config.get(model_arch)}'
    model_config = io_tools.load_config_from_yaml(model_config_path)
    
    # 注入训练配置中的超参数
    hyperparams = config.get('hyperparams')
    if hyperparams is not None:
        for key in hyperparams.keys():
            model_config.get('params')[key] = hyperparams.get(key)

    # 注入特征名称
    if feature_names is not None:
        model_config.get('params')['feature_names'] = feature_names
    if skip_revin is not None:
        model_config.get('params')['skip_revin'] = skip_revin
        
    model_class = io_tools.get_obj_from_str(model_config.get('target'))
    
    # 实例化并加载权重
    model = model_class.load_from_checkpoint(
        ckpt_path, 
        **model_config.get('params'),
        weights_only=False
    )
    model.cuda()
    model.eval()
    return model

if __name__ == "__main__":
    args = get_args()
    config = io_tools.load_config_from_yaml(f'{ROOT}/configs/training/{args.config}.yaml')
    use_volume = config.get('use_volume', args.use_volume)  
        
    # 这里必须与训练时的特征顺序完全一致
    # 模型 3: Open, High, Close, log_return_low
    # 获取 feature_names
    feature_names = []
    if use_volume:
        feature_names.append('Volume')
    # additional_features 里包含了 'log_return_low'
    feature_names += config.get('additional_features', [])
    skip_revin_list = config.get('skip_revin', [])

    print(f"Features for inference: {feature_names}")

    # 2. 加载模型
    model = load_model(config, args.ckpt_path, feature_names, skip_revin_list)

    # 3. 加载并处理数据
    data = pd.read_csv(args.data_path)
    if 'Date' in data.keys():
        data['Timestamp'] = [float(time.mktime(datetime.strptime(x, "%Y-%m-%d").timetuple())) for x in data['Date']]
    data = data.sort_values(by='Timestamp').reset_index(drop=True)

    # 特征工程 (Feature Engineering) 
    # 必须与 dataset.py 中的逻辑保持一致    
    # 计算 log_return_low (如果模型需要)
    # log_return_low = ln(Low / Open)
    if 'log_return_low' in feature_names:
        data['log_return_low'] = np.log(data['Low'] + 1e-8) - np.log(data['Open'] + 1e-8)

    # 新增处理 log_return_high (适配 High 模型)
    if 'log_return_high' in feature_names:
        # ln(High / Open)
        data['log_return_high'] = np.log(data['High'] + 1e-8) - np.log(data['Open'] + 1e-8)

    # 填充 NaN
    data = data.fillna(0.0).replace([np.inf, -np.inf], 0.0)
    
    # 确定预测日期
    if args.date is None:
        end_ts = max(data['Timestamp']) + 24 * 60 * 60
    else:
        end_ts = int(time.mktime(datetime.strptime(args.date, "%Y-%m-%d").timetuple()))
    
    pred_date = datetime.fromtimestamp(end_ts).strftime("%Y-%m-%d")
    data_window = data[data['Timestamp'] < end_ts].copy()
    
    window_size = model.window_size
    if len(data_window) < window_size + 1:
        raise ValueError(f"Not enough data to predict. Need at least {window_size+1} rows.")
    
    # 取最后 window_size 行作为输入
    input_df = data_window.iloc[-window_size:]
    
    # 记录基准价格 (用于还原)
    # 注意：log_return_low 是相对于 Open 的，所以我们需要明天的 Open
    # 但明天的 Open 我们不知道，通常假设 明日Open = 今日Close (或者如果你的实盘策略是明日开盘即入场)
    last_close_price = float(input_df.iloc[-1]['Close'])
    # last_open_price = float(input_df.iloc[-1]['Open']) # 如果你需要相对于今日Open
    
    txt_file = init_dirs(args, pred_date)

    # 4. 构建 Input Tensor
    features_map = {}
    for key in feature_names:
        tmp = list(input_df.get(key))
        if key == 'Volume':
            tmp = [x / 1e9 for x in tmp]
        # 注意：这里移除了 dataset 级别的 normalize 逻辑，因为你似乎没用 normalize=True
        features_map[key] = torch.tensor(tmp).float().reshape(1, -1)

    # 拼接: (Num_Features, Window_Size)
    x = torch.cat([features_map.get(k) for k in feature_names], dim=0)

    # 5. 推理
    with torch.no_grad():
        x_input = x.unsqueeze(0).cuda() # (1, C, L)
        
        # Model forward -> 得到 Raw Output (B, Output_Dim)
        # 注意：不要 reshape(-1)，保留 (1, Output_Dim) 结构
        pred_raw = model(x_input) 
        
        # RevIN 反归一化
        if hasattr(model.model, 'revin') and model.model.revin is not None:
             # BaseModule.denormalize 支持多头输出的反归一化
             # 传入 None 作为 target (y)，因为我们在推理
             _, pred_denorm = model.denormalize(None, pred_raw)
        else:
             pred_denorm = pred_raw

        # 转为 numpy
        pred_vals = pred_denorm.cpu().numpy().flatten()

    # 6. 结果解析与打印
    # 获取分位数列表 (从 config 中)
    quantiles = config.get('hyperparams', {}).get('quantiles', [])
    if len(quantiles) != len(pred_vals):
        print(f"Warning: Config quantiles count ({len(quantiles)}) != Prediction output count ({len(pred_vals)})")

    print('')
    print_and_write(txt_file, f'Prediction date: {pred_date}')
    print_and_write(txt_file, f'Ref Price (Last Close): {round(last_close_price, 2)}')
    print_and_write(txt_file, '-'*30)

    for i, q in enumerate(quantiles):
        # 这里的还原公式是通用的，但变量名在 High 模型下也是 log_return_high
        # Price = Open * exp(predicted_log_return)
        price_level = last_close_price * np.exp(pred_vals[i])
        
        # 动态调整 Label 显示
        label = f"q={q}"
        if q == 0.05: label += " (Low Stop)"
        elif q == 0.95: label += " (High Stop)" # 新增 High 模型标识
        elif q == 0.60: label += " (Buy 1/Sell 3)" # 简单示意
            
        print_and_write(txt_file, f'{label:<15}: {round(price_level, 2)} (Return: {pred_vals[i]*100:.2f}%)')

    print_and_write(txt_file, '-'*30)