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
from utils.trade import buy_sell_vanilla, buy_sell_smart
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
        type=int,
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
    normalize = model_config.get('normalize', False)
    # [新增] 注入参数
    if feature_names is not None:
        model_config.get('params')['feature_names'] = feature_names
    if skip_revin is not None:
        model_config.get('params')['skip_revin'] = skip_revin
    model_class = io_tools.get_obj_from_str(model_config.get('target'))
    model = model_class.load_from_checkpoint(ckpt_path, **model_config.get('params'),weights_only=False)
    model.cuda()
    return model, normalize


if __name__ == "__main__":

    args = get_args()

    config = io_tools.load_config_from_yaml(f'{ROOT}/configs/training/{args.config}.yaml')

    data_config = io_tools.load_config_from_yaml(f"{ROOT}/configs/data_configs/{config.get('data_config')}.yaml")

    use_volume = config.get('use_volume', args.use_volume)

    # [移位] 先创建 Transform 以获取 keys
    train_transform = DataTransform(is_train=True, use_volume=use_volume, additional_features=config.get('additional_features', []))
    val_transform = DataTransform(is_train=False, use_volume=use_volume, additional_features=config.get('additional_features', []))
    test_transform = DataTransform(is_train=False, use_volume=use_volume, additional_features=config.get('additional_features', []))

    # [新增] 获取 feature_names
    feature_names = [k for k in test_transform.keys if k != 'Timestamp_orig']
    skip_revin_list = config.get('skip_revin', [])

    print(f"Features for inference: {feature_names}")

    # 2. 加载模型
    model, normalize = load_model(config, args.ckpt_path, feature_names, skip_revin_list)

    data = pd.read_csv(args.data_path)
    if 'Date' in data.keys():
        data['Timestamp'] = [float(time.mktime(datetime.strptime(x, "%Y-%m-%d").timetuple())) for x in data['Date']]
    data = data.sort_values(by='Timestamp').reset_index(drop=True)

    # === [关键修改]: 手动计算 log_return ===
    # 因为 CSV 里可能没有这一列，或者计算方式不一致
    # log_return = ln(Close_t) - ln(Close_t-1)
    data['log_return'] = np.log(data['Close'] + 1e-8) - np.log(data['Close'].shift(1) + 1e-8)
    data['log_return'] = data['log_return'].fillna(0.0) # 填充第一行
    data['log_return'] = data['log_return'].replace([np.inf, -np.inf], 0.0)
    
    data_module = CMambaDataModule(data_config,
                                   train_transform=train_transform,
                                   val_transform=val_transform,
                                   test_transform=test_transform,
                                   batch_size=1,
                                   distributed_sampler=False,
                                   num_workers=1,
                                   normalize=normalize,
                                   )
    
    # end_date = "2024-27-10"
    if args.date is None:
        end_ts = max(data['Timestamp']) + 24 * 60 * 60
    else:
        end_ts = int(time.mktime(datetime.strptime(args.date, "%Y-%m-%d").timetuple()))
    
    # 截取过去 Window Size 的数据 (通常是 14天)
    # 我们多取一点缓冲
    start_ts = end_ts - 20 * 24 * 60 * 60 
    pred_date = datetime.fromtimestamp(end_ts).strftime("%Y-%m-%d")
    
    # 获取截止到预测日前一天的历史数据
    # 比如预测 2024-10-28，我们需要截至 2024-10-27 的 Close
    data_window = data[data['Timestamp'] < end_ts].copy()
    
    # 确保有足够的数据
    window_size = model.window_size # 比如 14
    if len(data_window) < window_size + 1:
        raise ValueError(f"Not enough data to predict. Need at least {window_size+1} rows.")
    
    # 取最后 window_size + 1 行 (类似于 dataset.py 的逻辑)
    # dataset.py 取 i:i+window_size+1，其中最后一行是 target，前面是 features
    # 但推理时，我们需要用最后 window_size 行作为输入，预测未来
    # 这里我们取最后 window_size 行作为 input features
    input_df = data_window.iloc[-window_size:]
    
    # 记录"今天" (input_df 的最后一行) 的价格，作为还原基准
    last_close_price = float(input_df.iloc[-1]['Close'])
    
    txt_file = init_dirs(args, pred_date)

    # 4. 构建 Input Tensor
    features_map = {}
    
    # 遍历模型需要的特征列表 (feature_names)
    for key in feature_names:
        tmp = list(input_df.get(key))
        
        # 处理 Volume 缩放
        if key == 'Volume':
            tmp = [x / 1e9 for x in tmp]
        
        # 归一化 (如果 normalize=True)
        if normalize and factors:
            scale = factors.get(key).get('max') - factors.get(key).get('min')
            shift = factors.get(key).get('min')
            tmp = [(x - shift) / scale for x in tmp]
        
        # 转换为 Tensor: (1, window_size)
        features_map[key] = torch.tensor(tmp).float().reshape(1, -1)

    # 拼接: (Num_Features, Window_Size)
    # 注意顺序必须与 feature_names 一致
    x = torch.cat([features_map.get(k) for k in feature_names], dim=0)

    # 5. 推理
    with torch.no_grad():
        # Input shape: (1, C, L) -> Batch=1
        x_input = x.unsqueeze(0).cuda()
        
        # Model forward -> 得到归一化的 log_return
        pred_raw = model(x_input).reshape(-1)
        
        # RevIN 反归一化
        if hasattr(model.model, 'revin') and model.model.revin is not None:
             # denormalize 内部需要找到 target index
             # 确保 base_module.denormalize 逻辑正确
             _, pred_log_return = model.denormalize(None, pred_raw)
             pred_log_return = float(pred_log_return.item())
        else:
             pred_log_return = float(pred_raw.item())

    # 6. 价格还原
    # Prediction = Last_Close * exp(Predicted_Log_Return)
    pred_price = last_close_price * np.exp(pred_log_return)
    
    # "Today value" 在这里的语境下，通常指最近已知的价格 (last_close_price)
    today_price = last_close_price

    print('')
    print_and_write(txt_file, f'Prediction date: {pred_date}\nPrediction: {round(pred_price, 2)}\nToday value: {round(today_price, 2)}')

    # 7. 交易逻辑
    # 注意：这里的 buy_sell_smart 需要传入 (Current_Price, Predicted_Price)
    b, s = buy_sell_smart(today_price, pred_price, 100, 100, risk=args.risk)
    
    print("\n")
    print(">>> Smart trade <<<")
    print(f"> 1st Order: {round(today_price, 2)}")
    
    slp = today_price
    if b < 100:
        tmp = round((100 - b), 2)
        slp = today_price * 0.98
        print_and_write(txt_file, f'> Smart trade: {tmp}% buy')
        print(f"> 2nd Order: {round(today_price * 0.99, 2)}")
    if s < 100:
        tmp = round((100 - s), 2)
        slp = today_price * 1.02
        print_and_write(txt_file, f'> Smart trade: {tmp}% sell')
        print(f"> 2nd Order: {round(today_price * 1.01, 2)}")
        
    print("\n")
    print(f"> TP price: {round(pred_price, 2)}")
    print(f"> SL price: {round(slp, 2)}")
    print(">>>..............<<<")
    print("\n")

    b, s = buy_sell_vanilla(today_price, pred_price, 100, 100)
    if b < 100:
        assert b == 0
        print_and_write(txt_file, f'Vanilla trade: buy')
    elif s < 100:
        assert s == 0
        print_and_write(txt_file, f'Vanilla trade: sell')
    else:
        print_and_write(txt_file, f'Vanilla trade: -')
