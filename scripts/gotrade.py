# gotrade.py
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
        "--model",
        required=False,
        type=str,
        default='v2',
        help="Path to model config file.",
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
    parser.add_argument(
        '--paper_trading', 
        default=False,   
        action='store_true',          
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

def load_model(config, model_path, ckpt_path, feature_names=None, skip_revin=None):
    model_config_path = f'{ROOT}/configs/models/CryptoMamba/{model_path}.yaml'
    model_config = io_tools.load_config_from_yaml(model_config_path)
    normalize = model_config.get('normalize', False)
    # 注入参数
    if feature_names is not None:
        model_config.get('params')['feature_names'] = feature_names
    if skip_revin is not None:
        model_config.get('params')['skip_revin'] = skip_revin
    model_class = io_tools.get_obj_from_str(model_config.get('target'))
    model = model_class.load_from_checkpoint(ckpt_path, **model_config.get('params'),weights_only=False)
    model.cuda()
    model.eval()
    return model, normalize

@torch.no_grad()
def run_model(model, dataloader):
    target_list = []
    preds_list = []
    timetamps = []
    with torch.no_grad():
        for batch in dataloader:
            ts = batch.get('Timestamp').numpy().reshape(-1)
            target = batch.get('Close').numpy().reshape(-1)
            features = batch.get('features').to(model.device)
            preds = model(features).cpu().numpy().reshape(-1)
            target_list += [float(x) for x in list(target)]
            preds_list += [float(x) for x in list(preds)]
            timetamps += [float(x) for x in list(ts)]
    targets = np.asarray(target_list)
    preds = np.asarray(preds_list)
    targets_tensor = torch.tensor(target_list)
    preds_tensor = torch.tensor(preds_list)
    timetamps = [datetime.fromtimestamp(int(x)) for x in timetamps]
    loss = float(model.loss(preds_tensor, targets_tensor))
    mape = float(model.mape(preds_tensor, targets_tensor))
    return timetamps, targets, preds, loss, mape



if __name__ == "__main__":

    args = get_args()

    config = io_tools.load_config_from_yaml(f'{ROOT}/configs/training/{args.config}.yaml')

    data_config = io_tools.load_config_from_yaml(f"{ROOT}/configs/data_configs/{config.get('data_config')}.yaml")

    use_volume = config.get('use_volume', args.use_volume)

    # [移位] 先创建 Transform 以获取 keys
    # train_transform = DataTransform(is_train=True, use_volume=use_volume, additional_features=config.get('additional_features', []))
    # val_transform = DataTransform(is_train=False, use_volume=use_volume, additional_features=config.get('additional_features', []))
    # test_transform = DataTransform(is_train=False, use_volume=use_volume, additional_features=config.get('additional_features', []))

    # [新增] 获取 feature_names
    # feature_names = [k for k in test_transform.keys if k != 'Timestamp_orig']
    feature_names = ['Open', 'High', 'Low', 'Close']
    if use_volume:
        feature_names.append('Volume')
    # additional_features 里包含了 'log_return'
    feature_names += config.get('additional_features', [])
    skip_revin_list = config.get('skip_revin', [])

    print(f"Features for inference: {feature_names}")

    # 2. 加载模型
    model_path = args.model
    model, normalize = load_model(config, model_path, args.ckpt_path, feature_names, skip_revin_list)

    data = pd.read_csv(args.data_path)
    if 'Date' in data.keys():
        data['Timestamp'] = [float(time.mktime(datetime.strptime(x, "%Y-%m-%d").timetuple())) for x in data['Date']]
    data = data.sort_values(by='Timestamp').reset_index()
    
    # end_date = "2024-27-10"
    if args.date is None:
        end_ts = max(data['Timestamp']) + 24 * 60 * 60
    else:
        end_ts = int(time.mktime(datetime.strptime(args.date, "%Y-%m-%d").timetuple()))
    start_ts = end_ts - 14 * 24 * 60 * 60 - 60 * 60
    pred_date = datetime.fromtimestamp(end_ts).strftime("%Y-%m-%d")
    data = data[data['Timestamp'] < end_ts]
    data = data[data['Timestamp'] >= start_ts - 60 * 60]

    txt_file = init_dirs(args, pred_date)
    
    
    features = {}
    key_list = ['Open', 'High', 'Low', 'Close']
    if use_volume:
        key_list.append('Volume')
    key_list += config.get('additional_features', [])  # 新增：包含 additional_features，如 ['marketCap']
    
    for key in key_list:
        tmp = list(data.get(key))
        if normalize:
            pass
            # scale = data_module.factors.get(key).get('max') - data_module.factors.get(key).get('min')
            # shift = data_module.factors.get(key).get('min')
        else:
            scale = 1
            shift = 0
        if key == 'Volume':
            tmp = [x / 1e9 for x in tmp]
        tmp = [(x - shift) / scale for x in tmp]
        features[key] = torch.tensor(tmp).reshape(1, -1)
        if key == 'Timestamp':
            t_scale = scale
            t_shift = shift
        if key == model.y_key:
            scale_pred = scale
            shift_pred = shift

    x = torch.cat([features.get(x) for x in key_list], dim=0)  # 注意：这里使用 key_list，确保顺序与训练一致

    # close_idx = -2 if use_volume else -1  # 如果添加了 marketCap，close_idx 需调整为 key_list.index('Close') - len(key_list)，但当前 -1 是 marketCap，-2 是 Close？不。
    # 修正：close_idx 应基于 key_list 的位置；当前 key_list[-2] 是 Close（因为最后是 marketCap），所以 close_idx = -2 if not use_volume else 调整
    # 但原代码 close_idx = -2 if use_volume else -1；由于 use_volume=False，且多了 marketCap，Close 是倒数第二，所以 close_idx = -2
    close_idx = key_list.index('Close')  # 用 index 动态获取，避免硬编码
    today = float(x[close_idx, -1]) * scale_pred + shift_pred

    with torch.no_grad():
        pred = model(x[None, ...].cuda()).cpu()
        if normalize:
            pred = float(pred) * scale_pred + shift_pred
        elif hasattr(model.model, 'revin') and model.model.revin is not None:
            pred_tensor = pred.reshape(-1).to(model.device)  # Add .to(model.device)
            _, pred_tensor = model.denormalize(None, pred_tensor)
            pred = float(pred_tensor.cpu()[0])  # Move back to CPU if needed
        else:
            pred = float(pred)
    
    # 6. 价格还原
    pred_price = pred
    today_price = today

    print('')
    print_and_write(txt_file, f'Prediction date: {pred_date}')
    print_and_write(txt_file, f'Today Close (Entry Ref): {round(today_price, 2)}')
    print_and_write(txt_file, f'Prediction Close: {round(pred_price, 2)}')
    
    # 计算预测涨跌幅
    pct_change = (pred_price - today_price) / today_price * 100
    print_and_write(txt_file, f'Predicted Change: {round(pct_change, 2)}%')

    # 7. 实战交易逻辑 (Real-World Trading Logic)
    print_and_write(txt_file, '-' * 30)
    # print_and_write(txt_file, '>>> TRADING SIGNAL ANALYSIS <<<')

    # A. 计算 Smart Factor (x)
    # x = |Pred - Today| / (Today * Risk%)
    risk_percent = args.risk / 100.0  # e.g., 2.2% -> 0.022
    diff_threshold = today_price * risk_percent
    raw_diff = pred_price - today_price
    
    # 避免除以0
    if diff_threshold == 0: diff_threshold = 1e-8
    x_factor = abs(raw_diff) / diff_threshold

    # 设定两个门槛
    X_STRONG_THRESHOLD = 0.5
    X_WEAK_THRESHOLD = 0.2

    print_and_write(txt_file, f'Signal Strength (x): {round(x_factor, 2)}')

    trade_mode = None
    direction = "LONG" if pred_price > today_price else "SHORT"
    
    # 决策逻辑分支
    if x_factor >= X_STRONG_THRESHOLD:
        trade_mode = 'aggressive'
        print_and_write(txt_file, f'[SIGNAL]: STRONG - NO TRADE')
    elif x_factor >= X_WEAK_THRESHOLD:
        trade_mode = 'conservative'
        print_and_write(txt_file, f'[SIGNAL]: WEAK')
    else:
        print_and_write(txt_file, f'[SIGNAL]: WAIT')

    # 执行计算
    if trade_mode and not args.paper_trading:
        pass
    elif trade_mode == 'aggressive':
        pass
    else:
        print_and_write(txt_file, '-' * 20)
        print_and_write(txt_file, f'{direction}')
        print_and_write(txt_file, '-' * 20)
