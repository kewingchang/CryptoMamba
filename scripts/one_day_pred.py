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

def save_all_hparams(log_dir, args):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    save_dict = vars(args)
    save_dict.pop('checkpoint_callback')
    with open(log_dir + '/hparams.yaml', 'w') as f:
        yaml.dump(save_dict, f)

def load_model(config, ckpt_path):
    arch_config = io_tools.load_config_from_yaml('configs/models/archs.yaml')
    model_arch = config.get('model')
    model_config_path = f'{ROOT}/configs/models/{arch_config.get(model_arch)}'
    model_config = io_tools.load_config_from_yaml(model_config_path)
    normalize = model_config.get('normalize', False)
    model_class = io_tools.get_obj_from_str(model_config.get('target'))
    model = model_class.load_from_checkpoint(ckpt_path, **model_config.get('params'))
    model.cuda()
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
    model, normalize = load_model(config, args.ckpt_path)

    data = pd.read_csv(args.data_path)
    if 'Date' in data.keys():
        data['Timestamp'] = [float(time.mktime(datetime.strptime(x, "%Y-%m-%d").timetuple())) for x in data['Date']]
    data = data.sort_values(by='Timestamp').reset_index()

    train_transform = DataTransform(is_train=True, use_volume=use_volume, additional_features=config.get('additional_features', []))
    val_transform = DataTransform(is_train=False, use_volume=use_volume, additional_features=config.get('additional_features', []))
    test_transform = DataTransform(is_train=False, use_volume=use_volume, additional_features=config.get('additional_features', []))
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
    pred_date = args.date
    if pred_date is None:
        pred_date = data_module.converter.end_date
    end_ts = data_module.converter.generate_timestamp(pred_date)
    start_ts = end_ts - model.window_size * 24 * 60 * 60
    data = data[data['Timestamp'] >= start_ts - 60 * 60]
    data = data.tail(model.window_size)  # 添加：裁剪到最后 window_size 行
    # if len(data) < model.window_size:
    #     raise ValueError(f"Data length {len(data)} < window_size = {model.window_size}, insufficient data")
    txt_file = init_dirs(args, pred_date)
    
    
    features = {}
    key_list = ['Timestamp', 'Open', 'High', 'Low', 'Close']
    if use_volume:
        key_list.append('Volume')
    scale_pred, shift_pred, t_scale, t_shift = None, None, None, None

    for key in key_list:
        tmp = list(data.get(key))
        if key not in ['Timestamp']:
            tmp = [np.log(x + 1e-8) for x in tmp]
        if normalize and key != 'Timestamp':
            scale = data_module.factors.get(key, {}).get('max', 1) - data_module.factors.get(key, {}).get('min', 0)
            shift = data_module.factors.get(key, {}).get('min', 0)
            tmp = [(x - shift) / scale if scale != 0 else x for x in tmp]
        else:
            tmp = [(x - data_module.min_ts) / data_module.scale_ts_diff if data_module.scale_ts_diff != 0 else 0 for x in tmp]  # 相对差 + [0,1]
        features[key] = torch.tensor(tmp).reshape(1, -1)
        if key == model.y_key:
            scale_pred = scale
            shift_pred = shift
        if key == 'Timestamp':
            t_scale = data_module.scale_ts_diff
            t_shift = data_module.min_ts
  
    x = torch.cat([features.get(x) for x in features.keys()], dim=0)
    close_idx = -2 if use_volume else -1
    today = float(x[close_idx, -1])
    with torch.no_grad():
        pred = float(model(x[None, ...].cuda()).cpu())
        if normalize:
            pred = np.exp(pred * scale_pred + shift_pred) - 1e-8
            pred = max(pred, 1e-6)
    today = np.exp(today * scale_pred + shift_pred) - 1e-8
    today = max(today, 1e-6)
    # pred_date 使用 min_ts 恢复
    pred_date = datetime.fromtimestamp(int(float(features['Timestamp'][0,-1]) * t_scale + t_shift))  # 修正：用 features['Timestamp'][-1]

    print('')
    print_and_write(txt_file, f'Prediction date: {pred_date}\nPrediction: {round(pred, 2)}\nToday value: {round(today, 2)}')

    b, s = buy_sell_smart(today, pred, 100, 100, risk=args.risk)
    if b < 100:
        tmp = round((100 - b), 2)
        print_and_write(txt_file, f'Smart trade: {tmp}% buy')
    if s < 100:
        tmp = round((100 - s), 2)
        print_and_write(txt_file, f'Smart trade: {tmp}% sell')

    b, s = buy_sell_vanilla(today, pred, 100, 100)
    if b < 100:
        assert b == 0
        print_and_write(txt_file, f'Vanilla trade: buy')
    elif s < 100:
        assert s == 0
        print_and_write(txt_file, f'Vanilla trade: sell')
    else:
        print_and_write(txt_file, f'Vanilla trade: -')

    

    


