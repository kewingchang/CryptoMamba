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
# ADDED FOR REVIN: import importlib for dynamic import
import importlib
# END ADDED FOR REVIN

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
    save_dict.pop('checkpoint_callback', None)
    with open(log_dir + '/hparams.yaml', 'w') as f:
        yaml.dump(save_dict, f)

def load_model(config, ckpt_path):
    arch_config = io_tools.load_config_from_yaml('configs/models/archs.yaml')
    model_arch = config.get('model')
    model_config_path = f'{ROOT}/configs/models/{arch_config.get(model_arch)}'
    model_config = io_tools.load_config_from_yaml(model_config_path)

    # MODIFIED FOR REVIN: disable normalize if use_revin
    if model_config.get('params', {}).get('use_revin', False):
        model_config['normalize'] = False
    # END MODIFIED FOR REVIN
    normalize = model_config.get('normalize', False)
    hyperparams = config.get('hyperparams')
    if hyperparams is not None:
        for key in hyperparams.keys():
            model_config.get('params')[key] = hyperparams.get(key)
    target = model_config.get('target')
    # MODIFIED FOR REVIN: dynamic import the target class
    module_path, class_name = target.rsplit('.', 1)
    module = importlib.import_module(module_path)
    target_class = getattr(module, class_name)
    model = target_class.load_from_checkpoint(ckpt_path, **model_config.get('params'))
    # END MODIFIED FOR REVIN
    model.cuda()
    model.eval()
    return model, normalize

if __name__ == "__main__":
    args = get_args()

    config = io_tools.load_config_from_yaml(f'{ROOT}/configs/training/{args.config}.yaml')

    data_config = io_tools.load_config_from_yaml(f"{ROOT}/configs/data_configs/{config.get('data_config')}.yaml")
    use_volume = args.use_volume

    if not use_volume:
        use_volume = config.get('use_volume')
    train_transform = DataTransform(is_train=False, use_volume=use_volume, additional_features=config.get('additional_features', []))
    val_transform = DataTransform(is_train=False, use_volume=use_volume, additional_features=config.get('additional_features', []))
    test_transform = DataTransform(is_train=False, use_volume=use_volume, additional_features=config.get('additional_features', []))

    model, normalize = load_model(config, args.ckpt_path)

    # ADDED FOR REVIN: set target_idx
    model.target_idx = test_transform.keys.index(model.y_key)
    # END ADDED FOR REVIN

    data = pd.read_csv(args.data_path, index_col=0)

    data_module = CMambaDataModule(
        data_config,
        train_transform=train_transform,
        val_transform=val_transform,
        test_transform=test_transform,
        batch_size=1,
        distributed_sampler=False,
        num_workers=1,
        normalize=normalize,
    )

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
    key_list = ['Timestamp', 'Open', 'High', 'Low', 'Close']
    if use_volume:
        key_list.append('Volume')

    for key in key_list:
        tmp = list(data.get(key))
        # MODIFIED FOR REVIN: skip normalization if use_revin
        if normalize and not model.use_revin:
            scale = data_module.factors.get(key).get('max') - data_module.factors.get(key).get('min')
            shift = data_module.factors.get(key).get('min')
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
    # END MODIFIED FOR REVIN

    x = torch.cat([features.get(x) for x in features.keys()], dim=0)

    close_idx = -2 if use_volume else -1
    # MODIFIED FOR REVIN: denormalize today value if not using RevIN
    today = float(x[close_idx, -1])
    if normalize and not model.use_revin:
        today = today * scale_pred + shift_pred
    # END MODIFIED FOR REVIN

    with torch.no_grad():
        # MODIFIED FOR REVIN: no manual denorm since RevIN handles it
        pred = float(model(x[None, ...].cuda()).cpu())
        # END MODIFIED FOR REVIN

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
