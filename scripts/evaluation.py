import os, sys, pathlib
sys.path.insert(0, os.path.dirname(pathlib.Path(__file__).parent.absolute()))

import yaml
import torch
import matplotlib
import numpy as np
from utils import io_tools
from datetime import datetime
import pytorch_lightning as pl
import matplotlib.ticker as ticker
from argparse import ArgumentParser
from pl_modules.data_module import CMambaDataModule
from data_utils.data_transforms import DataTransform
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings
# ADDED FOR REVIN: import importlib for dynamic import
import importlib
# END ADDED FOR REVIN
import seaborn as sns

warnings.simplefilter(action='ignore', category=FutureWarning)
sns.set_theme(style='whitegrid', context='paper', font_scale=3)
palette = sns.color_palette('muted')

ROOT = io_tools.get_root(__file__, num_returns=2)

def get_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--logdir",
        type=str,
        help="Logging directory.",
    )
    parser.add_argument(
        "--accelerator",
        type=str,
        default='gpu',
        help="The type of accelerator.",
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
        "--expname",
        type=str,
        default='Cmamba',
        help="Experiment name. Reconstructions will be saved under this folder.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default='cmamba_nv',
        help="Path to config file.",
    )
    parser.add_argument(
        "--logger_type",
        default='tb',
        type=str,
        help="Path to config file.",
    )
    parser.add_argument(
        '--use_volume', 
        default=False,   
        action='store_true',          
    )
    parser.add_argument(
        "--ckpt_path",
        required=True,
        type=str,
        help="Path to config file.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of parallel workers.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="batch_size",
    )
    args = parser.parse_args()
    return args

def print_and_write(file, txt, add_new_line=True):
    print(txt)
    if add_new_line:
        file.write(f'{txt}\n')
    else:
        file.write(txt)

def save_all_hparams(log_dir, args):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

# MODIFIED FOR REVIN: update load_model to handle use_revin and dynamic import
def load_model(config, ckpt_path):
    arch_config = io_tools.load_config_from_yaml('configs/models/archs.yaml')
    model_arch = config.get('model')
    model_config_path = f'{ROOT}/configs/models/{arch_config.get(model_arch)}'
    model_config = io_tools.load_config_from_yaml(model_config_path)

    if model_config.get('params', {}).get('use_revin', False):
        model_config['normalize'] = False
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

def init_dirs(args, name):
    path = f'{ROOT}/Evaluations/{args.config}/'
    if not os.path.isdir(path):
        os.makedirs(path)
    txt_file = open(f'{path}/{name}.txt', 'w')
    plot_path = f'{path}/{name}.png'
    return txt_file, plot_path

def run_model(model, dataloader, factors):
    timestamps = []
    targets = []
    preds = []
    for batch in dataloader:
        x = batch['features']
        y = batch[model.y_key]
        ts = batch['Timestamp']
        # MODIFIED FOR REVIN: skip denormalization if use_revin
        if factors is not None and not model.use_revin:
            scale = factors.get(model.y_key).get('max') - factors.get(model.y_key).get('min')
            shift = factors.get(model.y_key).get('min')
            y = y * scale + shift
        # END MODIFIED FOR REVIN
        with torch.no_grad():
            pred = model(x.cuda()).cpu().numpy()
        timestamps += list(ts.numpy())
        targets += list(y.numpy())
        preds += list(pred)
    targets = np.array(targets)
    preds = np.array(preds)
    mse = np.mean((targets - preds)**2)
    l1 = np.mean(np.abs(targets - preds))
    mape = np.mean(np.abs((targets - preds) / (targets + 1e-10)))
    return timestamps, targets, preds, mse, mape, l1

if __name__ == "__main__":
    args = get_args()
    pl.seed_everything(args.seed)

    config = io_tools.load_config_from_yaml(f'{ROOT}/configs/training/{args.config}.yaml')

    data_config = io_tools.load_config_from_yaml(f"{ROOT}/configs/data_configs/{config.get('data_config')}.yaml")
    use_volume = args.use_volume

    if not use_volume:
        use_volume = config.get('use_volume')
    train_transform = DataTransform(is_train=True, use_volume=use_volume, additional_features=config.get('additional_features', []))
    val_transform = DataTransform(is_train=False, use_volume=use_volume, additional_features=config.get('additional_features', []))
    test_transform = DataTransform(is_train=False, use_volume=use_volume, additional_features=config.get('additional_features', []))

    model, normalize = load_model(config, args.ckpt_path)

    # ADDED FOR REVIN: set target_idx
    model.target_idx = train_transform.keys.index(model.y_key)
    # END ADDED FOR REVIN

    data_module = CMambaDataModule(
        data_config,
        train_transform=train_transform,
        val_transform=val_transform,
        test_transform=test_transform,
        batch_size=args.batch_size,
        distributed_sampler=False,
        num_workers=args.num_workers,
        normalize=normalize,
    )

    dataloader_list = [data_module.train_dataloader(), data_module.val_dataloader(), data_module.test_dataloader()]
    titles = ['Train', 'Val', 'Test']
    colors = ['red', 'green', 'magenta']

    factors = None
    if normalize:
        factors = data_module.factors
    all_targets = []
    all_timestamps = []

    f, plot_path = init_dirs(args, config.get('name', args.expname))

    plt.figure(figsize=(20, 10))
    print_format = '{:^7} {:^15} {:^10} {:^7} {:^10}'
    txt = print_format.format('Split', 'MSE', 'RMSE', 'MAPE', 'MAE')
    print_and_write(f, txt)
    for key, dataloader, c in zip(titles, dataloader_list, colors):
        timestamps, targets, preds, mse, mape, l1 = run_model(model, dataloader, factors)
        all_timestamps += timestamps
        all_targets += list(targets)
        txt = print_format.format(key, round(mse, 3), round(np.sqrt(mse), 3), round(mape, 5), round(l1, 3))
        print_and_write(f, txt)
        if key == 'Test':
            prev_targets = np.roll(targets, 1)[1:]
            prev_preds = np.roll(preds, 1)[1:]
            actual_directions = targets[1:] > prev_targets
            pred_directions = preds[1:] > prev_preds
            direction_acc = np.mean(actual_directions == pred_directions) * 100
            print(f"Test Direction Accuracy: {direction_acc:.2f}%")
            actual_returns = (targets[1:] - prev_targets) / prev_targets
            trade_pnl = np.where(pred_directions, actual_returns, -actual_returns)
            win_rate = np.mean(trade_pnl > 0) * 100
            losses = trade_pnl[trade_pnl < 0]
            avg_loss = np.mean(losses) * 100 if len(losses) > 0 else 0
            max_loss = np.min(trade_pnl) * 100
            print(f"Test Win Rate: {win_rate:.2f}%")
            print(f"Test Average Trade Loss: {avg_loss:.2f}%")
            print(f"Test Maximum Trade Loss: {max_loss:.2f}%")
        sns.lineplot(x=timestamps, y=preds, color=c, linewidth=2.5, label=key)

    sns.lineplot(x=all_timestamps, y=all_targets, color='blue', zorder=0, linewidth=2.5, label='Target')
    plt.legend()
    plt.ylabel('Price ($)')
    plt.xlim([all_timestamps[0], all_timestamps[-1]])
    plt.xticks(rotation=30)
    ax = plt.gca()
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{:,.0f}K'.format(x/1000)))
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    f.close()
