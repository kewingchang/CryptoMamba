# dataset.py
import os
import torch
import time
import random
import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path
from utils import io_tools
from datetime import datetime
from utils.io_tools import load_config_from_yaml

    
class CMambaDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        data,
        split,
        window_size,
        transform,
    ):

        self.data = data
        # self.split = split
        self.transform = transform
        self.window_size = window_size
        self.data_module = None  # 添加
        self.split = split  # 新增，判断 train

        # === 新增检查 ===
        # 检查数据是否连续。如果不连续，iloc 切片训练就是错误的。
        if len(self.data) > 1:
            # 假设数据主要是日线，间隔86400。如果是小时线需调整。
            # 这里简单判断：如果最大间隔超过了最小间隔的1.1倍，说明有断档
            timestamps = self.data['Timestamp'].values
            diffs = timestamps[1:] - timestamps[:-1]
            if len(diffs) > 0:
                min_diff = np.min(diffs)
                max_diff = np.max(diffs)
                if max_diff > min_diff * 1.05: # 允许5%误差
                    print(f"Warning: {split} split data is not continuous! Max gap: {max_diff}, Min gap: {min_diff}")
                    # 在训练集严格报错
                    if split == 'train':
                         raise ValueError(f"Training data contains time gaps (missing days). Fix data before training.")
        # ===============
            
        print('{} data points loaded as {} split.'.format(len(self), split))

    def __len__(self):
        return max(0, len(self.data) - self.window_size - 1)

    def __getitem__(self, i: int):
        window = self.data.iloc[i: i + self.window_size + 1]
        other_window = None
        if self.split == 'train' and random.random() < 0.5:  # 假设 prob=0.5，可从 transform.mixup_prob 读
            other_i = random.randint(0, len(self) - 1)  # 随机另一个索引
            other_window = self.data.iloc[other_i: other_i + self.window_size + 1]
        sample = self.transform(window, other_window)  # 传 other_window
        return sample
    
    def set_data_module(self, data_module):  # 添加
            self.data_module = data_module

    
class DataConverter:
    def __init__(self, config) -> None:
        self.config = config
        self.root = config.get('root')
        self.jumps = config.get('jumps')
        self.date_format = config.get('date_format', "%Y-%m-%d")
        self.additional_features = config.get('additional_features', [])
        self.end_date = config.get('end_date')
        self.data_path = config.get('data_path')
        self.start_date = config.get('start_date')
        self.folder_name = f'{self.start_date}_{self.end_date}_{self.jumps}'
        self.file_path = f'{self.root}/{self.folder_name}'

    
    def process_data(self):
        data, start, stop = self.load_data()
        new_df = {}
        for key in ['Timestamp', 'High', 'Low', 'Open', 'Close', 'Volume']:
            new_df[key] = []
        for key in self.additional_features:
            new_df[key] = []
        for i in tqdm(range(start, stop - self.jumps // 60 + 1, self.jumps)):
            high, low, open, close, vol = self.merge_data(data, i, self.jumps)
            additional_features = self.merge_additional(data, i, self.jumps)

            if high is None:
           # === 修改开始：不仅 continue，还要报错或记录 ===
                # 方案 A: 严格模式 - 直接报错停止训练
                missing_date = datetime.fromtimestamp(i).strftime("%Y-%m-%d")
                raise ValueError(f"[Data Error] Missing data for timestamp {i} ({missing_date}). "
                                 f"Cannot train time-series model with gaps. Please fix the CSV.")             
            
            new_df.get('Timestamp').append(i)
            new_df.get('High').append(high)
            new_df.get('Low').append(low)
            new_df.get('Open').append(open)
            new_df.get('Close').append(close)
            new_df.get('Volume').append(vol)
            for key in additional_features.keys():
                new_df.get(key).append(additional_features.get(key))

        df = pd.DataFrame(new_df)
        # 双重检查：确保整个 dataframe 的时间戳是连续的
        if len(df) > 1:
            timestamps = df['Timestamp'].values
            diffs = np.diff(timestamps)
            # 检查是否有任何间隔不等于 jumps (通常是86400)
            if not np.allclose(diffs, self.jumps, atol=60):
                 raise ValueError(f"[Data Error] Generated DataFrame has time gaps! "
                                  f"Ensure source CSV covers every single day from {self.start_date} to {self.end_date}.")

        return df

    def get_data(self):
        tmp = '----'
        data_path = f'{self.file_path}/{tmp}.csv'
        yaml_path = f'{self.file_path}/config.pkl'
        if os.path.isfile(data_path.replace(tmp, 'train')):
            train = pd.read_csv(data_path.replace(tmp, 'train'), index_col=0)
            val = pd.read_csv(data_path.replace(tmp, 'val'), index_col=0)
            test = pd.read_csv(data_path.replace(tmp, 'test'), index_col=0)
            return train, val, test
        
        df = self.process_data()

        train, val, test = self.split(df)

        if not os.path.isdir(self.file_path):
            os.mkdir(self.file_path)

        train.to_csv(data_path.replace('----', 'train'))
        val.to_csv(data_path.replace('----', 'val'))
        test.to_csv(data_path.replace('----', 'test'))
        io_tools.save_yaml(self.config, yaml_path)
        return train, val, test
    

    def split(self, data):
        if self.config.get('train_ratio') is not None:
            total = len(data)
            n_train = int(self.train_ratio * total)
            n_test = int(self.test_ratio * total)
            total = list(range(total))
            random.shuffle(total)
        
            train = sorted(total[: n_train])
            val = sorted(total[n_train: -n_test])
            test = sorted(total[-n_test: ])
            train = data.iloc[train]
            val = data.iloc[val]
            test = data.iloc[test]
        else:
            tmp_dict = {}
            for key in ['train', 'val', 'test']:
                start, stop = self.config.get(f'{key}_interval')
                start = self.generate_timestamp(start)
                stop = self.generate_timestamp(stop)
                tmp = data[data['Timestamp'] >= start].reset_index(drop=True)
                tmp = tmp[tmp['Timestamp'] < stop].reset_index(drop=True)
                tmp_dict[key] = tmp
            train = tmp_dict.get('train')
            val = tmp_dict.get('val')
            test = tmp_dict.get('test')
        return train, val, test


    def load_data(self):
        df = pd.read_csv(self.data_path)
        if 'Timestamp' not in df.keys():
            dates = df.get('Date').to_list()
            df['Timestamp'] = [self.generate_timestamp(x) for x in dates]
        if self.start_date is None:
            self.start_date = self.convert_timestamp(min(list(df.get('Timestamp')))).strftime(self.date_format)
            # raise ValueError(self.start_date)
        if self.end_date is None:
            self.end_date = self.convert_timestamp(max(list(df.get('Timestamp'))) + self.jumps).strftime(self.date_format)
        start = self.generate_timestamp(self.start_date)
        stop = self.generate_timestamp(self.end_date)
        df = df[df['Timestamp'] >= start].reset_index(drop=True)
        df = df[df['Timestamp'] < stop].reset_index(drop=True)
        final_day = self.generate_timestamp(self.end_date)
        return df, start, final_day
    
    def merge_additional(self, data, start, jump):
        tmp = data[data['Timestamp'] >= start].reset_index(drop=True)
        tmp = tmp[tmp['Timestamp'] < start + jump].reset_index(drop=True)
        if len(tmp) == 0:
            return None, None, None, None, None
        row = tmp.iloc[-1]
        results = {}
        for key in self.additional_features:
            results[key] = float(row.get(key))
        return results
    
    def generate_timestamp(self, date):
        return int(time.mktime(datetime.strptime(date, self.date_format).timetuple()))
    
    @staticmethod
    def merge_data(data, start, jump):
        tmp = data[data['Timestamp'] >= start].reset_index(drop=True)
        tmp = tmp[tmp['Timestamp'] < start + jump].reset_index(drop=True)
        if len(tmp) == 0:
            return None, None, None, None, None
        _, high, low, open, close, _ = DataConverter.get_row_values(tmp.iloc[0])
        vol = 0
        for row in tmp.iterrows():
            _, h, l, _, close, v = DataConverter.get_row_values(row[1])
            high = max(high, h)
            low = min(low, l)
            vol += v
        return high, low, open, close, vol
    
    @staticmethod
    def get_row_values(row):
        ts = int(row.get('Timestamp'))
        high = float(row.get('High'))
        low = float(row.get('Low'))
        open = float(row.get('Open'))
        close = float(row.get('Close'))
        vol = float(row.get('Volume'))
        return ts, high, low, open, close, vol

    @staticmethod
    def convert_timestamp(timestamp):
        return datetime.fromtimestamp(timestamp)
