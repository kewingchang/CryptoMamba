import torch
import numpy as np

class DataTransform:
    def __init__(self, is_train, use_volume=False, additional_features=[]):
        self.is_train = is_train
        self.keys = ['Timestamp', 'Open', 'High', 'Low', 'Close']
        if use_volume:
            self.keys.append('Volume')
        self.keys += additional_features
        print(self.keys)

    def __call__(self, window):
        data_list = []
        output = {}
        if 'Timestamp_orig' in window.keys():
            self.keys.append('Timestamp_orig')
        for key in self.keys:
            data = torch.tensor(window.get(key).tolist())
            if key not in ['Timestamp', 'Timestamp_orig']:  # 排除 Timestamp 和 Timestamp_orig
                data = torch.log(data + 1e-8)  # Log 变换
            output[key] = data[-1]
            output[f'{key}_old'] = data[-2]
            if key == 'Timestamp_orig':
                continue
            data_list.append(data[:-1].reshape(1, -1))
        features = torch.cat(data_list, 0)
        output['features'] = features
        return output

    def set_initial_seed(self, seed):
        self.rng.seed(seed)
