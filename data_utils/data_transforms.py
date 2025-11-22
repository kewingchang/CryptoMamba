# data_transforms.py
import torch
import numpy as np  # 新增 np for Beta 采样

class DataTransform:
    def __init__(self, is_train, use_volume=False, additional_features=[], mixup_alpha=0.5, mixup_prob=0.3):
        self.is_train = is_train
        # self.keys = ['Timestamp', 'Open', 'High', 'Low', 'Close']
        self.keys = ['Open', 'High', 'Low', 'Close']
        if use_volume:
            self.keys.append('Volume')
        self.keys += additional_features
        print(self.keys)
        self.mixup_alpha = mixup_alpha  # 新增：Beta 分布 alpha (1.0 默认均匀混合)
        self.mixup_prob = mixup_prob    # 新增：应用 mixup 概率 (0.5 默认)

    def __call__(self, window, other_window=None):
        output = {}
        if 'Timestamp_orig' in window.keys():
            self.keys.append('Timestamp_orig')
        features_list = []
        for key in self.keys:
            data = torch.tensor(window.get(key).tolist())
            output[key] = data[-1]
            output[f'{key}_old'] = data[-2]
            if key == 'Timestamp_orig':
                continue
            features_list.append(data[:-1].reshape(1, -1))
        features = torch.cat(features_list, 0)
        output['features'] = features

        # 新增：如果 is_train 且 other_window，不需随机（已在 Dataset 判断），直接混合
        if self.is_train and other_window is not None:
            lambda_ = np.random.beta(self.mixup_alpha, self.mixup_alpha)  # 抽样 lambda
            lambda_ = torch.tensor(lambda_).to(features.device)  # 转 tensor

            # 混合 features (时序序列)
            other_features = torch.cat([torch.tensor(other_window.get(key).tolist())[:-1].reshape(1, -1) for key in self.keys if key != 'Timestamp_orig'], 0)
            output['features'] = lambda_ * features + (1 - lambda_) * other_features

            # 混合 label (Close 和 Close_old 等输出)
            for key in self.keys:
                if key == 'Timestamp_orig':
                    continue
                other_data = torch.tensor(other_window.get(key).tolist())
                output[key] = lambda_ * output[key] + (1 - lambda_) * other_data[-1]
                output[f'{key}_old'] = lambda_ * output[f'{key}_old'] + (1 - lambda_) * other_data[-2]

        # === 新增：始终输出 Timestamp（仅用于绘图、窗口筛选，不作为特征输入模型）===
        # 即使 self.keys 不包含 Timestamp，也把原始 window 的 Timestamp 序列/最后一个值传出去
        if 'Timestamp' in window:
            # 输出整个窗口的 Timestamp 序列（evaluation.py 需要用于画图）
            output['Timestamp_seq'] = torch.tensor(window['Timestamp'].tolist())        # 全序列
            output['Timestamp'] = torch.tensor(window['Timestamp'].tolist())[-1]       # 最后一个（兼容旧代码）
        elif 'Timestamp_orig' in window:
            # 兼容旧代码里可能有的 Timestamp_orig
            output['Timestamp_seq'] = torch.tensor(window['Timestamp_orig'].tolist())
            output['Timestamp'] = torch.tensor(window['Timestamp_orig'].tolist())[-1]
        else:
            # 极端情况防报错
            output['Timestamp'] = torch.tensor(0.0)
            output['Timestamp_seq'] = torch.tensor([0.0])

        return output

    def set_initial_seed(self, seed):
        self.rng.seed(seed)