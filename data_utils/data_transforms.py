# data_transforms.py
import torch
import numpy as np

class DataTransform:
    def __init__(self, is_train, use_volume=False, additional_features=[], mixup_alpha=0.5, mixup_prob=0.3):
        self.is_train = is_train
        
        # 从特征列表中移除 'Close'
        # 现在的特征: ['Open', 'High', 'Low', 'Volume'?, 'log_return'...]
        # self.keys = ['Open', 'High', 'Close']
        self.keys = []
                
        if use_volume:
            self.keys.append('Volume')
        
        # additional_features 里包含了 'log_return'
        self.keys += additional_features
        
        print(f"Model Input Features: {self.keys}") # 打印确认一下不含 Close
        
        self.mixup_alpha = mixup_alpha
        self.mixup_prob = mixup_prob

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

        # [修改点 2]: 强制单独提取 Close 价格
        # 这一步是为了 base_module.py 的“价格还原”逻辑能找到 batch['Close']
        # 它不会进入 output['features']，所以模型训练时看不到它
        close_data = torch.tensor(window.get('Close').tolist())
        output['Close'] = close_data[-1]
        output['Close_old'] = close_data[-2]

        # Mixup 逻辑 (保持不变)
        if self.is_train and other_window is not None:
            lambda_ = np.random.beta(self.mixup_alpha, self.mixup_alpha)
            lambda_ = torch.tensor(lambda_).to(features.device)

            # 混合 features
            # 注意：other_window 的特征提取也要遵循 self.keys (不含 Close)
            other_features_list = []
            for key in self.keys:
                if key != 'Timestamp_orig':
                    val = other_window.get(key)
                    if val is None:
                         d = torch.zeros(len(window))
                    else:
                         d = torch.tensor(val.tolist())
                    other_features_list.append(d[:-1].reshape(1, -1))
            
            other_features = torch.cat(other_features_list, 0)
            output['features'] = lambda_ * features + (1 - lambda_) * other_features

            # 混合 label (log_return 等)
            for key in self.keys:
                if key == 'Timestamp_orig': continue
                # 注意：这里只混合了 self.keys 里的 label
                # Close 的混合需要单独处理
                other_val = other_window.get(key)
                if other_val is not None:
                    other_data = torch.tensor(other_val.tolist())
                    output[key] = lambda_ * output[key] + (1 - lambda_) * other_data[-1]
                    output[f'{key}_old'] = lambda_ * output[f'{key}_old'] + (1 - lambda_) * other_data[-2]
            
            # [修改点 3]: 单独混合 Close (为了保持一致性)
            other_close = torch.tensor(other_window.get('Close').tolist())
            output['Close'] = lambda_ * output['Close'] + (1 - lambda_) * other_close[-1]
            output['Close_old'] = lambda_ * output['Close_old'] + (1 - lambda_) * other_close[-2]

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