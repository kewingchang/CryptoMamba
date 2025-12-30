import pandas as pd
import os
import shutil
import subprocess
import json
import yaml
from argparse import ArgumentParser


# train with some features
# 修改点：增加 skip_features 参数
def train_features(to_train_features, fixed_features_num, skip_features):
    # 1. 删除目录: "data"中的一个以数字开头的子目录及该子目录中的所有文件
    data_dir = "data"
    if os.path.exists(data_dir):
        for subdir in os.listdir(data_dir):
            if subdir[0].isdigit():
                subdir_path = os.path.join(data_dir, subdir)
                if os.path.isdir(subdir_path):
                    shutil.rmtree(subdir_path)
                    break  # 只删除一个匹配的子目录
    # 删除目录: 'log', 'result'及其目录下的所有文件（如果目录存在的话）
    for dir_name in ['logs', 'Result']:
        if os.path.exists(dir_name):
            shutil.rmtree(dir_name)

    # 2. 读取文件"configs/data_configs/mode_1.yaml"
    mode_1_path = "configs/data_configs/mode_1.yaml"
    with open(mode_1_path, 'r') as f:
        mode_1_data = yaml.safe_load(f)

    # 3. 将to_train_features赋值给additional_features并保存
    mode_1_data['additional_features'] = to_train_features
    
    # --- 修改需求 3: 将 y (skip_features) 赋值给 mode_1.yaml 中的 skip_revin ---
    mode_1_data['skip_revin'] = skip_features
    
    with open(mode_1_path, 'w') as f:
        yaml.dump(mode_1_data, f)

    # 4. 读取文件"configs/training/cmamba_v.yaml"
    cmamba_v_path = "configs/training/cmamba_v.yaml"
    with open(cmamba_v_path, 'r') as f:
        cmamba_v_data = yaml.safe_load(f)

    # 5. 将to_train_features赋值给additional_features并保存
    cmamba_v_data['additional_features'] = to_train_features
    
    # --- 修改需求 4: 将 y (skip_features) 赋值给 cmamba_v.yaml 中的 skip_revin ---
    cmamba_v_data['skip_revin'] = skip_features
    
    with open(cmamba_v_path, 'w') as f:
        yaml.dump(cmamba_v_data, f)

    # 6.删除log.txt如果存在
    log_file = 'log.txt'
    if os.path.exists(log_file):
        os.remove(log_file)
    # 新建log.txt并写入features_str
    with open(log_file, 'w') as f:
        # features_str = to_train_features[0]
        features_str = "+".join(to_train_features)
        features_str = "<<<<<<<<<<<<>>>>>>>>>>>>\n"+features_str+"\n<<<<<<<<<<<<>>>>>>>>>>>>\n"
        f.write(features_str)

    # 9. 读取文件"configs/models/CryptoMamba/v2.yaml"
    v2_path = "configs/models/CryptoMamba/v2.yaml"
    with open(v2_path, 'r') as f:
        v2_data = yaml.safe_load(f)

    # # 10. 修改num_features
    v2_data['params']['num_features'] = fixed_features_num + len(to_train_features)

    # # 8. 修改hidden_dims的第一个数字
    if v2_data['params']['hidden_dims']:
        v2_data['params']['hidden_dims'][0] = fixed_features_num + len(to_train_features)

    # 9. 更新并保存到v2.yaml
    with open(v2_path, 'w') as f:
        yaml.dump(v2_data, f)

    # 10. 运行命令
    subprocess.run("python scripts/training-earlystop.py --config cmamba_v --save_checkpoints >> ./log.txt 2>&1", shell=True, check=True)
    # 11. 运行命令
    # subprocess.run("python scripts/evaluation.py --config cmamba_v --ckpt_path ./logs/CMamba/version_0/checkpoints/epoch*.ckpt >> ./log.txt 2>&1", shell=True, check=True)
    # 12. 运行命令
    subprocess.run("python utils/save_to_drive.py", shell=True, check=True)

if __name__ == "__main__":

    # 解析命令行参数
    parser = ArgumentParser(description="Feature Selection for ETH Dataset")
    parser.add_argument('--filename', type=str, required=True, help="Path to the CSV file, e.g., ETH_Dataset_daily.csv")
    # --- 修改需求 1: 新增命令行参数 --skip_revin ---
    parser.add_argument('--skip_revin', type=str, required=False, default=None, help="Path to the JSON file containing features to skip revin (optional)")
    
    args = parser.parse_args()

    fixed_features = []
    num_fixed_features = len(fixed_features)

    # 加载数据
    print("Loading data from:", args.filename)    
    # json配置文件和csv数据文件:train3f.json, step1-org.csv
    json_name = args.filename
    
    # 读取主json文件
    with open(json_name, 'r') as f:
        data = json.load(f) # 这是一个列表的列表，对应 combo

    # --- 修改需求 2: 读取 --skip_revin 文件并准备数据 ---
    skip_data = []
    if args.skip_revin:
        print("Loading skip_revin features from:", args.skip_revin)
        with open(args.skip_revin, 'r') as f:
            skip_data = json.load(f) # 对应 y
            
        # 简单检查长度是否一致，防止IndexError
        if len(skip_data) != len(data):
            print(f"Warning: Length of skip_revin data ({len(skip_data)}) does not match length of feature data ({len(data)}).")
    else:
        # 如果 --skip_revin 为空，则 y 为空 list，构造一个与 data 长度相同的由空列表组成的列表
        skip_data = [[] for _ in range(len(data))]

    # 同步循环
    # zip(data, skip_data) 会把两个列表对应位置的元素打包在一起
    for i, (combo, y) in enumerate(zip(data, skip_data)):
        print(f"...Start process index {i}: {combo}, skip_revin: {y}")
        # 2.7 调用train_features()，传入 combo 和 y
        train_features(combo, num_fixed_features, y)
