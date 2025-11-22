import pandas as pd
import os
import shutil
import subprocess
import json
import yaml
from argparse import ArgumentParser


# train with some features
def train_features(to_train_features, fixed_features_num):
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
    with open(mode_1_path, 'w') as f:
        yaml.dump(mode_1_data, f)

    # 4. 读取文件"configs/training/cmamba_v.yaml"
    cmamba_v_path = "configs/training/cmamba_v.yaml"
    with open(cmamba_v_path, 'r') as f:
        cmamba_v_data = yaml.safe_load(f)

    # 5. 将to_train_features赋值给additional_features并保存
    cmamba_v_data['additional_features'] = to_train_features
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
    subprocess.run("python scripts/evaluation.py --config cmamba_v --ckpt_path ./logs/CMamba/version_0/checkpoints/epoch*.ckpt >> ./log.txt 2>&1", shell=True, check=True)
    # 12. 运行命令
    subprocess.run("python utils/save_to_drive.py", shell=True, check=True)

if __name__ == "__main__":

    # 解析命令行参数
    parser = ArgumentParser(description="Feature Selection for ETH Dataset")
    parser.add_argument('--filename', type=str, required=True, help="Path to the CSV file, e.g., ETH_Dataset_daily.csv")
    args = parser.parse_args()

    fixed_features = ['Open', 'High', 'Low', 'Close']
    num_fixed_features = len(fixed_features)

    # 加载数据
    print("Loading data from:", args.filename)    
    # json配置文件和csv数据文件:train3f.json, step1-org.csv
    json_name = args.filename
    # 读取json文件
    with open(json_name, 'r') as f:
        data = json.load(f)
    # for 每一行 in json
    for combo in data:
        print(f"...Start process: {combo}")
        # 2.7 调用train_features()
        train_features(combo, num_fixed_features)
