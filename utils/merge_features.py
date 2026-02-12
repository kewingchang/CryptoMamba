import os
import yaml
import argparse
import re
import copy
import sys

def merge_features(feature_dir, model_path):
    # 1. 读取基础 Model 文件
    try:
        with open(model_path, 'r', encoding='utf-8') as f:
            base_model_data = yaml.safe_load(f)
            # 如果模型文件是空的，初始化为空字典
            if base_model_data is None:
                base_model_data = {}
    except Exception as e:
        print(f"[错误] 无法读取 Model 文件: {e}")
        sys.exit(1)

    # 检查 feature_dir 是否存在
    if not os.path.exists(feature_dir):
        print(f"[错误] 目录不存在: {feature_dir}")
        sys.exit(1)

    print(f"正在扫描目录: {feature_dir}")
    print(f"基础模型: {model_path}")
    print("-" * 30)

    count = 0
    
    # 2. 遍历 feature_dir 中的文件
    for filename in os.listdir(feature_dir):
        # 正则匹配：匹配 n+数字-features.yaml，并提取 "n+数字" 部分
        # 这里的正则 (n\d+) 会匹配 n1, n100 等，并将其作为 group(1)
        match = re.match(r"(n\d+)-features\.yaml$", filename)
        
        if match:
            file_id = match.group(1) # 例如: "n1"
            feature_filepath = os.path.join(feature_dir, filename)

            try:
                # 读取 feature 文件
                with open(feature_filepath, 'r', encoding='utf-8') as f:
                    feature_data = yaml.safe_load(f)

                if feature_data and 'fixed_features' in feature_data:
                    extracted_features = feature_data['fixed_features']

                    # 3. 合并数据
                    # 使用 deepcopy 确保每次循环使用的是基础模型的干净副本，互不影响
                    new_model_data = copy.deepcopy(base_model_data)
                    
                    # 将提取出的 features 放入新模型的 fixed_features 字段
                    # 逻辑说明：如果基础模型里没有 fixed_features，则创建；如果有，则覆盖。
                    new_model_data['fixed_features'] = extracted_features

                    # 4. 生成新文件名并保存
                    output_filename = f"model-{file_id}.yaml"
                    
                    with open(output_filename, 'w', encoding='utf-8') as out_f:
                        # sort_keys=False 保证输出时字段顺序不乱（保持原model顺序）
                        # default_flow_style=False 保证列表以块状显示（- item），而不是行内显示 [item]
                        yaml.dump(new_model_data, out_f, sort_keys=False, default_flow_style=None, allow_unicode=True)
                    
                    print(f"[成功] 生成: {output_filename} (来自 {filename})")
                    count += 1
                else:
                    print(f"[跳过] {filename}: 未找到 fixed_features 字段")

            except Exception as e:
                print(f"[警告] 处理文件 {filename} 时出错: {e}")

    print("-" * 30)
    print(f"处理完成，共生成了 {count} 个新模型文件。")

if __name__ == "__main__":
    # 设置命令行参数解析
    parser = argparse.ArgumentParser(description="将特征文件合并到基础模型YAML中。")
    parser.add_argument('--feature_dir', type=str, required=True, help='包含特征YAML文件的目录路径')
    parser.add_argument('--model', type=str, required=True, help='基础模型YAML文件的路径')

    args = parser.parse_args()

    merge_features(args.feature_dir, args.model)