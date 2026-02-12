import os
import yaml
import argparse
import sys

def remove_feature_from_files(todel):
    current_dir = os.getcwd()
    print(f"正在扫描目录: {current_dir}")
    print(f"准备删除字段: {todel}")
    print("-" * 30)

    modified_count = 0

    # 遍历当前目录下的所有文件
    for filename in os.listdir(current_dir):
        # 只处理 yaml/yml 文件
        if filename.endswith(('.yaml', '.yml')):
            filepath = os.path.join(current_dir, filename)
            
            try:
                # 1. 读取文件
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = yaml.safe_load(f)

                # 2. 检查内容并修改
                file_modified = False
                
                # 确保 content 不为空且包含 fixed_features 字段
                if content and 'fixed_features' in content:
                    features = content['fixed_features']
                    
                    # 确保 fixed_features 是一个列表
                    if isinstance(features, list):
                        if todel in features:
                            # 执行删除操作
                            features.remove(todel)
                            content['fixed_features'] = features
                            file_modified = True
                            print(f"[修改] {filename}: 发现并已删除 '{todel}'")
                    else:
                        print(f"[警告] {filename}: fixed_features 不是列表格式，已跳过。")

                # 3. 如果文件被修改过，则保存
                if file_modified:
                    with open(filepath, 'w', encoding='utf-8') as f:
                        # sort_keys=False: 保持原文件字段顺序
                        # default_flow_style=None: 让列表保持块状格式 (- item) 而不是行内 ([item])
                        yaml.dump(content, f, sort_keys=False, default_flow_style=None, allow_unicode=True)
                    modified_count += 1

            except Exception as e:
                print(f"[错误] 处理文件 {filename} 时出错: {e}")

    print("-" * 30)
    if modified_count == 0:
        print(f"扫描完成。没有在任何文件中找到 '{todel}'。")
    else:
        print(f"扫描完成。共修改了 {modified_count} 个文件。")

if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="从当前目录的所有YAML文件中删除指定的feature。")
    parser.add_argument('--todel', type=str, required=True, help='要删除的字段名称')

    args = parser.parse_args()

    remove_feature_from_files(args.todel)