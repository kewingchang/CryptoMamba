import os
import yaml
from collections import Counter

def scan_yaml_duplicates():
    # 获取当前工作目录
    current_dir = os.getcwd()
    
    # 初始化计数器
    # 这里的计数器用于统计每个 feature 在多少个文件中出现过
    feature_counter = Counter()
    
    # 统计扫描了多少个文件
    files_scanned = 0
    
    print(f"正在扫描目录: {current_dir} ...\n")

    # 遍历当前目录下的所有文件
    for filename in os.listdir(current_dir):
        # 只处理 yaml 或 yml 结尾的文件
        if filename.endswith(('.yaml', '.yml')):
            filepath = os.path.join(current_dir, filename)
            
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = yaml.safe_load(f)
                    
                    # 检查文件内容是否为空，且包含 fixed_features 字段
                    if content and 'fixed_features' in content:
                        features = content['fixed_features']
                        
                        # 确保 features 是一个列表
                        if isinstance(features, list):
                            # 使用 set() 去重：确保同一个文件中如果写了两次同一个 feature，
                            # 只会被计算为“该文件包含此 feature”，避免虚高统计。
                            # 如果你希望统计绝对出现的总次数（哪怕在一个文件中重复），请去掉 set()
                            unique_features_in_file = set(features)
                            feature_counter.update(unique_features_in_file)
                            files_scanned += 1
                        else:
                            print(f"[警告] {filename}: fixed_features 格式不是列表，已跳过。")
                    else:
                        # 这是一个可选项，如果有些yaml没有这个字段，可以忽略
                        # print(f"[跳过] {filename}: 未找到 fixed_features 字段。")
                        pass
                        
            except Exception as e:
                print(f"[错误] 读取 {filename} 失败: {e}")

    print("-" * 30)
    print(f"扫描完成。共处理了 {files_scanned} 个包含有效数据的 YAML 文件。")
    print("-" * 30)
    print("【跨文件重复字段统计结果】")
    print(f"{'字段名':<30} | {'出现次数':<10}")
    print("-" * 45)

    # 筛选出出现次数大于 1 的字段（即重复的）
    # .most_common() 会按照出现次数从高到低排序
    duplicates_found = False
    for feature, count in feature_counter.most_common():
        if count > 1:
            print(f"{feature:<30} | {count:<10}")
            # print(f"{feature}")
            duplicates_found = True
    
    if not duplicates_found:
        print("未发现任何跨文件重复的字段。")

if __name__ == "__main__":
    scan_yaml_duplicates()