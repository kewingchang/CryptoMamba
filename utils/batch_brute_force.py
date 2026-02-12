import os
import shutil
import subprocess
import argparse
import sys

def run_brute_batch(model_dir, start_index):
    # 1. 定义配置列表
    min_features_list = [1, 5, 10, 15, 20]
    
    # 定义生成文件的原始名称（根据你的描述）
    source_filename = "suggested_features_brute.yaml"
    
    # 定义目标存放目录
    target_dir = "yamls"

    # 确保目标目录存在
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
        print(f"[初始化] 创建目录: {target_dir}")

    # 确保模型目录存在
    if not os.path.exists(model_dir):
        print(f"[错误] 模型目录不存在: {model_dir}")
        sys.exit(1)

    # 2. 扫描并排序 yaml 文件
    # 排序是为了保证每次运行的顺序一致
    yaml_files = [f for f in os.listdir(model_dir) if f.endswith(('.yaml', '.yml'))]
    yaml_files.sort()

    if not yaml_files:
        print(f"[警告] 目录 {model_dir} 中没有找到 YAML 文件。")
        sys.exit(0)

    print(f"找到 {len(yaml_files)} 个模型文件，准备开始处理...")
    print(f"Min Features List: {min_features_list}")
    print("-" * 60)

    idx = start_index

    # 3. 开始双重循环
    for yaml_file in yaml_files:
        # 获取 yaml 文件的完整路径
        yaml_path = os.path.join(model_dir, yaml_file)
        
        print(f"\n>>> 正在处理模型文件: {yaml_file}")

        for min_feat in min_features_list:
            print(f"    [执行中] Index: {idx} | Min Features: {min_feat}")

            # 构造命令
            cmd = (
                f"python scripts/select_features_brute_force_updown.py "
                f"--close_chg_col PF_Close_Chg "
                f"--min_features {min_feat} "
                f"--data_config {yaml_path} "
                f"--params_config configs/params_config/params_updown.yaml "
                f"--training_config configs/training_config/training_updown.yaml"
            )

            try:
                # 执行命令
                # check=True 表示如果脚本报错（返回非0），会抛出异常
                subprocess.run(cmd, shell=True, check=True)

                # 处理生成的文件
                if os.path.exists(source_filename):
                    target_name = f"n{idx}-features.yaml"
                    target_path = os.path.join(target_dir, target_name)
                    
                    # 移动并重命名
                    shutil.move(source_filename, target_path)
                    print(f"    [成功] 生成并移动: {target_path}")
                else:
                    print(f"    [错误] 未找到输出文件 {source_filename}，跳过移动。")

            except subprocess.CalledProcessError as e:
                print(f"    [失败] 命令执行出错，退出码: {e.returncode}")
                # 根据需求，这里可以选择 sys.exit(1) 停止整个脚本，
                # 或者 continue 继续下一个循环。这里选择停止以防止错误蔓延。
                sys.exit(1)
            except Exception as e:
                print(f"    [异常] {e}")
                sys.exit(1)
            
            # 4. 无论如何，idx 自增
            idx += 1

    print("\n" + "=" * 60)
    print(f"所有任务完成。下一个可用的 Index 是: {idx}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="批量运行暴力特征选择脚本")
    
    # 命令行参数
    parser.add_argument('--model_dir', type=str, default="configs/data_config/brute_models", help='包含模型YAML文件的目录路径')
    parser.add_argument('--index', type=int, default=131, help='起始编号 (int)')

    args = parser.parse_args()

    run_brute_batch(args.model_dir, args.index)