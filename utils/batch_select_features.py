import os
import shutil
import subprocess
import sys

def run_batch_process():
    # 1. 定义参数列表
    step_size_list = [1, 5]
    min_features_list = [1, 5, 10, 15, 20]
    
    # 2. 定义初始编号
    idx = 65

    # 定义目标文件夹
    target_dir = "yamls"
    
    # 确保 yamls 目录存在，如果不存在则创建
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
        print(f"[信息] 已创建目录: {target_dir}")

    # 定义要按顺序执行的脚本列表
    # 这样写可以避免复制粘贴 4 遍相同的逻辑
    scripts_to_run = [
        "scripts/select_features_stepwise_updown.py",
        "scripts/select_features_stepwise_updown_zeroimp.py",
        "scripts/select_features_stepwise_updown_allfeats.py",
        "scripts/select_features_stepwise_updown_zeroimp_allfeats.py"
    ]

    print(f"开始批量处理任务...")
    print(f"Step Sizes: {step_size_list}")
    print(f"Min Features: {min_features_list}")
    print("-" * 60)

    # 3. 开始循环
    for i in step_size_list:
        for j in min_features_list:
            
            # 针对当前的 i (step_size) 和 j (min_features)，依次运行 4 个脚本
            for script_path in scripts_to_run:
                print(f"\n[正在处理] ID: n{idx} | Step: {i} | Min: {j} | Script: {os.path.basename(script_path)}")
                
                # 构造命令
                cmd = (
                    f"python {script_path} "
                    f"--close_chg_col PF_Close_Chg "
                    f"--step_size {i} "
                    f"--min_features {j} "
                    f"--data_config configs/data_config/model.yaml "
                    f"--params_config configs/params_config/params_updown.yaml "
                    f"--training_config configs/training_config/training_updown.yaml"
                )
                
                try:
                    # 1. 执行命令
                    # check=True 表示如果命令返回非0错误码，会抛出异常
                    subprocess.run(cmd, shell=True, check=True)
                    
                    # 2. 处理生成的文件
                    source_file = "suggested_features.yaml"
                    target_filename = f"n{idx}-features.yaml"
                    target_path = os.path.join(target_dir, target_filename)

                    if os.path.exists(source_file):
                        # 移动并重命名
                        # shutil.move 会自动处理跨文件系统的移动
                        shutil.move(source_file, target_path)
                        print(f"[成功] 文件已生成并移动至: {target_path}")
                    else:
                        print(f"[错误] 命令执行完成，但未找到生成的 {source_file}。跳过移动操作。")
                        # 如果没有生成文件，根据需求决定是否退出，这里选择继续运行但不增加 idx 或报错
                        # sys.exit(1) 

                    # 3. 编号递增 (只有成功运行并预期生成文件后才加，或者无论如何都加？)
                    # 根据你的伪代码，无论如何都要执行 idx += 1
                    idx += 1

                except subprocess.CalledProcessError as e:
                    print(f"[严重错误] 执行命令失败，退出码: {e.returncode}")
                    print(f"失败命令: {cmd}")
                    # 如果遇到错误想停止整个脚本，取消下面这行的注释
                    # sys.exit(1) 
                except Exception as e:
                    print(f"[未知错误] {e}")
                    sys.exit(1)

    print("\n" + "=" * 60)
    print("所有批量任务处理完毕！")

if __name__ == "__main__":
    run_batch_process()