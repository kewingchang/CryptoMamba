import argparse
import subprocess
import sys

def run_batch(start_idx, end_idx):
    print(f"准备执行批量任务: 从 {start_idx} 到 {end_idx}")

    # range 的第二个参数是开区间，所以需要 +1 才能包含 end_idx
    for i in range(start_idx, end_idx + 1):
        # 构造命令字符串
        # f-string 中的 {i} 会被替换为当前的循环数字
        cmd = (
            f"python scripts/training_updown.py "
            f"--close_chg_col PF_Close_Chg "
            f"--data_config configs/data_config/model-n{i}.yaml "
            f"--params_config configs/params_config/params_updown.yaml "
            f"--training_config configs/training_config/training_updown.yaml"
        )

        print("\n" + "=" * 80)
        print(f"正在执行 Index: {i}")
        print(f"执行命令: {cmd}")
        print("-" * 80)

        try:
            # shell=True 表示在 shell 中执行命令
            # check=True 表示如果命令执行出错（返回码非0），则抛出异常停止脚本
            subprocess.run(cmd, shell=True, check=True)
        
        except subprocess.CalledProcessError as e:
            print(f"\n[错误] 任务 (Index: {i}) 执行失败，返回码: {e.returncode}")
            # print("脚本已停止。请检查报错信息。")
            # sys.exit(1)
        except KeyboardInterrupt:
            print("\n[用户中断] 批量任务已手动停止。")
            sys.exit(0)

    print("\n" + "=" * 80)
    print("所有任务执行完毕！")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="批量运行 training_updown.py 脚本")
    
    # 定义命令行参数
    parser.add_argument('--start', type=int, required=True, help='起始编号 (整数)')
    parser.add_argument('--end', type=int, required=True, help='结束编号 (整数)')

    args = parser.parse_args()

    # 简单的参数校验
    if args.start > args.end:
        print("[错误] start 不能大于 end")
        sys.exit(1)

    run_batch(args.start, args.end)