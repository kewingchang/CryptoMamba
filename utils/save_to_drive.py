# @title Save model
import os
import shutil
from datetime import datetime

if __name__ == "__main__":
	# 生成时间戳
	dir_time = datetime.now().strftime('%Y%m%d')
	time_stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
	target_dir = f'/content/drive/MyDrive/training/{dir_time}/{time_stamp}'

	# 创建目录
	os.makedirs(target_dir, exist_ok=True)
	print(f"Created model dir: {target_dir}")

	# 拷贝文件
	source_dir = 'logs/CMamba/version_0/checkpoints'
	for file in os.listdir(source_dir):
	    if file.startswith('epoch') and file.endswith('.ckpt'):
	        shutil.copy(os.path.join(source_dir, file), target_dir)
	        print(f"Copied {file} to {target_dir}")

	# 拷贝其他文件和目录
	shutil.copy('log.txt', target_dir)
	print(f"Copied log.txt to {target_dir}")
	shutil.copy('logs/CMamba/version_0/hparams.yaml', target_dir)
	print(f"Copied hparams.yaml to {target_dir}")
	shutil.copytree('data', os.path.join(target_dir, 'data'), dirs_exist_ok=True)
	print(f"Copied data/ to {target_dir}")
	shutil.copytree('configs', os.path.join(target_dir, 'configs'), dirs_exist_ok=True)
	print(f"Copied configs/ to {target_dir}")
	# shutil.copytree('Results', os.path.join(target_dir, 'results'), dirs_exist_ok=True)
	# print(f"Copied results/ to {target_dir}")

	print("Copy done.")
