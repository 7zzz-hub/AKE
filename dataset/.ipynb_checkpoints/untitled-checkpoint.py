import kagglehub
import shutil
import os

# 下载数据集
path = kagglehub.dataset_download("vimalvk22/imagenet-1k-validation")
print("原始路径:", path)

# 目标文件夹
target_path = "/root/autodl-tmp/mllm_ke/akedit/dataset/imagenet"  # 修改为你的目标路径

# 移动文件夹
shutil.move(path, target_path)
print("已移动到:", target_path)