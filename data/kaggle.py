import kagglehub
import os
import shutil

# 下载数据集
path = kagglehub.dataset_download("zhuoqihe/ak-bench")
print('下载完成，源路径:', path)

# 目标路径
target_path = '.'

# 确保目标目录存在
os.makedirs(target_path, exist_ok=True)

# 移动所有文件到目标路径
for item in os.listdir(path):
    src = os.path.join(path, item)
    dst = os.path.join(target_path, item)
    if os.path.isdir(src):
        shutil.copytree(src, dst, dirs_exist_ok=True)
    else:
        shutil.copy2(src, dst)

print('数据集已移动到:', target_path)
