from modelscope.hub.api import HubApi
import os
import tempfile

api = HubApi()
api.login("ms-26cd6c38-c344-4f57-aab5-4282398810d9")

# 创建临时目录并复制文件
file_path = "/root/autodl-tmp/mllm_ke/AKE-main/results/models/SERAC_MULTI/blip2_260624_202940-step_5000.pt"
target_filename = "blip2_serac_clevr-step_5000.pt"

with tempfile.TemporaryDirectory() as tmpdir:
    # 复制文件到临时目录
    import shutil
    shutil.copy(file_path, os.path.join(tmpdir, target_filename))
    
    # 上传整个文件夹
    api.upload_folder(
        repo_id="zukia12/mllm_ke",
        folder_path=tmpdir,
        commit_message="upload single model file"
    )
print("上传完成")

