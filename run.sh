#!/bin/bash

set -e
set -o pipefail

########################################

# 彩色输出

########################################
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() {
echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warn() {
echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
echo -e "${RED}[ERROR]${NC} $1"
}

run_step() {
echo ""
echo -e "${GREEN}====================================================${NC}"
echo -e "${GREEN}$1${NC}"
echo -e "${GREEN}====================================================${NC}"
}

########################################

# 环境信息

########################################

run_step "[0/5] 检查运行环境"

python --version || true
pip --version || true

if command -v nvidia-smi &> /dev/null; then
nvidia-smi
else
log_warn "未检测到 NVIDIA GPU"
fi

########################################

# 安装依赖

########################################

run_step "[1/5] 安装 Python 依赖"

pip install -r requirements.txt

log_info "安装 PyTorch CUDA 12.1"

pip install torch==2.1.2 \
torchvision==0.16.2 \
torchaudio==2.1.2 \
--index-url https://download.pytorch.org/whl/cu121

log_success "Python环境安装完成"

########################################

# 数据集下载

########################################

run_step "[2/5] 下载数据集"

apt-get update
apt-get install -y aria2 unzip wget curl

cd data/clevr_dataset

# if [ ! -f "CLEVR_CoGenT_v1.0.zip" ]; then
# log_info "下载 CLEVR CoGenT 数据集"
# aria2c -x 16 -s 16 --summary-interval=5 https://dl.fbaipublicfiles.com/clevr/CLEVR_CoGenT_v1.0.zip
# else
# log_warn "CLEVR压缩包已存在，跳过下载"
# fi

# if [ ! -d "CLEVR_CoGenT_v1.0" ]; then
# log_info "解压 CLEVR 数据集"
# unzip -q CLEVR_CoGenT_v1.0.zip
# else
# log_warn "CLEVR数据集已存在，跳过解压"
# fi

rm -f CLEVR_CoGenT_v1.0.zip

cd ../

log_info "安装 KaggleHub"

pip install -U kagglehub

python kaggle.py

log_success "数据集准备完成"

########################################

# HuggingFace 模型下载

########################################

run_step "[3/5] 下载 HuggingFace 模型"

cd ../

mkdir -p huggingface_cache

cd huggingface_cache

export HF_ENDPOINT=https://hf-mirror.com

download_model() {

```
MODEL_NAME=$1
LOCAL_DIR=$2

echo ""
log_info "下载模型: ${MODEL_NAME}"

if [ -d "${LOCAL_DIR}" ]; then
    log_warn "${LOCAL_DIR} 已存在，跳过"
    return
fi

huggingface-cli download \
    "${MODEL_NAME}" \
    --local-dir "${LOCAL_DIR}"

log_success "${MODEL_NAME} 下载完成"
```

}

download_model google-bert/bert-base-uncased bert-base-uncased
download_model distilbert/distilbert-base-cased distilbert-base-cased
download_model liuhaotian/llava-v1.5-7b llava-v1.5-7b
download_model facebook/opt-6.7b opt-6.7b
download_model facebook/opt-1.3b opt-1.3b
download_model Qwen/Qwen2.5-7B Qwen2.5-7B
download_model Qwen/Qwen2.5-VL-7B-Instruct Qwen2.5-VL
download_model lmsys/vicuna-7b-v1.5 vicuna-7b-v1.5
download_model openai/clip-vit-large-patch14-336 clip-vit-large-patch14-336

log_info "下载 BLIP2 权重"

[ -f blip2_pretrained_opt2.7b.pth ] || 
aria2c -x 16 -s 16 
https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_pretrained_opt6.7b.pth

[ -f eva_vit_g.pth ] || 
aria2c -x 16 -s 16 
https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/eva_vit_g.pth

log_success "HuggingFace模型下载完成"

########################################

# 预训练编辑模型下载

########################################

run_step "[4/5] 下载编辑模型"

cd ../

pip install -U modelscope

mkdir -p results/models/MEND
mkdir -p results/models/SERAC_MULTI

modelscope download \
--model zukia12/mllm_ke \
llava_mend_clevr-step_12000.pt \
--local_dir results/models/MEND/

modelscope download \
--model zukia12/mllm_ke \
blip2_mend_clevr-step_3000.pt \
--local_dir results/models/MEND/

modelscope download \
--model zukia12/mllm_ke \
llava_serac_clevr-step_14000.pt \
--local_dir results/models/SERAC_MULTI/

modelscope download \
--model zukia12/mllm_ke \
llava_serac_supple-step_12000.pt \
--local_dir results/models/SERAC_MULTI/

modelscope download \
--model zukia12/mllm_ke \
blip2_serac_clevr-step_6000.pt \
--local_dir results/models/SERAC_MULTI/

modelscope download \
--model zukia12/mllm_ke \
blip2_serac_supple-step_6000.pt \
--local_dir results/models/SERAC_MULTI/

log_success "编辑模型下载完成"

########################################

# 完成

########################################

run_step "[5/5] 安装完成"

echo ""
log_success "所有依赖安装完成"
log_success "所有数据集下载完成"
log_success "所有模型下载完成"

echo ""
echo "========================================"
echo " 环境已准备完毕"
echo "========================================"
echo ""
