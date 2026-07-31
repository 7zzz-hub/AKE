#!/bin/bash

# ============================================================================
# 颜色输出设置
# ============================================================================
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ============================================================================
# 日志函数
# ============================================================================
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_step() {
    echo -e "\n${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}\n"
}

# ============================================================================
# 步骤1: 安装 Python 依赖
# ============================================================================
step_install_dependencies() {
    log_step "[1/4] 安装 Python 依赖"
    
    if [ ! -f "requirements.txt" ]; then
        log_error "requirements.txt 文件不存在"
        exit 1
    fi
    
    log_info "开始安装 Python 依赖..."
    pip install -r requirements.txt

    pip install torch==2.1.2 \
    torchvision==0.16.2 \
    torchaudio==2.1.2 \
    --index-url https://download.pytorch.org/whl/cu121

    if [ $? -ne 0 ]; then
        log_error "依赖安装失败，请检查 requirements.txt"
        exit 1
    fi
    log_success "依赖安装完成"
    
    # 确保 huggingface-hub 已安装
    if ! command -v huggingface-cli &> /dev/null; then
        log_warn "huggingface-cli 未安装，正在安装..."
        pip install huggingface-hub
        if [ $? -ne 0 ]; then
            log_error "huggingface-hub 安装失败"
            exit 1
        fi
        log_success "huggingface-hub 安装完成"
    fi
}

# ============================================================================
# 步骤2: 下载 CLEVR 数据集
# ============================================================================
step_download_dataset() {
    log_step "[2/4] 下载 CLEVR 数据集"
    
    # 安装必要工具
    log_info "安装下载工具..."
    apt-get update -qq
    apt-get install -y -qq aria2 unzip wget curl
    
    # 创建数据目录
    mkdir -p data/clevr_dataset
    cd data/clevr_dataset || exit 1
    
    # 下载数据集
    if [ ! -f "CLEVR_CoGenT_v1.0.zip" ]; then
        log_info "下载 CLEVR CoGenT 数据集..."
        aria2c -x 16 -s 16 --summary-interval=5 \
            https://dl.fbaipublicfiles.com/clevr/CLEVR_CoGenT_v1.0.zip
        if [ $? -ne 0 ]; then
            log_error "数据集下载失败"
            cd ../..
            exit 1
        fi
        log_success "数据集下载完成"
    else
        log_warn "CLEVR压缩包已存在，跳过下载"
    fi
    
    # 解压数据集
    if [ ! -d "CLEVR_CoGenT_v1.0" ]; then
        log_info "解压 CLEVR 数据集..."
        unzip -q CLEVR_CoGenT_v1.0.zip
        if [ $? -ne 0 ]; then
            log_error "数据集解压失败"
            cd ../..
            exit 1
        fi
        log_success "数据集解压完成"
    else
        log_warn "CLEVR数据集已存在，跳过解压"
    fi
    
    # 清理压缩包
    rm -f CLEVR_CoGenT_v1.0.zip
    
    cd ../..
    log_success "CLEVR 数据集准备完成"
}

# ============================================================================
# 步骤3: 下载 Kaggle 数据集
# ============================================================================
step_download_kaggle() {
    log_step "[3/4] 下载 Kaggle 数据集"
    
    log_info "安装 KaggleHub..."
    pip install -U kagglehub -q
    
    if [ ! -f "kaggle.py" ]; then
        log_error "kaggle.py 文件不存在"
        exit 1
    fi
    
    log_info "运行 kaggle.py 下载数据集..."
    python kaggle.py
    
    if [ $? -ne 0 ]; then
        log_error "Kaggle 数据集下载失败"
        exit 1
    fi
    
    log_success "Kaggle 数据集准备完成"
}

# ============================================================================
# 步骤4: 下载 HuggingFace 模型
# ============================================================================
step_download_models() {
    log_step "[4/4] 下载 HuggingFace 模型"
    
    # 创建缓存目录
    mkdir -p huggingface_cache
    
    # 下载模型函数
    download_model() {
        local MODEL_NAME=$1
        local LOCAL_DIR=$2
        
        echo ""
        log_info "下载模型: ${MODEL_NAME}"
        log_info "保存路径: ${LOCAL_DIR}"
        
        # 检查目录是否已存在且非空
        if [ -d "${LOCAL_DIR}" ] && [ "$(ls -A ${LOCAL_DIR} 2>/dev/null)" ]; then
            log_warn "${LOCAL_DIR} 已存在且非空，跳过下载"
            return 0
        fi
        
        # 下载模型
        huggingface-cli download \
            "${MODEL_NAME}" \
            --local-dir "${LOCAL_DIR}" \
            --local-dir-use-symlinks False \
            --resume-download
        
        if [ $? -eq 0 ]; then
            log_success "${MODEL_NAME} 下载完成"
            return 0
        else
            log_error "${MODEL_NAME} 下载失败"
            return 1
        fi
    }
    
    # 模型列表
    declare -A MODELS=(
        ["Qwen/Qwen2-VL-7B-Instruct"]="huggingface_cache/Qwen2-VL"
        ["Salesforce/blip2-opt-6.7b"]="huggingface_cache/blip2-opt-6.7b"
        ["llava-hf/llava-1.5-7b-hf"]="huggingface_cache/llava-1.5-7b-hf"
    )
    
    # 记录下载失败的模型
    local FAILED_MODELS=()
    
    log_info "开始下载 ${#MODELS[@]} 个模型..."
    
    for MODEL_NAME in "${!MODELS[@]}"; do
        download_model "${MODEL_NAME}" "${MODELS[${MODEL_NAME}]}"
        if [ $? -ne 0 ]; then
            FAILED_MODELS+=("${MODEL_NAME}")
        fi
    done
    
    # 输出结果
    echo ""
    if [ ${#FAILED_MODELS[@]} -eq 0 ]; then
        log_success "所有模型下载完成！"
        return 0
    else
        log_error "以下模型下载失败："
        for model in "${FAILED_MODELS[@]}"; do
            echo "  - ${model}"
        done
        log_error "请检查网络连接或重试"
        return 1
    fi
}

# ============================================================================
# 主函数
# ============================================================================
main() {
    log_step "开始环境初始化"
    
    # 检查 Python 环境
    if ! command -v python &> /dev/null; then
        log_error "Python 未安装"
        exit 1
    fi
    
    if ! command -v pip &> /dev/null; then
        log_error "pip 未安装"
        exit 1
    fi
    
    # 依次执行各步骤
    step_install_dependencies
    step_download_dataset
    step_download_kaggle
    step_download_models
    
    # 完成
    echo ""
    log_step "🎉 所有步骤完成！"
    log_success "环境初始化成功"
}

# ============================================================================
# 执行主函数
# ============================================================================
main
