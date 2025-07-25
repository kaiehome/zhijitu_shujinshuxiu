#!/bin/bash

# 蜀锦蜀绣AI打样图生成工具 - 增强版安装脚本
# 自动安装所有依赖和配置环境

set -e  # 遇到错误立即退出

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 日志函数
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 检查系统要求
check_system_requirements() {
    log_info "检查系统要求..."
    
    # 检查Python版本
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
        log_info "检测到Python版本: $PYTHON_VERSION"
        
        if python3 -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)"; then
            log_success "Python版本满足要求 (>= 3.8)"
        else
            log_error "Python版本过低，需要3.8或更高版本"
            exit 1
        fi
    else
        log_error "未检测到Python3，请先安装Python 3.8+"
        exit 1
    fi
    
    # 检查pip
    if command -v pip3 &> /dev/null; then
        log_success "检测到pip3"
    else
        log_error "未检测到pip3，请先安装pip"
        exit 1
    fi
    
    # 检查系统内存
    if command -v free &> /dev/null; then
        MEMORY_GB=$(free -g | awk '/^Mem:/{print $2}')
        log_info "系统内存: ${MEMORY_GB}GB"
        
        if [ "$MEMORY_GB" -lt 4 ]; then
            log_warning "系统内存较少 (${MEMORY_GB}GB)，建议至少4GB内存"
        fi
    fi
    
    # 检查磁盘空间
    DISK_SPACE=$(df -BG . | awk 'NR==2 {print $4}' | sed 's/G//')
    log_info "可用磁盘空间: ${DISK_SPACE}GB"
    
    if [ "$DISK_SPACE" -lt 10 ]; then
        log_warning "磁盘空间较少 (${DISK_SPACE}GB)，建议至少10GB可用空间"
    fi
}

# 检查GPU支持
check_gpu_support() {
    log_info "检查GPU支持..."
    
    # 检查NVIDIA GPU
    if command -v nvidia-smi &> /dev/null; then
        log_success "检测到NVIDIA GPU"
        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits | while IFS=, read -r name memory; do
            log_info "GPU: $name, 显存: ${memory}MB"
        done
        
        # 检查CUDA版本
        if command -v nvcc &> /dev/null; then
            CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $6}' | cut -c2-)
            log_info "CUDA版本: $CUDA_VERSION"
        else
            log_warning "未检测到CUDA编译器，GPU加速功能可能不可用"
        fi
    else
        log_warning "未检测到NVIDIA GPU，将使用CPU模式"
    fi
}

# 创建虚拟环境
create_virtual_environment() {
    log_info "创建Python虚拟环境..."
    
    if [ ! -d "venv" ]; then
        python3 -m venv venv
        log_success "虚拟环境创建成功"
    else
        log_info "虚拟环境已存在"
    fi
    
    # 激活虚拟环境
    source venv/bin/activate
    
    # 升级pip
    log_info "升级pip..."
    pip install --upgrade pip
    
    log_success "虚拟环境配置完成"
}

# 安装基础依赖
install_basic_dependencies() {
    log_info "安装基础依赖..."
    
    # 安装基础包
    pip install -r requirements_enhanced.txt
    
    log_success "基础依赖安装完成"
}

# 安装GPU相关依赖（可选）
install_gpu_dependencies() {
    if command -v nvidia-smi &> /dev/null; then
        log_info "安装GPU相关依赖..."
        
        # 检测CUDA版本并安装对应的PyTorch
        if command -v nvcc &> /dev/null; then
            CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $6}' | cut -c2- | cut -d. -f1,2)
            log_info "检测到CUDA版本: $CUDA_VERSION"
            
            # 根据CUDA版本安装PyTorch
            case $CUDA_VERSION in
                "11.8"|"11.9")
                    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
                    ;;
                "12.1"|"12.2")
                    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
                    ;;
                *)
                    log_warning "未识别的CUDA版本，安装CPU版本PyTorch"
                    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
                    ;;
            esac
        else
            log_warning "未检测到CUDA编译器，安装CPU版本PyTorch"
            pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
        fi
        
        log_success "GPU依赖安装完成"
    else
        log_info "跳过GPU依赖安装（未检测到NVIDIA GPU）"
    fi
}

# 创建必要的目录
create_directories() {
    log_info "创建必要的目录..."
    
    mkdir -p outputs
    mkdir -p test_data
    mkdir -p test_results
    mkdir -p models
    mkdir -p logs
    mkdir -p quality_assessment
    mkdir -p cache
    
    log_success "目录创建完成"
}

# 配置环境变量
setup_environment() {
    log_info "配置环境变量..."
    
    # 创建.env文件
    cat > .env << EOF
# 蜀锦蜀绣AI打样图生成工具环境配置

# 基础配置
ENVIRONMENT=development
DEBUG=true
LOG_LEVEL=INFO

# 路径配置
OUTPUTS_DIR=outputs
TEST_DATA_DIR=test_data
TEST_RESULTS_DIR=test_results
MODELS_DIR=models
LOGS_DIR=logs
CACHE_DIR=cache

# 性能配置
USE_GPU=true
USE_PARALLEL=true
MAX_WORKERS=4
CACHE_SIZE=100

# 深度学习配置
DEEP_LEARNING_ENABLED=true
MODEL_DEVICE=auto
BATCH_SIZE=1

# 质量评估配置
QUALITY_ASSESSMENT_ENABLED=true
SAVE_ASSESSMENT_RESULTS=true

# 日志配置
LOG_FORMAT=json
LOG_FILE=logs/app.log
LOG_MAX_SIZE=100MB
LOG_BACKUP_COUNT=5
EOF
    
    log_success "环境变量配置完成"
}

# 运行基础测试
run_basic_tests() {
    log_info "运行基础测试..."
    
    # 测试Python导入
    python3 -c "
import cv2
import numpy as np
import logging
print('基础库导入测试通过')
"
    
    # 测试新开发的组件
    python3 -c "
try:
    from parallel_processor import ParallelProcessor
    from gpu_accelerator import GPUAccelerator
    from memory_manager import MemoryManager
    from enhanced_image_processor import EnhancedImageProcessor
    from quality_assessment import QualityAssessmentSystem
    print('新组件导入测试通过')
except ImportError as e:
    print(f'组件导入测试失败: {e}')
    exit(1)
"
    
    log_success "基础测试通过"
}

# 运行综合测试套件
run_comprehensive_tests() {
    log_info "运行综合测试套件..."
    
    if [ -f "backend/comprehensive_test_suite.py" ]; then
        cd backend
        python3 comprehensive_test_suite.py --verbose
        cd ..
        log_success "综合测试套件运行完成"
    else
        log_warning "综合测试套件文件不存在，跳过"
    fi
}

# 生成安装报告
generate_installation_report() {
    log_info "生成安装报告..."
    
    REPORT_FILE="installation_report.txt"
    
    cat > $REPORT_FILE << EOF
蜀锦蜀绣AI打样图生成工具 - 安装报告
=====================================

安装时间: $(date)
Python版本: $(python3 --version)
系统信息: $(uname -a)

已安装的组件:
- 并行处理系统
- GPU加速支持
- 智能内存管理
- 增强图像处理器
- 深度学习模型接口
- 质量评估系统
- 综合测试套件

目录结构:
$(find . -type d -name "venv" -prune -o -type d -print | head -20)

环境变量配置:
$(cat .env)

使用说明:
1. 激活虚拟环境: source venv/bin/activate
2. 运行测试: python3 backend/comprehensive_test_suite.py
3. 启动应用: python3 backend/app.py

注意事项:
- 确保已激活虚拟环境
- GPU功能需要NVIDIA显卡和CUDA支持
- 深度学习功能需要安装PyTorch或TensorFlow
EOF
    
    log_success "安装报告已生成: $REPORT_FILE"
}

# 显示使用说明
show_usage_instructions() {
    log_info "显示使用说明..."
    
    cat << EOF

🎉 安装完成！蜀锦蜀绣AI打样图生成工具增强版已成功安装

📋 使用说明:
1. 激活虚拟环境:
   source venv/bin/activate

2. 运行综合测试:
   python3 backend/comprehensive_test_suite.py --verbose

3. 启动应用:
   python3 backend/app.py

4. 查看安装报告:
   cat installation_report.txt

🔧 主要功能:
- 并行图像处理
- GPU加速支持
- 智能内存管理
- 深度学习模型集成
- 质量评估系统
- 综合测试套件

📁 重要目录:
- outputs/: 输出文件
- test_data/: 测试数据
- test_results/: 测试结果
- models/: 模型文件
- logs/: 日志文件

⚠️  注意事项:
- 确保已激活虚拟环境
- GPU功能需要NVIDIA显卡支持
- 深度学习功能需要额外安装PyTorch/TensorFlow

🚀 开始使用:
source venv/bin/activate
python3 backend/comprehensive_test_suite.py

EOF
}

# 主安装流程
main() {
    log_info "开始安装蜀锦蜀绣AI打样图生成工具增强版..."
    
    # 检查系统要求
    check_system_requirements
    
    # 检查GPU支持
    check_gpu_support
    
    # 创建虚拟环境
    create_virtual_environment
    
    # 安装基础依赖
    install_basic_dependencies
    
    # 安装GPU依赖（可选）
    install_gpu_dependencies
    
    # 创建必要目录
    create_directories
    
    # 配置环境变量
    setup_environment
    
    # 运行基础测试
    run_basic_tests
    
    # 运行综合测试套件
    run_comprehensive_tests
    
    # 生成安装报告
    generate_installation_report
    
    # 显示使用说明
    show_usage_instructions
    
    log_success "🎉 安装完成！"
}

# 错误处理
trap 'log_error "安装过程中发生错误，请检查日志"; exit 1' ERR

# 运行主函数
main "$@" 