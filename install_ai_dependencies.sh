#!/bin/bash

echo "=========================================="
echo "AI大模型依赖安装脚本"
echo "=========================================="

# 检查Python版本
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python版本: $python_version"

# 检查CUDA是否可用
if command -v nvidia-smi &> /dev/null; then
    echo "✓ 检测到NVIDIA GPU"
    cuda_available=true
else
    echo "⚠ 未检测到NVIDIA GPU，将使用CPU版本"
    cuda_available=false
fi

# 安装基础依赖
echo ""
echo "1. 安装基础依赖..."
pip3 install --upgrade pip

# 安装PyTorch
echo ""
echo "2. 安装PyTorch..."
if [ "$cuda_available" = true ]; then
    echo "安装CUDA版本的PyTorch..."
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
else
    echo "安装CPU版本的PyTorch..."
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

# 安装其他依赖
echo ""
echo "3. 安装其他依赖..."
pip3 install matplotlib tqdm wandb pillow numpy opencv-python scikit-learn

# 验证安装
echo ""
echo "4. 验证安装..."
python3 -c "
import torch
import torchvision
import matplotlib
import tqdm
import PIL
import cv2
import sklearn
print('✓ 所有依赖安装成功')
print(f'PyTorch版本: {torch.__version__}')
print(f'CUDA可用: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU数量: {torch.cuda.device_count()}')
    print(f'当前GPU: {torch.cuda.get_device_name(0)}')
"

echo ""
echo "=========================================="
echo "安装完成！"
echo "=========================================="
echo ""
echo "下一步操作："
echo "1. 准备训练数据（上传图片到uploads目录）"
echo "2. 运行训练脚本：python3 backend/train_ai_model.py"
echo "3. 测试模型：python3 backend/test_ai_model.py --input your_image.jpg"
echo "4. 启动API服务器：python3 backend/main.py"
echo "5. 测试API：python3 test_ai_model_api.py"
echo "" 