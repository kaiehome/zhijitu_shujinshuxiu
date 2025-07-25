# AI大模型织机识别图生成系统

## 概述

本系统使用深度学习技术训练AI大模型，用于生成专业的织机识别图，替代传统的手工图像处理方法。

## 系统架构

### 核心组件

1. **AI模型训练器** (`backend/ai_model_trainer.py`)
   - 基于GAN（生成对抗网络）架构
   - 包含生成器和判别器
   - 支持多种损失函数和优化策略

2. **训练脚本** (`backend/train_ai_model.py`)
   - 完整的训练流程
   - 自动数据预处理
   - 模型保存和加载

3. **模型API** (`backend/ai_model_api.py`)
   - FastAPI集成
   - 模型推理接口
   - 状态监控

4. **测试工具** (`backend/test_ai_model.py`)
   - 模型性能测试
   - 生成质量评估

## 安装步骤

### 1. 安装依赖

```bash
# 运行安装脚本
chmod +x install_ai_dependencies.sh
./install_ai_dependencies.sh
```

### 2. 准备训练数据

```bash
# 创建数据目录
mkdir -p target_images

# 将专业软件生成的识别图放入target_images目录
# 将原始图片放入uploads目录
```

### 3. 训练模型

```bash
# 开始训练
cd backend
python3 train_ai_model.py --epochs 100 --batch-size 8
```

### 4. 测试模型

```bash
# 测试训练好的模型
python3 test_ai_model.py --input ../uploads/your_image.jpg
```

### 5. 启动API服务

```bash
# 启动服务器
python3 main.py
```

### 6. 测试API

```bash
# 测试AI模型API
python3 ../test_ai_model_api.py
```

## API接口

### 1. 生成织机识别图

**POST** `/api/generate-ai-model`

上传原始图片，使用AI模型生成织机识别图。

**请求：**
- Content-Type: `multipart/form-data`
- Body: `file` (图片文件)

**响应：**
- Content-Type: `image/jpeg`
- Body: 生成的织机识别图

### 2. 获取模型状态

**GET** `/api/ai-model-status`

获取AI模型的加载状态和配置信息。

**响应：**
```json
{
    "is_loaded": true,
    "model_dir": "trained_models",
    "model_epoch": "final",
    "device_used": "cuda",
    "generator_params": 1234567,
    "discriminator_params": 987654
}
```

## 训练参数

### 模型配置

- **生成器架构**: U-Net with ResNet blocks
- **判别器架构**: PatchGAN
- **输入尺寸**: 256x256
- **批次大小**: 8 (可调整)
- **学习率**: 0.0002
- **优化器**: Adam

### 损失函数

- **生成器损失**: 
  - 对抗损失
  - L1重建损失
  - 感知损失
- **判别器损失**: 对抗损失

### 训练策略

- **数据增强**: 随机翻转、旋转、颜色抖动
- **学习率调度**: 线性衰减
- **早停机制**: 验证损失监控
- **模型保存**: 最佳模型和最新模型

## 性能优化

### GPU加速

系统自动检测CUDA可用性：
- 有GPU：使用CUDA加速训练和推理
- 无GPU：使用CPU模式

### 内存优化

- 梯度累积
- 混合精度训练
- 动态批次大小

### 推理优化

- 模型量化
- 批处理推理
- 缓存机制

## 质量评估

### 评估指标

1. **PSNR** (Peak Signal-to-Noise Ratio)
2. **SSIM** (Structural Similarity Index)
3. **FID** (Fréchet Inception Distance)
4. **LPIPS** (Learned Perceptual Image Patch Similarity)

### 可视化工具

- 训练过程曲线
- 生成结果对比
- 损失函数变化

## 故障排除

### 常见问题

1. **CUDA内存不足**
   ```bash
   # 减少批次大小
   python3 train_ai_model.py --batch-size 4
   ```

2. **模型加载失败**
   ```bash
   # 检查模型文件
   ls -la trained_models/
   ```

3. **训练数据不足**
   - 增加训练图片数量
   - 使用数据增强
   - 预训练模型迁移

### 日志查看

```bash
# 查看训练日志
tail -f ai_training.log

# 查看API日志
tail -f api_server.log
```

## 扩展功能

### 1. 多风格支持

可以训练多个模型支持不同的识别图风格：
- 传统风格
- 现代风格
- 艺术风格

### 2. 实时训练

支持在线学习和模型更新：
- 增量训练
- 模型版本管理
- A/B测试

### 3. 批量处理

支持批量图片处理：
- 队列管理
- 进度监控
- 结果汇总

## 技术栈

- **深度学习框架**: PyTorch
- **图像处理**: OpenCV, PIL
- **Web框架**: FastAPI
- **数据科学**: NumPy, scikit-learn
- **可视化**: Matplotlib
- **实验跟踪**: Weights & Biases

## 贡献指南

1. Fork项目
2. 创建功能分支
3. 提交更改
4. 创建Pull Request

## 许可证

MIT License

## 联系方式

如有问题或建议，请提交Issue或联系开发团队。 