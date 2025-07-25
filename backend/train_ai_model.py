#!/usr/bin/env python3
"""
AI大模型训练脚本
用于训练织机识别图生成模型
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ai_model_trainer import LoomRecognitionTrainer, create_target_images_dataset

def setup_logging():
    """设置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('ai_training.log'),
            logging.StreamHandler()
        ]
    )

def check_dependencies():
    """检查依赖"""
    try:
        import torch
        import torchvision
        import matplotlib
        import tqdm
        print("✓ 所有依赖已安装")
        return True
    except ImportError as e:
        print(f"✗ 缺少依赖: {e}")
        print("请运行: pip install torch torchvision matplotlib tqdm wandb")
        return False

def check_data():
    """检查训练数据"""
    uploads_dir = Path("uploads")
    if not uploads_dir.exists():
        print("✗ uploads目录不存在")
        return False
    
    source_files = list(uploads_dir.glob("*.jpg")) + list(uploads_dir.glob("*.png"))
    if len(source_files) == 0:
        print("✗ uploads目录中没有图片文件")
        return False
    
    print(f"✓ 找到 {len(source_files)} 个源图片")
    return True

def main():
    parser = argparse.ArgumentParser(description="训练AI大模型生成织机识别图")
    parser.add_argument("--epochs", type=int, default=50, help="训练轮数")
    parser.add_argument("--batch-size", type=int, default=4, help="批次大小")
    parser.add_argument("--image-size", type=int, default=256, help="图片尺寸")
    parser.add_argument("--learning-rate", type=float, default=0.0002, help="学习率")
    parser.add_argument("--save-interval", type=int, default=5, help="保存间隔")
    parser.add_argument("--device", type=str, default="auto", help="设备 (cuda/cpu/auto)")
    parser.add_argument("--skip-data-check", action="store_true", help="跳过数据检查")
    
    args = parser.parse_args()
    
    # 设置日志
    setup_logging()
    logger = logging.getLogger(__name__)
    
    print("=" * 60)
    print("AI大模型训练 - 织机识别图生成器")
    print("=" * 60)
    
    # 检查依赖
    if not check_dependencies():
        return
    
    # 检查数据
    if not args.skip_data_check and not check_data():
        return
    
    # 确定设备
    if args.device == "auto":
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    print(f"使用设备: {device}")
    
    try:
        # 创建目标数据集
        print("\n1. 创建目标数据集...")
        target_count = create_target_images_dataset()
        
        if target_count == 0:
            print("错误：没有找到目标图片")
            print("请先运行一些图像处理生成专业识别图，或者手动创建target_images目录")
            return
        
        print(f"✓ 创建了 {target_count} 个目标图片")
        
        # 初始化训练器
        print("\n2. 初始化AI训练器...")
        trainer = LoomRecognitionTrainer(
            device=device,
            learning_rate=args.learning_rate
        )
        
        # 准备数据
        print("\n3. 准备训练数据...")
        dataloader = trainer.prepare_data(
            batch_size=args.batch_size,
            image_size=args.image_size
        )
        
        print(f"✓ 数据加载器准备完成，批次大小: {args.batch_size}")
        
        # 开始训练
        print(f"\n4. 开始训练 ({args.epochs} 轮)...")
        trainer.train(
            dataloader,
            epochs=args.epochs,
            save_interval=args.save_interval
        )
        
        print("\n✓ AI模型训练完成！")
        print("\n模型文件保存在 trained_models/ 目录中")
        print("可以使用以下命令测试模型:")
        print("python test_ai_model.py --input your_image.jpg")
        
    except Exception as e:
        logger.error(f"训练过程中出现错误: {e}")
        print(f"✗ 训练失败: {e}")
        return

if __name__ == "__main__":
    main() 