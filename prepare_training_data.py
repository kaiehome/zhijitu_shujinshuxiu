#!/usr/bin/env python3
"""
训练数据准备脚本
将现有图片转换为AI模型训练所需的数据集
"""

import os
import shutil
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import logging

def setup_logging():
    """设置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def create_directories():
    """创建必要的目录"""
    directories = [
        "target_images",
        "trained_models",
        "training_logs",
        "validation_images"
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✓ 创建目录: {directory}")

def prepare_sample_data():
    """准备示例训练数据"""
    print("\n准备示例训练数据...")
    
    # 检查uploads目录中的图片
    uploads_dir = Path("uploads")
    if not uploads_dir.exists():
        print("⚠ uploads目录不存在，创建空目录")
        uploads_dir.mkdir(exist_ok=True)
        return
    
    # 查找图片文件
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        image_files.extend(uploads_dir.glob(ext))
    
    if not image_files:
        print("⚠ uploads目录中没有找到图片文件")
        print("请上传一些图片到uploads目录作为训练数据")
        return
    
    print(f"找到 {len(image_files)} 个图片文件")
    
    # 创建示例目标图片（使用现有的图像处理方法）
    target_dir = Path("target_images")
    target_dir.mkdir(exist_ok=True)
    
    try:
        from backend.local_ai_generator import LocalAIGenerator
        
        generator = LocalAIGenerator()
        
        for i, image_file in enumerate(image_files[:5]):  # 只处理前5张图片作为示例
            print(f"处理图片 {i+1}/{min(5, len(image_files))}: {image_file.name}")
            
            try:
                # 使用本地AI增强生成器创建目标图片
                output_path = generator.generate_local_ai_enhanced_image(str(image_file))
                
                # 复制到target_images目录
                target_path = target_dir / f"target_{image_file.stem}.jpg"
                shutil.copy2(output_path, target_path)
                
                print(f"  ✓ 生成目标图片: {target_path.name}")
                
            except Exception as e:
                print(f"  ✗ 处理失败: {e}")
                continue
    
    except ImportError:
        print("⚠ 无法导入LocalAIGenerator，跳过示例数据生成")
        print("请先确保backend目录中的模块可用")

def validate_data():
    """验证训练数据"""
    print("\n验证训练数据...")
    
    source_dir = Path("uploads")
    target_dir = Path("target_images")
    
    if not source_dir.exists() or not target_dir.exists():
        print("✗ 源目录或目标目录不存在")
        return False
    
    source_images = list(source_dir.glob("*.jpg")) + list(source_dir.glob("*.png"))
    target_images = list(target_dir.glob("*.jpg")) + list(target_dir.glob("*.png"))
    
    print(f"源图片数量: {len(source_images)}")
    print(f"目标图片数量: {len(target_images)}")
    
    if len(source_images) == 0:
        print("✗ 没有找到源图片")
        return False
    
    if len(target_images) == 0:
        print("✗ 没有找到目标图片")
        print("请运行 prepare_sample_data() 生成示例数据")
        return False
    
    # 检查图片尺寸
    print("\n检查图片尺寸...")
    for img_path in source_images[:3]:  # 检查前3张
        try:
            with Image.open(img_path) as img:
                print(f"  {img_path.name}: {img.size}")
        except Exception as e:
            print(f"  ✗ 无法读取 {img_path.name}: {e}")
    
    return True

def create_data_config():
    """创建数据配置文件"""
    config = {
        "source_dir": "uploads",
        "target_dir": "target_images",
        "image_size": 256,
        "batch_size": 8,
        "validation_split": 0.2,
        "augmentation": {
            "horizontal_flip": True,
            "vertical_flip": False,
            "rotation": 10,
            "brightness": 0.1,
            "contrast": 0.1
        }
    }
    
    import json
    with open("training_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    print("✓ 创建训练配置文件: training_config.json")

def main():
    """主函数"""
    print("=" * 60)
    print("AI模型训练数据准备工具")
    print("=" * 60)
    
    setup_logging()
    
    # 创建目录
    create_directories()
    
    # 准备示例数据
    prepare_sample_data()
    
    # 验证数据
    if validate_data():
        print("\n✓ 训练数据准备完成")
        
        # 创建配置文件
        create_data_config()
        
        print("\n下一步操作:")
        print("1. 检查 target_images 目录中的目标图片质量")
        print("2. 如有需要，手动调整或替换目标图片")
        print("3. 运行训练脚本: python3 backend/train_ai_model.py")
    else:
        print("\n✗ 训练数据准备失败")
        print("请检查数据文件并重试")

if __name__ == "__main__":
    main() 