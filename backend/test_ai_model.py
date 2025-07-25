#!/usr/bin/env python3
"""
AI模型测试脚本
用于测试训练好的织机识别图生成模型
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ai_model_trainer import LoomRecognitionTrainer

def setup_logging():
    """设置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def check_model_files():
    """检查模型文件"""
    model_dir = Path("trained_models")
    if not model_dir.exists():
        print("✗ trained_models目录不存在")
        return False
    
    generator_file = model_dir / "generator_epoch_final.pth"
    if not generator_file.exists():
        print("✗ 生成器模型文件不存在")
        print("请先运行训练脚本: python train_ai_model.py")
        return False
    
    print("✓ 模型文件存在")
    return True

def main():
    parser = argparse.ArgumentParser(description="测试AI大模型生成织机识别图")
    parser.add_argument("--input", type=str, required=True, help="输入图片路径")
    parser.add_argument("--output", type=str, help="输出图片路径")
    parser.add_argument("--model-epoch", type=str, default="final", help="模型epoch")
    parser.add_argument("--device", type=str, default="auto", help="设备 (cuda/cpu/auto)")
    
    args = parser.parse_args()
    
    # 设置日志
    setup_logging()
    logger = logging.getLogger(__name__)
    
    print("=" * 60)
    print("AI大模型测试 - 织机识别图生成器")
    print("=" * 60)
    
    # 检查输入文件
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"✗ 输入文件不存在: {args.input}")
        return
    
    # 检查模型文件
    if not check_model_files():
        return
    
    # 确定设备
    if args.device == "auto":
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    print(f"使用设备: {device}")
    
    try:
        # 初始化训练器
        print("\n1. 加载AI模型...")
        trainer = LoomRecognitionTrainer(device=device)
        
        # 加载模型
        trainer.load_models("trained_models", args.model_epoch)
        
        # 生成输出路径
        if args.output:
            output_path = args.output
        else:
            input_stem = input_path.stem
            output_path = f"ai_generated_{input_stem}.png"
        
        print(f"\n2. 生成织机识别图...")
        print(f"输入: {args.input}")
        print(f"输出: {output_path}")
        
        # 生成图片
        result_path = trainer.generate_image(args.input, output_path)
        
        print(f"\n✓ AI生成完成！")
        print(f"结果保存在: {result_path}")
        
        # 显示文件信息
        if os.path.exists(result_path):
            file_size = os.path.getsize(result_path) / 1024  # KB
            print(f"文件大小: {file_size:.1f} KB")
        
    except Exception as e:
        logger.error(f"测试过程中出现错误: {e}")
        print(f"✗ 测试失败: {e}")
        return

if __name__ == "__main__":
    main() 