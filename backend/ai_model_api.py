#!/usr/bin/env python3
"""
AI模型API集成
将训练好的AI模型集成到FastAPI服务器中
"""

import os
import sys
import logging
import time
from pathlib import Path
from typing import Optional

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ai_model_trainer import LoomRecognitionTrainer

class AIModelAPI:
    """
    AI模型API封装
    用于在FastAPI中集成AI模型
    """
    
    def __init__(self, 
                 model_dir: str = "trained_models",
                 model_epoch: str = "final",
                 device: str = "auto"):
        
        # 模型文件在backend目录下
        self.model_dir = model_dir
        self.model_epoch = model_epoch
        self.device = device
        self.trainer = None
        self.is_loaded = False
        
        self.logger = logging.getLogger(__name__)
        
        # 自动加载模型
        self._load_model()
    
    def _load_model(self):
        """加载AI模型"""
        try:
            # 检查模型文件
            generator_path = os.path.join(self.model_dir, f"generator_epoch_{self.model_epoch}.pth")
            if not os.path.exists(generator_path):
                self.logger.warning(f"模型文件不存在: {generator_path}")
                return
            
            # 确定设备
            if self.device == "auto":
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                device = self.device
            
            # 初始化训练器
            self.trainer = LoomRecognitionTrainer(device=device)
            
            # 加载模型
            self.trainer.load_models(self.model_dir, self.model_epoch)
            
            self.is_loaded = True
            self.logger.info(f"AI模型加载成功，使用设备: {device}")
            
        except Exception as e:
            self.logger.error(f"AI模型加载失败: {e}")
            self.is_loaded = False
    
    def is_model_available(self) -> bool:
        """检查模型是否可用"""
        return self.is_loaded and self.trainer is not None
    
    def generate_image(self, source_image_path: str, output_path: str = None) -> Optional[str]:
        """
        使用AI模型生成图片
        
        Args:
            source_image_path: 源图片路径
            output_path: 输出图片路径（可选）
            
        Returns:
            生成的图片路径，失败返回None
        """
        if not self.is_model_available():
            self.logger.error("AI模型未加载")
            return None
        
        try:
            # 生成输出路径
            if output_path is None:
                timestamp = int(time.time() * 1000)
                source_name = Path(source_image_path).stem
                output_path = f"uploads/ai_generated_{timestamp}_{source_name}.png"
            
            # 确保输出目录存在
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # 生成图片
            result_path = self.trainer.generate_image(source_image_path, output_path)
            
            self.logger.info(f"AI生成成功: {result_path}")
            return result_path
            
        except Exception as e:
            self.logger.error(f"AI生成失败: {e}")
            return None
    
    def get_model_info(self) -> dict:
        """获取模型信息"""
        info = {
            "is_loaded": self.is_loaded,
            "model_dir": self.model_dir,
            "model_epoch": self.model_epoch,
            "device": self.device
        }
        
        if self.is_loaded and self.trainer:
            import torch
            info.update({
                "generator_params": sum(p.numel() for p in self.trainer.generator.parameters()),
                "discriminator_params": sum(p.numel() for p in self.trainer.discriminator.parameters()),
                "device_used": str(self.trainer.device)
            })
        
        return info

# 全局AI模型实例
ai_model = None

def get_ai_model() -> AIModelAPI:
    """获取AI模型实例（单例模式）"""
    global ai_model
    if ai_model is None:
        ai_model = AIModelAPI()
    return ai_model

def init_ai_model_api():
    """初始化AI模型API"""
    global ai_model
    ai_model = AIModelAPI()
    return ai_model

if __name__ == "__main__":
    # 测试AI模型API
    logging.basicConfig(level=logging.INFO)
    
    api = AIModelAPI()
    
    print("AI模型信息:")
    info = api.get_model_info()
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    if api.is_model_available():
        print("\n✓ AI模型加载成功")
    else:
        print("\n✗ AI模型加载失败") 