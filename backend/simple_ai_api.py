#!/usr/bin/env python3
"""
简化的AI模型API
使用训练好的简化模型生成织机识别图
"""

import os
import sys
import logging
import time
from pathlib import Path
from typing import Optional
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

class SimpleGenerator(nn.Module):
    """简化的生成器"""
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2)
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, 4, 2, 1),
            nn.Tanh()
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

class SimpleAIModelAPI:
    """
    简化的AI模型API封装
    用于在FastAPI中集成简化的AI模型
    """
    
    def __init__(self, 
                 model_dir: str = "../trained_models",
                 device: str = "auto"):
        
        self.model_dir = model_dir
        self.device = device
        self.generator = None
        self.is_loaded = False
        
        self.logger = logging.getLogger(__name__)
        
        # 自动加载模型
        self._load_model()
    
    def _load_model(self):
        """加载简化的AI模型"""
        try:
            # 检查模型文件
            generator_path = os.path.join(self.model_dir, "generator_simple.pth")
            if not os.path.exists(generator_path):
                self.logger.warning(f"模型文件不存在: {generator_path}")
                return
            
            # 确定设备
            if self.device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                device = self.device
            
            # 创建生成器
            self.generator = SimpleGenerator().to(device)
            
            # 加载模型权重
            self.generator.load_state_dict(torch.load(generator_path, map_location=device))
            self.generator.eval()
            
            self.is_loaded = True
            self.logger.info(f"简化AI模型加载成功，使用设备: {device}")
            
        except Exception as e:
            self.logger.error(f"简化AI模型加载失败: {e}")
            self.is_loaded = False
    
    def is_model_available(self) -> bool:
        """检查模型是否可用"""
        return self.is_loaded and self.generator is not None
    
    def generate_image(self, source_image_path: str, output_path: str = None) -> Optional[str]:
        """
        使用简化的AI模型生成图片
        
        Args:
            source_image_path: 源图片路径
            output_path: 输出图片路径（可选）
            
        Returns:
            生成的图片路径，失败返回None
        """
        if not self.is_model_available():
            self.logger.error("简化AI模型未加载")
            return None
        
        try:
            # 生成输出路径
            if output_path is None:
                timestamp = int(time.time() * 1000)
                source_name = Path(source_image_path).stem
                output_path = f"../uploads/simple_ai_generated_{timestamp}_{source_name}.jpg"
            
            # 确保输出目录存在
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # 加载和预处理图片
            transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])
            
            source_img = Image.open(source_image_path).convert('RGB')
            device = next(self.generator.parameters()).device
            source_tensor = transform(source_img).unsqueeze(0).to(device)
            
            # 生成图片
            with torch.no_grad():
                fake_target = self.generator(source_tensor)
                
                # 转换回图片格式
                fake_target = (fake_target + 1) / 2
                fake_target = fake_target.clamp(0, 1)
                fake_target = fake_target.cpu().squeeze(0).permute(1, 2, 0).numpy()
                
                # 转换为PIL图片
                fake_target = (fake_target * 255).astype(np.uint8)
                fake_target_img = Image.fromarray(fake_target)
                
                # 保存图片
                fake_target_img.save(output_path, "JPEG", quality=95)
            
            self.logger.info(f"图片生成成功: {output_path}")
            return output_path
            
        except Exception as e:
            self.logger.error(f"图片生成失败: {e}")
            return None
    
    def get_model_info(self) -> dict:
        """获取模型信息"""
        device = "unknown"
        if self.generator is not None:
            try:
                device = str(next(self.generator.parameters()).device)
            except:
                device = "unknown"
        
        return {
            "is_loaded": self.is_loaded,
            "model_dir": self.model_dir,
            "model_type": "simple_gan",
            "device": device
        }

# 单例模式
_simple_ai_model = None

def get_simple_ai_model() -> SimpleAIModelAPI:
    """获取简化的AI模型实例（单例模式）"""
    global _simple_ai_model
    if _simple_ai_model is None:
        _simple_ai_model = SimpleAIModelAPI()
    return _simple_ai_model 