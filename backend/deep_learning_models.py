"""
深度学习模型接口
统一管理U-Net、DeepLab等语义分割模型和预训练特征提取模型
"""

import numpy as np
import cv2
import logging
import time
from typing import Dict, Any, Optional, Tuple, List, Union
from pathlib import Path
import os
import json

# 尝试导入深度学习库
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torchvision import transforms, models
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logging.warning("PyTorch未安装，深度学习功能将不可用")

try:
    import tensorflow as tf
    from tensorflow import keras
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    logging.warning("TensorFlow未安装，深度学习功能将不可用")

logger = logging.getLogger(__name__)


class ModelConfig:
    """模型配置类"""
    
    def __init__(self, model_type: str, model_path: str = None, 
                 input_size: Tuple[int, int] = (512, 512),
                 num_classes: int = 2, device: str = "auto"):
        self.model_type = model_type
        self.model_path = model_path
        self.input_size = input_size
        self.num_classes = num_classes
        self.device = device
        
        # 自动检测设备
        if device == "auto":
            if TORCH_AVAILABLE and torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "model_type": self.model_type,
            "model_path": self.model_path,
            "input_size": self.input_size,
            "num_classes": self.num_classes,
            "device": self.device
        }


class BaseModel:
    """基础模型类"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = None
        self.is_loaded = False
        
    def load_model(self) -> bool:
        """加载模型"""
        raise NotImplementedError
    
    def predict(self, image: np.ndarray) -> np.ndarray:
        """预测"""
        raise NotImplementedError
    
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """预处理"""
        raise NotImplementedError
    
    def postprocess(self, prediction: np.ndarray) -> np.ndarray:
        """后处理"""
        raise NotImplementedError


class UNetModel(BaseModel):
    """U-Net模型实现"""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch未安装，无法使用U-Net模型")
    
    def load_model(self) -> bool:
        """加载U-Net模型"""
        try:
            # 这里应该加载预训练的U-Net模型
            # 为了演示，我们创建一个简单的U-Net结构
            self.model = self._create_unet()
            
            if self.config.model_path and os.path.exists(self.config.model_path):
                self.model.load_state_dict(torch.load(self.config.model_path, 
                                                     map_location=self.config.device))
            
            self.model.to(self.config.device)
            self.model.eval()
            self.is_loaded = True
            
            logger.info(f"U-Net模型加载成功: {self.config.device}")
            return True
            
        except Exception as e:
            logger.error(f"U-Net模型加载失败: {e}")
            return False
    
    def _create_unet(self) -> nn.Module:
        """创建U-Net模型结构"""
        class UNet(nn.Module):
            def __init__(self, in_channels=3, out_channels=1):
                super(UNet, self).__init__()
                
                # 编码器
                self.enc1 = self._make_layer(in_channels, 64)
                self.enc2 = self._make_layer(64, 128)
                self.enc3 = self._make_layer(128, 256)
                self.enc4 = self._make_layer(256, 512)
                
                # 解码器
                self.dec4 = self._make_layer(512, 256)
                self.dec3 = self._make_layer(256, 128)
                self.dec2 = self._make_layer(128, 64)
                self.dec1 = nn.Conv2d(64, out_channels, kernel_size=1)
                
            def _make_layer(self, in_channels, out_channels):
                return nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True)
                )
            
            def forward(self, x):
                # 编码
                e1 = self.enc1(x)
                e2 = self.enc2(F.max_pool2d(e1, 2))
                e3 = self.enc3(F.max_pool2d(e2, 2))
                e4 = self.enc4(F.max_pool2d(e3, 2))
                
                # 解码
                d4 = self.dec4(F.interpolate(e4, scale_factor=2, mode='bilinear'))
                d3 = self.dec3(F.interpolate(d4, scale_factor=2, mode='bilinear'))
                d2 = self.dec2(F.interpolate(d3, scale_factor=2, mode='bilinear'))
                d1 = self.dec1(d2)
                
                return torch.sigmoid(d1)
        
        return UNet()
    
    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        """预处理图像"""
        # 调整大小
        resized = cv2.resize(image, self.config.input_size)
        
        # 归一化
        normalized = resized.astype(np.float32) / 255.0
        
        # 转换为张量
        tensor = torch.from_numpy(normalized).permute(2, 0, 1).unsqueeze(0)
        
        return tensor.to(self.config.device)
    
    def predict(self, image: np.ndarray) -> np.ndarray:
        """预测分割"""
        if not self.is_loaded:
            if not self.load_model():
                raise RuntimeError("模型加载失败")
        
        with torch.no_grad():
            # 预处理
            input_tensor = self.preprocess(image)
            
            # 预测
            output = self.model(input_tensor)
            
            # 后处理
            prediction = self.postprocess(output)
            
            return prediction
    
    def postprocess(self, prediction: torch.Tensor) -> np.ndarray:
        """后处理预测结果"""
        # 转换为numpy
        pred_np = prediction.cpu().numpy()[0, 0]  # 移除批次和通道维度
        
        # 调整回原始大小
        if hasattr(self, '_original_size'):
            pred_np = cv2.resize(pred_np, self._original_size)
        
        # 二值化
        pred_np = (pred_np > 0.5).astype(np.uint8) * 255
        
        return pred_np


class DeepLabModel(BaseModel):
    """DeepLab模型实现"""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow未安装，无法使用DeepLab模型")
    
    def load_model(self) -> bool:
        """加载DeepLab模型"""
        try:
            # 这里应该加载预训练的DeepLab模型
            # 为了演示，我们创建一个简单的DeepLab结构
            self.model = self._create_deeplab()
            
            if self.config.model_path and os.path.exists(self.config.model_path):
                self.model.load_weights(self.config.model_path)
            
            logger.info("DeepLab模型加载成功")
            self.is_loaded = True
            return True
            
        except Exception as e:
            logger.error(f"DeepLab模型加载失败: {e}")
            return False
    
    def _create_deeplab(self) -> keras.Model:
        """创建DeepLab模型结构"""
        # 简化的DeepLab结构
        inputs = keras.Input(shape=(*self.config.input_size, 3))
        
        # 编码器（使用ResNet50作为backbone）
        base_model = keras.applications.ResNet50(
            include_top=False, 
            weights='imagenet',
            input_tensor=inputs
        )
        
        # 解码器
        x = base_model.output
        x = keras.layers.Conv2DTranspose(256, 3, strides=2, padding='same')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.ReLU()(x)
        
        x = keras.layers.Conv2DTranspose(128, 3, strides=2, padding='same')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.ReLU()(x)
        
        x = keras.layers.Conv2DTranspose(64, 3, strides=2, padding='same')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.ReLU()(x)
        
        outputs = keras.layers.Conv2D(self.config.num_classes, 1, activation='softmax')(x)
        
        model = keras.Model(inputs=inputs, outputs=outputs)
        return model
    
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """预处理图像"""
        # 调整大小
        resized = cv2.resize(image, self.config.input_size)
        
        # 归一化
        normalized = resized.astype(np.float32) / 255.0
        
        # 添加批次维度
        batched = np.expand_dims(normalized, axis=0)
        
        return batched
    
    def predict(self, image: np.ndarray) -> np.ndarray:
        """预测分割"""
        if not self.is_loaded:
            if not self.load_model():
                raise RuntimeError("模型加载失败")
        
        # 记录原始尺寸
        self._original_size = (image.shape[1], image.shape[0])
        
        # 预处理
        input_array = self.preprocess(image)
        
        # 预测
        prediction = self.model.predict(input_array)
        
        # 后处理
        result = self.postprocess(prediction)
        
        return result
    
    def postprocess(self, prediction: np.ndarray) -> np.ndarray:
        """后处理预测结果"""
        # 获取类别预测
        pred_class = np.argmax(prediction[0], axis=-1)
        
        # 调整回原始大小
        if hasattr(self, '_original_size'):
            pred_class = cv2.resize(pred_class.astype(np.uint8), self._original_size)
        
        # 转换为二值掩码
        mask = (pred_class > 0).astype(np.uint8) * 255
        
        return mask


class FeatureExtractor:
    """特征提取器"""
    
    def __init__(self, model_name: str = "resnet50", device: str = "auto"):
        self.model_name = model_name
        self.device = device
        self.model = None
        self.is_loaded = False
        
        if device == "auto":
            if TORCH_AVAILABLE and torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"
    
    def load_model(self) -> bool:
        """加载预训练模型"""
        try:
            if TORCH_AVAILABLE:
                # 加载预训练的ResNet模型
                if self.model_name == "resnet50":
                    self.model = models.resnet50(pretrained=True)
                elif self.model_name == "resnet101":
                    self.model = models.resnet101(pretrained=True)
                elif self.model_name == "efficientnet_b0":
                    self.model = models.efficientnet_b0(pretrained=True)
                else:
                    raise ValueError(f"不支持的模型: {self.model_name}")
                
                # 移除最后的分类层
                self.model = nn.Sequential(*list(self.model.children())[:-1])
                self.model.to(self.device)
                self.model.eval()
                
                logger.info(f"特征提取模型加载成功: {self.model_name}")
                self.is_loaded = True
                return True
            else:
                logger.error("PyTorch未安装，无法加载特征提取模型")
                return False
                
        except Exception as e:
            logger.error(f"特征提取模型加载失败: {e}")
            return False
    
    def extract_features(self, image: np.ndarray) -> np.ndarray:
        """提取特征"""
        if not self.is_loaded:
            if not self.load_model():
                raise RuntimeError("模型加载失败")
        
        with torch.no_grad():
            # 预处理
            preprocessed = self._preprocess_image(image)
            
            # 提取特征
            features = self.model(preprocessed)
            
            # 转换为numpy
            features_np = features.cpu().numpy()
            
            return features_np.flatten()  # 展平为一维向量
    
    def _preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """预处理图像"""
        # 转换为RGB
        if len(image.shape) == 3 and image.shape[2] == 3:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            rgb_image = image
        
        # 调整大小
        resized = cv2.resize(rgb_image, (224, 224))
        
        # 归一化
        normalized = resized.astype(np.float32) / 255.0
        
        # 标准化（ImageNet统计）
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        normalized = (normalized - mean) / std
        
        # 转换为张量
        tensor = torch.from_numpy(normalized).permute(2, 0, 1).unsqueeze(0)
        
        return tensor.to(self.device)


class DeepLearningModelManager:
    """深度学习模型管理器"""
    
    def __init__(self, models_dir: str = "models"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True, parents=True)
        
        self.models = {}
        self.feature_extractors = {}
        
        # 性能统计
        self.stats = {
            "total_predictions": 0,
            "total_time": 0,
            "average_time": 0
        }
        
        logger.info("深度学习模型管理器初始化完成")
    
    def load_segmentation_model(self, model_name: str, config: ModelConfig) -> bool:
        """加载分割模型"""
        try:
            if model_name.lower() == "unet":
                model = UNetModel(config)
            elif model_name.lower() == "deeplab":
                model = DeepLabModel(config)
            else:
                raise ValueError(f"不支持的分割模型: {model_name}")
            
            if model.load_model():
                self.models[model_name] = model
                logger.info(f"分割模型加载成功: {model_name}")
                return True
            else:
                return False
                
        except Exception as e:
            logger.error(f"加载分割模型失败: {model_name}, 错误: {e}")
            return False
    
    def load_feature_extractor(self, model_name: str) -> bool:
        """加载特征提取器"""
        try:
            extractor = FeatureExtractor(model_name)
            if extractor.load_model():
                self.feature_extractors[model_name] = extractor
                logger.info(f"特征提取器加载成功: {model_name}")
                return True
            else:
                return False
                
        except Exception as e:
            logger.error(f"加载特征提取器失败: {model_name}, 错误: {e}")
            return False
    
    def segment_image(self, image: np.ndarray, model_name: str) -> np.ndarray:
        """图像分割"""
        if model_name not in self.models:
            raise ValueError(f"模型未加载: {model_name}")
        
        start_time = time.time()
        
        try:
            model = self.models[model_name]
            result = model.predict(image)
            
            # 更新统计
            processing_time = time.time() - start_time
            self._update_stats(processing_time)
            
            return result
            
        except Exception as e:
            logger.error(f"图像分割失败: {model_name}, 错误: {e}")
            raise
    
    def extract_features(self, image: np.ndarray, model_name: str) -> np.ndarray:
        """提取特征"""
        if model_name not in self.feature_extractors:
            raise ValueError(f"特征提取器未加载: {model_name}")
        
        try:
            extractor = self.feature_extractors[model_name]
            features = extractor.extract_features(image)
            return features
            
        except Exception as e:
            logger.error(f"特征提取失败: {model_name}, 错误: {e}")
            raise
    
    def _update_stats(self, processing_time: float):
        """更新统计信息"""
        self.stats["total_predictions"] += 1
        self.stats["total_time"] += processing_time
        self.stats["average_time"] = (
            self.stats["total_time"] / self.stats["total_predictions"]
        )
    
    def get_available_models(self) -> Dict[str, List[str]]:
        """获取可用模型列表"""
        return {
            "segmentation": list(self.models.keys()),
            "feature_extraction": list(self.feature_extractors.keys())
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return self.stats.copy()
    
    def save_model_config(self, model_name: str, config: ModelConfig):
        """保存模型配置"""
        config_path = self.models_dir / f"{model_name}_config.json"
        with open(config_path, 'w') as f:
            json.dump(config.to_dict(), f, indent=2)
    
    def load_model_config(self, model_name: str) -> Optional[ModelConfig]:
        """加载模型配置"""
        config_path = self.models_dir / f"{model_name}_config.json"
        if config_path.exists():
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
            return ModelConfig(**config_dict)
        return None 