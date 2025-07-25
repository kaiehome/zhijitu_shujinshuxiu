# image_to_image_generator.py
# 图生图生成器 - 纯图生图功能，无需文字输入

import cv2
import numpy as np
import os
import logging
from pathlib import Path
from typing import Optional, Tuple, List, Dict
from datetime import datetime
import json

# 导入现有的处理模块
from image_processor import SichuanBrocadeProcessor
from ai_segmentation import AISegmenter
from feature_detector import FeatureDetector
from structural_generator_core import StructuralCore

logger = logging.getLogger(__name__)

class ImageToImageGenerator:
    """
    图生图生成器
    基于现有算法实现纯图生图功能，无需文字输入
    """
    
    def __init__(self, 
                 color_count: int = 16,
                 use_ai_segmentation: bool = True,
                 use_feature_detection: bool = True,
                 style_preset: str = "weaving_machine"):
        
        self.color_count = color_count
        self.use_ai_segmentation = use_ai_segmentation
        self.use_feature_detection = use_feature_detection
        self.style_preset = style_preset
        
        # 初始化处理器
        self.brocade_processor = SichuanBrocadeProcessor()
        self.structural_core = StructuralCore()
        
        # 初始化AI模块（可选）
        self.ai_segmenter = None
        self.feature_detector = None
        
        if use_ai_segmentation:
            try:
                self.ai_segmenter = AISegmenter()
                logger.info("AI分割模块初始化成功")
            except Exception as e:
                logger.warning(f"AI分割模块初始化失败: {e}")
        
        if use_feature_detection:
            try:
                self.feature_detector = FeatureDetector()
                logger.info("特征检测模块初始化成功")
            except Exception as e:
                logger.warning(f"特征检测模块初始化失败: {e}")
        
        # 预定义风格预设
        self.style_presets = {
            "weaving_machine": {
                "color_count": 16,
                "edge_enhancement": True,
                "noise_reduction": True,
                "saturation_boost": 1.3,
                "contrast_boost": 1.2,
                "smooth_kernel": 3,
                "quantization_method": "kmeans_lab"
            },
            "embroidery": {
                "color_count": 20,
                "edge_enhancement": True,
                "noise_reduction": False,
                "saturation_boost": 1.5,
                "contrast_boost": 1.4,
                "smooth_kernel": 5,
                "quantization_method": "structural"
            },
            "pixel_art": {
                "color_count": 12,
                "edge_enhancement": False,
                "noise_reduction": True,
                "saturation_boost": 1.8,
                "contrast_boost": 1.6,
                "smooth_kernel": 0,
                "quantization_method": "force_limited"
            }
        }
    
    def generate_loom_recognition_image(self, 
                                      source_image_path: str,
                                      output_path: str = None,
                                      style_preset: str = None) -> Optional[str]:
        """
        生成织机识别图 - 纯图生图
        
        Args:
            source_image_path: 源图片路径
            output_path: 输出图片路径（可选）
            style_preset: 风格预设（可选）
            
        Returns:
            生成的图片路径，失败返回None
        """
        try:
            logger.info(f"开始图生图处理: {source_image_path}")
            
            # 加载源图片
            source_image = cv2.imread(source_image_path)
            if source_image is None:
                logger.error(f"无法加载源图片: {source_image_path}")
                return None
            
            # 获取风格预设
            preset = self._get_style_preset(style_preset)
            
            # 执行图生图处理
            result_image = self._process_image_to_image(source_image, preset)
            
            # 生成输出路径
            if output_path is None:
                timestamp = int(time.time() * 1000)
                source_name = Path(source_image_path).stem
                output_path = f"uploads/img2img_{timestamp}_{source_name}.jpg"
            
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # 保存结果
            cv2.imwrite(output_path, result_image)
            logger.info(f"图生图处理完成: {output_path}")
            
            return output_path
            
        except Exception as e:
            logger.error(f"图生图处理失败: {e}")
            return None
    
    def _get_style_preset(self, style_preset: str = None) -> Dict:
        """获取风格预设"""
        preset_name = style_preset or self.style_preset
        if preset_name in self.style_presets:
            return self.style_presets[preset_name].copy()
        else:
            logger.warning(f"未知风格预设: {preset_name}，使用默认预设")
            return self.style_presets["weaving_machine"].copy()
    
    def _process_image_to_image(self, source_image: np.ndarray, preset: Dict) -> np.ndarray:
        """
        执行图生图处理流程
        """
        logger.info("开始图生图处理流程")
        
        # 1. 预处理
        processed_image = self._preprocess_image(source_image, preset)
        
        # 2. AI分割（可选）
        if self.use_ai_segmentation and self.ai_segmenter:
            processed_image = self._apply_ai_segmentation(processed_image)
        
        # 3. 特征检测（可选）
        if self.use_feature_detection and self.feature_detector:
            processed_image = self._apply_feature_detection(processed_image)
        
        # 4. 颜色量化
        processed_image = self._apply_color_quantization(processed_image, preset)
        
        # 5. 后处理
        processed_image = self._postprocess_image(processed_image, preset)
        
        logger.info("图生图处理流程完成")
        return processed_image
    
    def _preprocess_image(self, image: np.ndarray, preset: Dict) -> np.ndarray:
        """预处理图像"""
        logger.info("执行图像预处理")
        
        # 调整大小（保持宽高比）
        target_size = (512, 512)
        h, w = image.shape[:2]
        scale = min(target_size[0] / w, target_size[1] / h)
        new_size = (int(w * scale), int(h * scale))
        resized = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
        
        # 噪声减少
        if preset.get("noise_reduction", True):
            resized = cv2.bilateralFilter(resized, 9, 75, 75)
        
        # 饱和度增强
        saturation_boost = preset.get("saturation_boost", 1.0)
        if saturation_boost != 1.0:
            hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation_boost, 0, 255)
            resized = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        # 对比度增强
        contrast_boost = preset.get("contrast_boost", 1.0)
        if contrast_boost != 1.0:
            lab = cv2.cvtColor(resized, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=contrast_boost, tileGridSize=(8, 8))
            l = clahe.apply(l)
            lab = cv2.merge([l, a, b])
            resized = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        return resized
    
    def _apply_ai_segmentation(self, image: np.ndarray) -> np.ndarray:
        """应用AI分割"""
        try:
            logger.info("应用AI分割")
            mask = self.ai_segmenter.segment(image)
            if mask is not None:
                # 使用分割掩码增强前景
                segmented = cv2.bitwise_and(image, image, mask=mask)
                # 混合原图和分割结果
                result = cv2.addWeighted(image, 0.7, segmented, 0.3, 0)
                return result
        except Exception as e:
            logger.warning(f"AI分割失败: {e}")
        
        return image
    
    def _apply_feature_detection(self, image: np.ndarray) -> np.ndarray:
        """应用特征检测"""
        try:
            logger.info("应用特征检测")
            features = self.feature_detector.detect(image)
            if features:
                # 在特征点周围增强对比度
                result = image.copy()
                for feature in features:
                    x, y = int(feature[0]), int(feature[1])
                    if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
                        # 在特征点周围创建增强区域
                        cv2.circle(result, (x, y), 5, (255, 255, 255), -1)
                return result
        except Exception as e:
            logger.warning(f"特征检测失败: {e}")
        
        return image
    
    def _apply_color_quantization(self, image: np.ndarray, preset: Dict) -> np.ndarray:
        """应用颜色量化"""
        logger.info("执行颜色量化")
        
        method = preset.get("quantization_method", "kmeans_lab")
        color_count = preset.get("color_count", self.color_count)
        
        if method == "structural":
            # 使用结构化分色
            result = self.structural_core.structural_color_separation(
                image, color_count, n_segments=400, compactness=12
            )
            return result["structural_image"]
        
        elif method == "force_limited":
            # 使用强制限制颜色
            pixels = image.reshape(-1, 3)
            kmeans = KMeans(n_clusters=color_count, random_state=42, n_init=10)
            labels = kmeans.fit_predict(pixels)
            colors = kmeans.cluster_centers_.astype(np.uint8)
            result = colors[labels].reshape(image.shape)
            return result
        
        else:
            # 默认K-means LAB色彩空间
            lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            pixels = lab_image.reshape(-1, 3)
            kmeans = KMeans(n_clusters=color_count, random_state=42, n_init=10)
            labels = kmeans.fit_predict(pixels)
            colors = kmeans.cluster_centers_.astype(np.uint8)
            quantized_lab = colors[labels].reshape(lab_image.shape)
            result = cv2.cvtColor(quantized_lab, cv2.COLOR_LAB2BGR)
            return result
    
    def _postprocess_image(self, image: np.ndarray, preset: Dict) -> np.ndarray:
        """后处理图像"""
        logger.info("执行图像后处理")
        
        # 边缘增强
        if preset.get("edge_enhancement", True):
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            edges = cv2.Laplacian(gray, cv2.CV_64F, ksize=3)
            edges = np.uint8(np.absolute(edges))
            enhanced = cv2.addWeighted(gray, 1.5, edges, 0.5, 0)
            image = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
        
        # 边界平滑
        smooth_kernel = preset.get("smooth_kernel", 3)
        if smooth_kernel > 0:
            image = cv2.bilateralFilter(image, smooth_kernel, 75, 75)
        
        # 最终颜色优化
        image = self._optimize_final_colors(image, preset)
        
        return image
    
    def _optimize_final_colors(self, image: np.ndarray, preset: Dict) -> np.ndarray:
        """最终颜色优化"""
        # 确保颜色数量符合要求
        color_count = preset.get("color_count", self.color_count)
        unique_colors = np.unique(image.reshape(-1, 3), axis=0)
        
        if len(unique_colors) > color_count:
            # 重新量化到指定颜色数
            pixels = image.reshape(-1, 3)
            kmeans = KMeans(n_clusters=color_count, random_state=42, n_init=10)
            labels = kmeans.fit_predict(pixels)
            colors = kmeans.cluster_centers_.astype(np.uint8)
            image = colors[labels].reshape(image.shape)
        
        return image
    
    def get_available_styles(self) -> List[str]:
        """获取可用的风格预设"""
        return list(self.style_presets.keys())
    
    def add_custom_style(self, style_name: str, style_config: Dict):
        """添加自定义风格"""
        self.style_presets[style_name] = style_config
        logger.info(f"添加自定义风格: {style_name}")
    
    def get_processing_info(self) -> Dict:
        """获取处理信息"""
        return {
            "color_count": self.color_count,
            "use_ai_segmentation": self.use_ai_segmentation,
            "use_feature_detection": self.use_feature_detection,
            "current_style": self.style_preset,
            "available_styles": self.get_available_styles()
        }

# 导入必要的模块
import time
from sklearn.cluster import KMeans 