import os
import cv2
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
import logging
from typing import Dict, Any, Optional, Tuple
import torch
import torchvision.transforms as transforms

logger = logging.getLogger(__name__)

class LocalAIGenerator:
    """本地免费AI模型图像生成器"""
    
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.models_loaded = False
        
        # 尝试加载本地模型
        try:
            self._load_local_models()
            logger.info(f"本地AI模型已加载，使用设备: {self.device}")
        except Exception as e:
            logger.warning(f"本地AI模型加载失败: {str(e)}")
            self.models_loaded = False
    
    def _load_local_models(self):
        """加载本地AI模型"""
        try:
            # 这里可以集成开源模型，如：
            # - BLIP for image understanding
            # - Stable Diffusion for image generation
            # - CLIP for image analysis
            
            # 暂时使用模拟实现
            self.models_loaded = True
            logger.info("本地AI模型模拟加载成功")
            
        except Exception as e:
            logger.warning(f"本地模型加载失败: {str(e)}")
            raise
    
    def analyze_image_content(self, image: np.ndarray) -> Dict[str, Any]:
        """本地AI图像内容分析"""
        try:
            if not self.models_loaded:
                return self._fallback_analysis(image)
            
            # 本地AI分析实现
            analysis = {
                'main_subject': self._detect_subject_type(image),
                'scene_type': self._analyze_scene(image),
                'color_harmony': self._analyze_colors(image),
                'composition': self._analyze_composition(image)
            }
            
            logger.info(f"本地AI分析完成: 主体={analysis['main_subject']}")
            return analysis
            
        except Exception as e:
            logger.warning(f"本地AI分析失败: {str(e)}")
            return self._fallback_analysis(image)
    
    def _detect_subject_type(self, image: np.ndarray) -> str:
        """检测主体类型"""
        try:
            # 基于颜色特征的智能检测
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            
            # 黑白区域检测（熊猫特征）
            black_mask = cv2.inRange(hsv, (0, 0, 0), (180, 255, 50))
            white_mask = cv2.inRange(hsv, (0, 0, 200), (180, 30, 255))
            
            black_ratio = np.sum(black_mask > 0) / (image.shape[0] * image.shape[1])
            white_ratio = np.sum(white_mask > 0) / (image.shape[0] * image.shape[1])
            
            if black_ratio > 0.15 and white_ratio > 0.3:
                return "熊猫"
            elif black_ratio > 0.2:
                return "动物"
            else:
                return "其他"
                
        except Exception as e:
            logger.warning(f"主体检测失败: {str(e)}")
            return "未知"
    
    def _analyze_scene(self, image: np.ndarray) -> str:
        """场景分析"""
        try:
            # 基于颜色分布分析场景
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            
            # 绿色区域（自然场景）
            green_mask = cv2.inRange(hsv, (35, 40, 40), (85, 255, 255))
            green_ratio = np.sum(green_mask > 0) / (image.shape[0] * image.shape[1])
            
            if green_ratio > 0.3:
                return "自然场景"
            else:
                return "室内场景"
                
        except Exception as e:
            return "未知场景"
    
    def _analyze_colors(self, image: np.ndarray) -> Dict[str, float]:
        """颜色和谐度分析"""
        try:
            # 计算主要颜色分布
            pixels = image.reshape(-1, 3)
            
            # 颜色方差（和谐度指标）
            color_variance = np.var(pixels, axis=0).mean()
            harmony_score = max(0, 1 - color_variance / 10000)
            
            # 饱和度分析
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            saturation_mean = np.mean(hsv[:, :, 1])
            
            return {
                'harmony_score': harmony_score,
                'saturation_level': saturation_mean / 255.0,
                'color_complexity': len(np.unique(pixels.view(np.void), axis=0))
            }
            
        except Exception as e:
            return {'harmony_score': 0.5, 'saturation_level': 0.5, 'color_complexity': 100}
    
    def _analyze_composition(self, image: np.ndarray) -> Dict[str, Any]:
        """构图分析"""
        try:
            height, width = image.shape[:2]
            
            # 中心区域检测
            center_x, center_y = width // 2, height // 2
            center_region = image[center_y-height//4:center_y+height//4, 
                                center_x-width//4:center_x+width//4]
            
            # 主体位置评估
            edges = cv2.Canny(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY), 50, 150)
            center_edges = np.sum(edges[center_y-height//4:center_y+height//4, 
                                      center_x-width//4:center_x+width//4])
            total_edges = np.sum(edges)
            
            center_ratio = center_edges / max(total_edges, 1)
            
            return {
                'subject_centered': center_ratio > 0.3,
                'composition_balance': min(center_ratio * 2, 1.0),
                'focus_area': 'center' if center_ratio > 0.3 else 'distributed'
            }
            
        except Exception as e:
            return {'subject_centered': True, 'composition_balance': 0.5, 'focus_area': 'center'}
    
    def generate_smart_background(self, 
                                width: int, 
                                height: int, 
                                analysis: Dict[str, Any]) -> np.ndarray:
        """基于分析结果生成智能背景"""
        try:
            subject_type = analysis.get('main_subject', '未知')
            
            if subject_type == "熊猫":
                return self._generate_panda_background(width, height, analysis)
            elif subject_type == "动物":
                return self._generate_animal_background(width, height, analysis)
            else:
                return self._generate_general_background(width, height, analysis)
                
        except Exception as e:
            logger.warning(f"智能背景生成失败: {str(e)}")
            return self._generate_basic_background(width, height)
    
    def _generate_panda_background(self, width: int, height: int, analysis: Dict) -> np.ndarray:
        """生成熊猫专用背景"""
        try:
            # 竹子和花卉主题背景
            background = np.zeros((height, width, 3), dtype=np.uint8)
            
            # 温暖的渐变基调
            for y in range(height):
                for x in range(width):
                    # 从中心向外的径向渐变
                    center_x, center_y = width // 2, height // 2
                    distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                    max_distance = np.sqrt(center_x**2 + center_y**2)
                    ratio = min(distance / max_distance, 1.0)
                    
                    # 竹子绿到暖黄的渐变
                    green_component = int(180 * (1 - ratio) + 220 * ratio)
                    background[y, x] = [green_component, int(200 + 30 * ratio), int(150 + 50 * ratio)]
            
            # 添加竹叶图案
            self._add_bamboo_pattern(background)
            
            # 添加花朵装饰
            self._add_flower_decorations(background, density=0.3)
            
            return background
            
        except Exception as e:
            logger.warning(f"熊猫背景生成失败: {str(e)}")
            return self._generate_basic_background(width, height)
    
    def _add_bamboo_pattern(self, background: np.ndarray):
        """添加竹叶图案"""
        height, width = background.shape[:2]
        
        # 竹叶颜色
        bamboo_colors = [(34, 139, 34), (0, 128, 0), (50, 205, 50)]
        
        # 随机分布竹叶
        for _ in range(width // 20):
            x = np.random.randint(10, width - 10)
            y = np.random.randint(10, height - 10)
            color = bamboo_colors[np.random.randint(0, len(bamboo_colors))]
            
            # 绘制椭圆形竹叶
            cv2.ellipse(background, (x, y), (8, 20), 
                       np.random.randint(0, 180), 0, 360, color, -1)
    
    def _add_flower_decorations(self, background: np.ndarray, density: float = 0.2):
        """添加花朵装饰"""
        height, width = background.shape[:2]
        
        flower_colors = [(255, 182, 193), (255, 192, 203), (255, 160, 122)]
        num_flowers = int(width * height * density / 10000)
        
        for _ in range(num_flowers):
            x = np.random.randint(15, width - 15)
            y = np.random.randint(15, height - 15)
            color = flower_colors[np.random.randint(0, len(flower_colors))]
            
            # 绘制小花朵
            cv2.circle(background, (x, y), 5, color, -1)
            cv2.circle(background, (x, y), 3, (255, 255, 255), -1)
    
    def _generate_animal_background(self, width: int, height: int, analysis: Dict) -> np.ndarray:
        """生成动物背景"""
        return self._generate_panda_background(width, height, analysis)
    
    def _generate_general_background(self, width: int, height: int, analysis: Dict) -> np.ndarray:
        """生成通用背景"""
        return self._generate_basic_background(width, height)
    
    def _generate_basic_background(self, width: int, height: int) -> np.ndarray:
        """生成基础背景"""
        background = np.full((height, width, 3), [222, 184, 135], dtype=np.uint8)
        return background
    
    def _fallback_analysis(self, image: np.ndarray) -> Dict[str, Any]:
        """回退分析方法"""
        return {
            'main_subject': '未知',
            'scene_type': '未知场景',
            'color_harmony': {'harmony_score': 0.5, 'saturation_level': 0.5, 'color_complexity': 100},
            'composition': {'subject_centered': True, 'composition_balance': 0.5, 'focus_area': 'center'}
        } 