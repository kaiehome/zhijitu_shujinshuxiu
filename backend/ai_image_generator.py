import os
import json
import requests
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance
import logging
from typing import Tuple, Dict, Any, Optional, List
import base64
from io import BytesIO
import time

logger = logging.getLogger(__name__)

class AIImageGenerator:
    """基于大模型的AI图像生成器"""
    
    def __init__(self):
        self.deepseek_api_key = os.getenv('DEEPSEEK_API_KEY')
        self.qwen_api_key = os.getenv('QWEN_API_KEY')
        
        # API配置
        self.api_timeout = int(os.getenv('API_TIMEOUT', '30'))
        self.ai_enabled = os.getenv('AI_ENHANCED_MODE', 'true').lower() == 'true'
        
        if not self.deepseek_api_key and not self.qwen_api_key:
            logger.warning("未找到AI API密钥，将使用增强传统方法")
            self.ai_enabled = False
    
    def generate_professional_background(self, 
                                       width: int, 
                                       height: int, 
                                       subject_type: str = "熊猫") -> np.ndarray:
        """生成专业级背景图案"""
        try:
            if self.ai_enabled and self.qwen_api_key:
                # 使用AI生成背景
                ai_background = self._generate_ai_background(width, height, subject_type)
                if ai_background is not None:
                    return ai_background
            
            # 使用增强传统方法
            return self._generate_enhanced_traditional_background(width, height, subject_type)
            
        except Exception as e:
            logger.warning(f"专业背景生成失败: {str(e)}")
            return self._generate_basic_background(width, height)
    
    def enhance_weaving_style(self, 
                            image: np.ndarray, 
                            subject_mask: np.ndarray,
                            style: str = "蜀锦") -> np.ndarray:
        """AI增强织机风格化"""
        try:
            if self.ai_enabled:
                # 使用AI进行风格化处理
                styled_image = self._ai_style_transfer(image, subject_mask, style)
                if styled_image is not None:
                    return styled_image
            
            # 传统风格化处理
            return self._traditional_style_enhancement(image, subject_mask, style)
            
        except Exception as e:
            logger.warning(f"织机风格化失败: {str(e)}")
            return image
    
    def _generate_enhanced_traditional_background(self, 
                                                width: int, 
                                                height: int, 
                                                subject_type: str) -> np.ndarray:
        """生成增强的传统背景"""
        try:
            # 创建多层背景
            background = np.zeros((height, width, 3), dtype=np.uint8)
            
            # 第一层：渐变基础
            background = self._create_gradient_base(background, subject_type)
            
            # 第二层：花卉图案
            background = self._add_floral_patterns(background)
            
            # 第三层：织物纹理
            background = self._add_fabric_texture(background)
            
            return background
            
        except Exception as e:
            logger.warning(f"增强传统背景生成失败: {str(e)}")
            return self._generate_basic_background(width, height)
    
    def _create_gradient_base(self, background: np.ndarray, subject_type: str) -> np.ndarray:
        """创建渐变基础"""
        height, width = background.shape[:2]
        
        if subject_type == "熊猫":
            # 熊猫适配色彩：暖黄、棕色渐变
            colors = [
                (245, 222, 179),  # 小麦色
                (222, 184, 135),  # 浅棕色
                (205, 133, 63),   # 秘鲁色
                (160, 82, 45),    # 鞍褐色
            ]
        else:
            # 通用色彩
            colors = [
                (240, 230, 140),  # 卡其色
                (189, 183, 107),  # 深卡其色
                (128, 128, 0),    # 橄榄色
                (85, 107, 47),    # 深橄榄绿
            ]
        
        # 创建径向渐变
        center_x, center_y = width // 2, height // 2
        max_radius = np.sqrt(center_x**2 + center_y**2)
        
        for y in range(height):
            for x in range(width):
                # 计算到中心的距离
                distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                ratio = min(distance / max_radius, 1.0)
                
                # 颜色插值
                color_index = ratio * (len(colors) - 1)
                idx = int(color_index)
                if idx >= len(colors) - 1:
                    background[y, x] = colors[-1]
                else:
                    # 线性插值
                    t = color_index - idx
                    c1 = np.array(colors[idx])
                    c2 = np.array(colors[idx + 1])
                    background[y, x] = c1 * (1 - t) + c2 * t
        
        return background
    
    def _add_floral_patterns(self, background: np.ndarray) -> np.ndarray:
        """添加花卉图案"""
        height, width = background.shape[:2]
        
        # 使用OpenCV绘制花卉图案
        overlay = background.copy()
        
        # 花朵网格
        spacing = 50
        for y in range(spacing//2, height, spacing):
            for x in range(spacing//2, width, spacing):
                # 添加随机偏移
                offset_x = np.random.randint(-15, 15)
                offset_y = np.random.randint(-15, 15)
                flower_x = x + offset_x
                flower_y = y + offset_y
                
                if 0 <= flower_x < width and 0 <= flower_y < height:
                    self._draw_flower(overlay, flower_x, flower_y)
        
        # 混合原背景和花朵图案
        alpha = 0.4
        result = cv2.addWeighted(background, 1-alpha, overlay, alpha, 0)
        
        return result
    
    def _draw_flower(self, image: np.ndarray, center_x: int, center_y: int):
        """绘制单个花朵"""
        # 花朵颜色
        flower_colors = [
            (255, 215, 0),    # 金色
            (255, 140, 0),    # 深橙色
            (178, 34, 34),    # 火砖红
            (139, 69, 19),    # 鞍褐色
        ]
        
        color = flower_colors[np.random.randint(0, len(flower_colors))]
        
        # 绘制花瓣（5瓣）
        petal_length = 15
        for angle in range(0, 360, 72):
            rad = np.radians(angle)
            end_x = center_x + int(petal_length * np.cos(rad))
            end_y = center_y + int(petal_length * np.sin(rad))
            
            # 绘制椭圆花瓣
            axes = (6, 12)
            cv2.ellipse(image, (center_x, center_y), axes, angle, 0, 180, color, -1)
        
        # 绘制花心
        cv2.circle(image, (center_x, center_y), 4, (139, 69, 19), -1)
    
    def _add_fabric_texture(self, background: np.ndarray) -> np.ndarray:
        """添加织物纹理"""
        # 创建微妙的织物纹理效果
        noise = np.random.randint(-8, 9, background.shape, dtype=np.int16)
        textured = background.astype(np.int16) + noise
        textured = np.clip(textured, 0, 255).astype(np.uint8)
        
        # 混合原图和纹理
        alpha = 0.1  # 很轻的纹理
        result = cv2.addWeighted(background, 1-alpha, textured, alpha, 0)
        
        return result
    
    def _generate_basic_background(self, width: int, height: int) -> np.ndarray:
        """生成基础背景（回退方案）"""
        background = np.zeros((height, width, 3), dtype=np.uint8)
        base_color = np.array([222, 184, 135])  # 浅棕色
        
        for y in range(height):
            for x in range(width):
                background[y, x] = base_color
        
        return background
    
    def _generate_ai_background(self, width: int, height: int, subject_type: str) -> Optional[np.ndarray]:
        """使用真实AI API生成背景图案 - 智能提示词优化"""
        try:
            # 构建专业背景生成提示词
            base_prompt = self._build_professional_prompt(subject_type)
            
            # 为AI生成添加额外的专业软件特征描述
            enhanced_prompt = (
                "专业织机识别软件输出效果，工业级图像处理标准，" +
                base_prompt +
                "，必须完全平面化，禁止渐变和阴影，禁止复杂纹理，"
                "边缘必须锐利清晰，颜色必须纯净饱和，"
                "符合织机设备识别要求的标准化图案"
            )
            
            # 优先使用通义千问的文生图API
            if self.qwen_api_key:
                generated_image = self._call_qwen_image_generation(enhanced_prompt, width, height)
                if generated_image is not None:
                    logger.info("通义千问背景生成成功")
                    return generated_image
            
            # 备用：使用DeepSeek API
            if self.deepseek_api_key:
                generated_image = self._call_deepseek_image_generation(prompt, width, height)
                if generated_image is not None:
                    logger.info("DeepSeek背景生成成功")
                    return generated_image
            
            # 最后回退到增强传统方法
            logger.warning("AI API不可用，使用增强传统方法")
            return self._generate_enhanced_traditional_background(width, height, subject_type)
            
        except Exception as e:
            logger.warning(f"AI背景生成失败: {str(e)}")
            return self._generate_enhanced_traditional_background(width, height, subject_type)
    
    def _build_professional_prompt(self, subject_type: str) -> str:
        """构建专业AI背景生成提示词 - 模拟织机识别软件效果"""
        # 核心：完全模拟专业织机识别软件的视觉特征
        base_prompt = "专业织机识别软件风格，工业化图像处理效果，"
        
        if subject_type == "熊猫":
            prompt = (base_prompt + 
                     "竹叶花卉背景图案，极简化处理，只使用4-6种纯色，"
                     "金黄色#FFD700、橙红色#FF4500、深绿色#228B22、棕色#8B4513，"
                     "完全平面化，无渐变无阴影无纹理，锐利边缘分割，"
                     "卡通风格简化图案，类似丝网印刷或版画效果，"
                     "高饱和度纯色填充，边界清晰，工业标准化处理，"
                     "织机可精确识别的简单几何图案")
        elif subject_type == "动物":
            prompt = (base_prompt + 
                     "花卉装饰图案，极简色彩，4-5种主色调，"
                     "平面化设计，高对比度，无细节纹理，"
                     "锐利边缘，工业化色彩处理，织机识别标准")
        else:
            # 对于unknown或其他类型，使用通用的极简风格
            prompt = (base_prompt + 
                     "抽象几何图案，极简色彩，3-4种主色调，"
                     "红色#DC143C、黄色#FFD700、蓝色#4169E1、绿色#32CD32，"
                     "完全平面化，无渐变，高对比度边缘，"
                     "工业化处理，织机识别专用，简单几何形状")
        
        return prompt
    
    def _analyze_image_for_prompt(self, image: np.ndarray) -> str:
        """分析图像内容，生成更精准的提示词"""
        try:
            # 分析图像的主要颜色
            dominant_colors = self._extract_dominant_colors(image)
            
            # 分析图像复杂度
            complexity = self._analyze_complexity(image)
            
            # 根据分析结果调整提示词
            if complexity > 0.7:
                style_modifier = "极度简化，强制平面化，去除所有细节，"
            elif complexity > 0.4:
                style_modifier = "中度简化，平面化处理，减少细节，"
            else:
                style_modifier = "轻度简化，保持基本形状，"
            
            return style_modifier
            
        except Exception as e:
            logger.warning(f"图像分析失败: {str(e)}")
            return "标准简化处理，"
    
    def _extract_dominant_colors(self, image: np.ndarray, k: int = 5) -> List[Tuple[int, int, int]]:
        """提取图像主要颜色"""
        try:
            from sklearn.cluster import KMeans
            
            # 重塑图像数据
            data = image.reshape((-1, 3))
            
            # K-means聚类
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(data)
            
            # 获取主要颜色
            colors = kmeans.cluster_centers_.astype(int)
            return [tuple(color) for color in colors]
            
        except Exception as e:
            logger.warning(f"颜色提取失败: {str(e)}")
            return [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    
    def _analyze_complexity(self, image: np.ndarray) -> float:
        """分析图像复杂度"""
        try:
            import cv2
            
            # 转换为灰度图
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            # 计算边缘密度
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / edges.size
            
            # 计算颜色变化
            unique_colors = len(np.unique(image.reshape(-1, 3), axis=0))
            color_complexity = min(1.0, unique_colors / 1000)
            
            # 综合复杂度
            complexity = (edge_density * 0.6 + color_complexity * 0.4)
            
            return complexity
            
        except Exception as e:
            logger.warning(f"复杂度分析失败: {str(e)}")
            return 0.5
    
    def _call_qwen_image_generation(self, prompt: str, width: int, height: int) -> Optional[np.ndarray]:
        """调用通义千问文生图API"""
        try:
            import dashscope
            from dashscope import ImageSynthesis
            
            # 配置API
            dashscope.api_key = self.qwen_api_key
            
            logger.info(f"正在调用通义千问API生成背景: {prompt[:50]}...")
            
            # 调用文生图API - 使用API支持的标准尺寸
            # 通义千问支持的尺寸: 1024*1024, 720*1280, 1280*720, 768*1152
            api_size = '1024*1024'  # 使用默认的1024x1024尺寸
            if width >= height:
                api_size = '1280*720'  # 横向图片
            else:
                api_size = '720*1280'  # 纵向图片
                
            response = ImageSynthesis.call(
                model='wanx-v1',
                prompt=prompt,
                size=api_size,
                n=1,
                style='<photography>'
            )
            
            # 添加详细的调试信息
            logger.info(f"通义千问API响应状态: {response.status_code}")
            logger.info(f"通义千问API响应内容: {response}")
            
            if response.status_code == 200 and response.output and len(response.output.results) > 0:
                # 获取生成的图片URL
                image_url = response.output.results[0]['url']
                logger.info(f"获取到图片URL: {image_url}")
                
                # 下载图片
                generated_image = self._download_image_from_url(image_url)
                
                if generated_image is not None:
                    # 调整到目标尺寸
                    resized_image = cv2.resize(generated_image, (width, height))
                    return resized_image
            else:
                logger.warning(f"通义千问API响应异常 - 状态码: {response.status_code}")
                if hasattr(response, 'message'):
                    logger.warning(f"错误信息: {response.message}")
                if hasattr(response, 'output'):
                    logger.warning(f"输出内容: {response.output}")
                    
            logger.warning("通义千问API未返回有效图片")
            return None
            
        except ImportError:
            logger.warning("dashscope库未安装，请运行: pip install dashscope")
            return None
        except Exception as e:
            logger.warning(f"通义千问API调用失败: {str(e)}")
            logger.warning(f"错误类型: {type(e)}")
            return None
    
    def _call_deepseek_image_generation(self, prompt: str, width: int, height: int) -> Optional[np.ndarray]:
        """调用DeepSeek文生图API"""
        try:
            import requests
            
            logger.info(f"正在调用DeepSeek API生成背景: {prompt[:50]}...")
            
            # DeepSeek API endpoint (需要根据实际API文档调整)
            url = "https://api.deepseek.com/v1/images/generations"
            
            headers = {
                "Authorization": f"Bearer {self.deepseek_api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "prompt": prompt,
                "size": f"{width}x{height}",
                "n": 1,
                "response_format": "url"
            }
            
            response = requests.post(url, headers=headers, json=data, timeout=self.api_timeout)
            
            if response.status_code == 200:
                result = response.json()
                image_url = result['data'][0]['url']
                
                # 下载图片
                generated_image = self._download_image_from_url(image_url)
                
                if generated_image is not None:
                    # 调整到目标尺寸
                    resized_image = cv2.resize(generated_image, (width, height))
                    return resized_image
                    
            logger.warning(f"DeepSeek API调用失败: HTTP {response.status_code}")
            return None
            
        except Exception as e:
            logger.warning(f"DeepSeek API调用失败: {str(e)}")
            return None
    
    def _download_image_from_url(self, image_url: str) -> Optional[np.ndarray]:
        """从URL下载图片"""
        try:
            import requests
            from io import BytesIO
            
            response = requests.get(image_url, timeout=30)
            response.raise_for_status()
            
            # 转换为numpy数组
            image_data = BytesIO(response.content)
            pil_image = Image.open(image_data)
            
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
                
            return np.array(pil_image)
            
        except Exception as e:
            logger.warning(f"图片下载失败: {str(e)}")
            return None
    
    def _ai_style_transfer(self, 
                          image: np.ndarray, 
                          subject_mask: np.ndarray, 
                          style: str) -> Optional[np.ndarray]:
        """AI风格迁移"""
        try:
            # 这里可以调用风格迁移API
            # 暂时使用增强的传统方法
            return self._traditional_style_enhancement(image, subject_mask, style)
            
        except Exception as e:
            logger.warning(f"AI风格迁移失败: {str(e)}")
            return None
    
    def _traditional_style_enhancement(self, 
                                     image: np.ndarray, 
                                     subject_mask: np.ndarray, 
                                     style: str) -> np.ndarray:
        """传统风格增强"""
        try:
            enhanced = image.copy()
            
            # 增强对比度
            enhanced = cv2.convertScaleAbs(enhanced, alpha=1.2, beta=10)
            
            # 增强饱和度
            hsv = cv2.cvtColor(enhanced, cv2.COLOR_RGB2HSV)
            hsv[:, :, 1] = cv2.multiply(hsv[:, :, 1], 1.3)
            enhanced = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
            
            return enhanced
            
        except Exception as e:
            logger.warning(f"传统风格增强失败: {str(e)}")
            return image 