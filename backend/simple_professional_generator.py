#!/usr/bin/env python3
"""
极简专业织机识别图像生成器
完全模拟专业软件的真实处理方式：保持主体可识别，只做颜色简化和边缘增强
"""

import os
import cv2
import numpy as np
from PIL import Image, ImageEnhance
from sklearn.cluster import KMeans
import logging
from typing import Tuple
import time

logger = logging.getLogger(__name__)

class SimpleProfessionalGenerator:
    """极简专业织机识别图像生成器"""
    
    def __init__(self):
        """初始化"""
        self.config = {
            # 极简配置，模拟专业软件的真实参数
            "color_count": 8,           # 8种主要颜色
            "saturation_boost": 1.8,    # 1.8倍饱和度
            "contrast_boost": 1.4,      # 1.4倍对比度
            "edge_enhancement": 1.2,    # 轻度边缘增强
            "smoothness": 3,            # 轻度平滑
        }
        logger.info("极简专业织机生成器已初始化")
    
    def generate_professional_image(self, input_path: str, job_id: str) -> Tuple[str, str, float]:
        """生成专业织机识别图像 - 极简版本"""
        start_time = time.time()
        
        try:
            # 1. 加载原图
            image = self._load_image(input_path)
            logger.info(f"原图尺寸: {image.shape}")
            
            # 2. 极简专业处理流水线
            professional_image = self._simple_professional_pipeline(image)
            
            # 3. 保存结果
            output_dir = f"outputs/{job_id}"
            os.makedirs(output_dir, exist_ok=True)
            
            professional_path = os.path.join(output_dir, f"{job_id}_simple_professional.png")
            comparison_path = os.path.join(output_dir, f"{job_id}_simple_comparison.png")
            
            # 保存专业图像
            self._save_image(professional_image, professional_path)
            
            # 创建对比图
            comparison = self._create_comparison(image, professional_image)
            self._save_image(comparison, comparison_path)
            
            processing_time = time.time() - start_time
            logger.info(f"极简专业图像生成完成，耗时: {processing_time:.2f}秒")
            
            return professional_path, comparison_path, processing_time
            
        except Exception as e:
            raise Exception(f"极简专业图像生成失败: {str(e)}")
    
    def _load_image(self, input_path: str) -> np.ndarray:
        """加载图像"""
        # 确保使用绝对路径
        if not os.path.isabs(input_path):
            # 如果是相对路径，转换为绝对路径
            input_path = os.path.abspath(input_path)
        
        logger.info(f"尝试加载图像: {input_path}")
        
        # 检查文件是否存在
        if not os.path.exists(input_path):
            raise ValueError(f"图像文件不存在: {input_path}")
        
        # 检查文件大小
        file_size = os.path.getsize(input_path)
        if file_size == 0:
            raise ValueError(f"图像文件为空: {input_path}")
        
        logger.info(f"图像文件大小: {file_size} bytes")
        
        # 尝试用OpenCV加载
        image = cv2.imread(input_path)
        if image is None:
            # 如果OpenCV失败，尝试用PIL加载
            try:
                from PIL import Image as PILImage
                pil_image = PILImage.open(input_path)
                # 转换为RGB并转换为numpy数组
                if pil_image.mode != 'RGB':
                    pil_image = pil_image.convert('RGB')
                image = np.array(pil_image)
                logger.info("使用PIL成功加载图像")
                return image
            except Exception as pil_error:
                raise ValueError(f"无法加载图像 {input_path}: OpenCV失败，PIL也失败 - {pil_error}")
        
        # OpenCV成功加载，转换颜色空间
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    def _simple_professional_pipeline(self, image: np.ndarray) -> np.ndarray:
        """极简专业处理流水线 - 完全模拟专业软件"""
        try:
            logger.info("开始极简专业处理...")
            
            # 第1步：轻度预处理（去噪但保持细节）
            denoised = cv2.bilateralFilter(image, 5, 50, 50)
            logger.info("  ✓ 轻度去噪完成")
            
            # 第2步：颜色量化（专业软件的核心功能）
            quantized = self._color_quantization(denoised)
            logger.info("  ✓ 颜色量化完成")
            
            # 第3步：边缘保持平滑（保持主体轮廓清晰）
            smoothed = self._edge_preserving_smooth(quantized)
            logger.info("  ✓ 边缘保持平滑完成")
            
            # 第4步：色彩增强（增强饱和度和对比度）
            enhanced = self._color_enhancement(smoothed)
            logger.info("  ✓ 色彩增强完成")
            
            # 第5步：轻度锐化（增强边缘但不破坏结构）
            sharpened = self._gentle_sharpening(enhanced)
            logger.info("  ✓ 轻度锐化完成")
            
            return sharpened
            
        except Exception as e:
            raise Exception(f"极简专业处理失败: {str(e)}")
    
    def _color_quantization(self, image: np.ndarray) -> np.ndarray:
        """颜色量化 - 专业软件的核心功能"""
        try:
            # 将图像重塑为像素数组
            pixels = image.reshape(-1, 3)
            
            # K-means聚类到指定颜色数
            kmeans = KMeans(
                n_clusters=self.config["color_count"],
                init='k-means++',
                n_init=20,
                max_iter=300,
                random_state=42
            )
            
            kmeans.fit(pixels)
            
            # 获取聚类中心（主要颜色）
            colors = kmeans.cluster_centers_.astype(np.uint8)
            
            # 将每个像素替换为最近的聚类中心颜色
            labels = kmeans.labels_
            quantized_pixels = colors[labels]
            
            # 重塑回图像形状
            quantized_image = quantized_pixels.reshape(image.shape)
            
            return quantized_image
            
        except Exception as e:
            logger.warning(f"颜色量化失败: {str(e)}")
            return image
    
    def _edge_preserving_smooth(self, image: np.ndarray) -> np.ndarray:
        """边缘保持平滑 - 平滑区域但保持边缘"""
        try:
            # 使用边缘保持滤波器
            smoothed = cv2.edgePreservingFilter(
                image, 
                flags=cv2.RECURS_FILTER,
                sigma_s=50,
                sigma_r=0.4
            )
            
            return smoothed
            
        except Exception as e:
            logger.warning(f"边缘保持平滑失败: {str(e)}")
            return image
    
    def _color_enhancement(self, image: np.ndarray) -> np.ndarray:
        """色彩增强 - 增强饱和度和对比度"""
        try:
            # 转换为PIL图像进行色彩调整
            pil_image = Image.fromarray(image)
            
            # 增强饱和度
            enhancer = ImageEnhance.Color(pil_image)
            enhanced = enhancer.enhance(self.config["saturation_boost"])
            
            # 增强对比度
            enhancer = ImageEnhance.Contrast(enhanced)
            enhanced = enhancer.enhance(self.config["contrast_boost"])
            
            # 转换回numpy数组
            return np.array(enhanced)
            
        except Exception as e:
            logger.warning(f"色彩增强失败: {str(e)}")
            return image
    
    def _gentle_sharpening(self, image: np.ndarray) -> np.ndarray:
        """轻度锐化 - 增强边缘但不破坏结构"""
        try:
            # 使用轻度锐化核
            kernel = np.array([
                [0, -0.5, 0],
                [-0.5, 3, -0.5],
                [0, -0.5, 0]
            ], dtype=np.float32)
            
            # 应用锐化
            sharpened = cv2.filter2D(image, -1, kernel)
            
            # 混合原图和锐化图，保持自然效果
            result = cv2.addWeighted(
                image, 1.0 - self.config["edge_enhancement"] + 1.0,
                sharpened, self.config["edge_enhancement"] - 1.0,
                0
            )
            
            return np.clip(result, 0, 255).astype(np.uint8)
            
        except Exception as e:
            logger.warning(f"轻度锐化失败: {str(e)}")
            return image
    
    def _create_comparison(self, original: np.ndarray, processed: np.ndarray) -> np.ndarray:
        """创建对比图"""
        try:
            # 确保两个图像尺寸相同
            if original.shape != processed.shape:
                processed = cv2.resize(processed, (original.shape[1], original.shape[0]))
            
            # 水平拼接
            comparison = np.hstack([original, processed])
            
            return comparison
            
        except Exception as e:
            logger.warning(f"创建对比图失败: {str(e)}")
            return processed
    
    def _save_image(self, image: np.ndarray, output_path: str):
        """保存图像"""
        try:
            # 转换为PIL图像并保存
            pil_image = Image.fromarray(image)
            pil_image.save(output_path, 'PNG', optimize=True)
            
            # 获取文件大小
            file_size = os.path.getsize(output_path) / (1024 * 1024)
            logger.info(f"图像已保存: {output_path} ({file_size:.2f} MB)")
            
        except Exception as e:
            raise Exception(f"保存图像失败: {str(e)}")

def main():
    """测试函数"""
    logging.basicConfig(level=logging.INFO)
    
    generator = SimpleProfessionalGenerator()
    
    # 测试图像
    test_image = "uploads/250625_205648.jpg"
    if not os.path.exists(test_image):
        print(f"测试图像不存在: {test_image}")
        return
    
    # 生成专业图像
    job_id = f"simple_test_{int(time.time())}"
    
    try:
        professional_path, comparison_path, processing_time = generator.generate_professional_image(
            test_image, job_id
        )
        
        print(f"✅ 极简专业图像生成成功！")
        print(f"📁 专业图像: {professional_path}")
        print(f"📊 对比图像: {comparison_path}")
        print(f"⏱️  处理时间: {processing_time:.2f}秒")
        
    except Exception as e:
        print(f"❌ 生成失败: {str(e)}")

if __name__ == "__main__":
    main() 