#!/usr/bin/env python3
"""
优化专业织机识别图像生成器
基于极简版本，进一步提升结构相似性和专业效果
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

class OptimizedProfessionalGenerator:
    """优化专业织机识别图像生成器"""
    
    def __init__(self):
        """初始化"""
        self.config = {
            # 优化配置，确保结构保持的同时达到专业效果
            "color_count": 12,          # 增加到12种颜色，保持更多细节
            "saturation_boost": 2.0,    # 2.0倍饱和度
            "contrast_boost": 1.6,      # 1.6倍对比度
            "edge_enhancement": 1.5,    # 适度边缘增强
            "structure_preservation": 0.7,  # 结构保持权重
        }
        logger.info("优化专业织机生成器已初始化")
    
    def generate_professional_image(self, input_path: str, job_id: str) -> Tuple[str, str, float]:
        """生成专业织机识别图像 - 优化版本"""
        start_time = time.time()
        
        try:
            # 1. 加载原图
            image = self._load_image(input_path)
            logger.info(f"原图尺寸: {image.shape}")
            
            # 2. 优化专业处理流水线
            professional_image = self._optimized_professional_pipeline(image)
            
            # 3. 保存结果
            output_dir = f"outputs/{job_id}"
            os.makedirs(output_dir, exist_ok=True)
            
            professional_path = os.path.join(output_dir, f"{job_id}_optimized_professional.png")
            comparison_path = os.path.join(output_dir, f"{job_id}_optimized_comparison.png")
            
            # 保存专业图像
            self._save_image(professional_image, professional_path)
            
            # 创建对比图
            comparison = self._create_comparison(image, professional_image)
            self._save_image(comparison, comparison_path)
            
            processing_time = time.time() - start_time
            logger.info(f"优化专业图像生成完成，耗时: {processing_time:.2f}秒")
            
            return professional_path, comparison_path, processing_time
            
        except Exception as e:
            raise Exception(f"优化专业图像生成失败: {str(e)}")
    
    def _load_image(self, input_path: str) -> np.ndarray:
        """加载图像"""
        image = cv2.imread(input_path)
        if image is None:
            raise ValueError(f"无法加载图像: {input_path}")
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    def _optimized_professional_pipeline(self, image: np.ndarray) -> np.ndarray:
        """优化专业处理流水线"""
        try:
            logger.info("开始优化专业处理...")
            
            # 保存原图用于结构保持
            original = image.copy()
            
            # 第1步：智能颜色量化（保持结构）
            quantized = self._structure_preserving_quantization(image)
            logger.info("  ✓ 结构保持颜色量化完成")
            
            # 第2步：专业色彩增强
            enhanced = self._professional_color_enhancement(quantized)
            logger.info("  ✓ 专业色彩增强完成")
            
            # 第3步：智能边缘增强（不破坏结构）
            sharpened = self._intelligent_edge_enhancement(enhanced, original)
            logger.info("  ✓ 智能边缘增强完成")
            
            # 第4步：结构保持融合
            final_result = self._structure_preserving_blend(original, sharpened)
            logger.info("  ✓ 结构保持融合完成")
            
            return final_result
            
        except Exception as e:
            raise Exception(f"优化专业处理失败: {str(e)}")
    
    def _structure_preserving_quantization(self, image: np.ndarray) -> np.ndarray:
        """结构保持颜色量化"""
        try:
            # 检测边缘区域
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 30, 100)
            
            # 创建边缘掩码
            edge_mask = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
            
            # 分别处理边缘区域和非边缘区域
            edge_pixels = image[edge_mask > 0]
            non_edge_pixels = image[edge_mask == 0]
            
            # 对非边缘区域进行颜色量化
            if len(non_edge_pixels) > 0:
                pixels = non_edge_pixels.reshape(-1, 3)
                
                kmeans = KMeans(
                    n_clusters=min(self.config["color_count"], len(pixels)),
                    init='k-means++',
                    n_init=20,
                    max_iter=300,
                    random_state=42
                )
                
                kmeans.fit(pixels)
                colors = kmeans.cluster_centers_.astype(np.uint8)
                labels = kmeans.labels_
                quantized_pixels = colors[labels]
                
                # 重建图像
                result = image.copy()
                result[edge_mask == 0] = quantized_pixels.reshape(-1, 3)
                
                # 边缘区域保持原始颜色但做轻微调整
                if len(edge_pixels) > 0:
                    # 对边缘区域做轻微的颜色调整，使其与量化后的颜色协调
                    edge_enhanced = self._adjust_edge_colors(edge_pixels, colors)
                    result[edge_mask > 0] = edge_enhanced
                
                return result
            else:
                return image
            
        except Exception as e:
            logger.warning(f"结构保持颜色量化失败: {str(e)}")
            return image
    
    def _adjust_edge_colors(self, edge_pixels: np.ndarray, palette_colors: np.ndarray) -> np.ndarray:
        """调整边缘颜色使其与调色板协调"""
        try:
            adjusted_pixels = []
            
            for pixel in edge_pixels:
                # 找到最接近的调色板颜色
                distances = np.sum((palette_colors - pixel) ** 2, axis=1)
                closest_color = palette_colors[np.argmin(distances)]
                
                # 混合原始颜色和最接近的调色板颜色
                adjusted_pixel = (pixel * 0.7 + closest_color * 0.3).astype(np.uint8)
                adjusted_pixels.append(adjusted_pixel)
            
            return np.array(adjusted_pixels)
            
        except Exception as e:
            logger.warning(f"边缘颜色调整失败: {str(e)}")
            return edge_pixels
    
    def _professional_color_enhancement(self, image: np.ndarray) -> np.ndarray:
        """专业色彩增强"""
        try:
            # 转换为PIL图像
            pil_image = Image.fromarray(image)
            
            # 增强饱和度
            enhancer = ImageEnhance.Color(pil_image)
            enhanced = enhancer.enhance(self.config["saturation_boost"])
            
            # 增强对比度
            enhancer = ImageEnhance.Contrast(enhanced)
            enhanced = enhancer.enhance(self.config["contrast_boost"])
            
            # 轻微增强锐度
            enhancer = ImageEnhance.Sharpness(enhanced)
            enhanced = enhancer.enhance(1.2)
            
            return np.array(enhanced)
            
        except Exception as e:
            logger.warning(f"专业色彩增强失败: {str(e)}")
            return image
    
    def _intelligent_edge_enhancement(self, image: np.ndarray, original: np.ndarray) -> np.ndarray:
        """智能边缘增强"""
        try:
            # 检测原图的边缘
            gray_orig = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)
            edges_orig = cv2.Canny(gray_orig, 50, 150)
            
            # 创建锐化核
            kernel = np.array([
                [-0.3, -0.7, -0.3],
                [-0.7, 4.0, -0.7],
                [-0.3, -0.7, -0.3]
            ], dtype=np.float32)
            
            # 应用锐化
            sharpened = cv2.filter2D(image, -1, kernel)
            
            # 只在原图边缘区域应用锐化
            edges_3d = cv2.cvtColor(edges_orig, cv2.COLOR_GRAY2RGB) / 255.0
            edges_3d = cv2.dilate(edges_3d, np.ones((3, 3), np.uint8), iterations=1)
            
            # 混合锐化和原图
            enhancement_strength = self.config["edge_enhancement"] - 1.0
            result = image * (1 - edges_3d * enhancement_strength) + sharpened * (edges_3d * enhancement_strength)
            
            return np.clip(result, 0, 255).astype(np.uint8)
            
        except Exception as e:
            logger.warning(f"智能边缘增强失败: {str(e)}")
            return image
    
    def _structure_preserving_blend(self, original: np.ndarray, processed: np.ndarray) -> np.ndarray:
        """结构保持融合"""
        try:
            # 计算结构保持权重
            preservation_weight = self.config["structure_preservation"]
            
            # 检测重要结构区域（高梯度区域）
            gray_orig = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)
            
            # 计算梯度
            grad_x = cv2.Sobel(gray_orig, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray_orig, cv2.CV_64F, 0, 1, ksize=3)
            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            
            # 归一化梯度
            gradient_normalized = gradient_magnitude / np.max(gradient_magnitude)
            gradient_3d = np.stack([gradient_normalized] * 3, axis=2)
            
            # 在高梯度区域更多地保持原图结构
            structure_mask = gradient_3d * preservation_weight
            
            # 融合
            result = processed * (1 - structure_mask) + original * structure_mask
            
            return np.clip(result, 0, 255).astype(np.uint8)
            
        except Exception as e:
            logger.warning(f"结构保持融合失败: {str(e)}")
            return processed
    
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
    
    generator = OptimizedProfessionalGenerator()
    
    # 测试图像
    test_image = "uploads/250625_205648.jpg"
    if not os.path.exists(test_image):
        print(f"测试图像不存在: {test_image}")
        return
    
    # 生成专业图像
    job_id = f"optimized_test_{int(time.time())}"
    
    try:
        professional_path, comparison_path, processing_time = generator.generate_professional_image(
            test_image, job_id
        )
        
        print(f"✅ 优化专业图像生成成功！")
        print(f"📁 专业图像: {professional_path}")
        print(f"📊 对比图像: {comparison_path}")
        print(f"⏱️  处理时间: {processing_time:.2f}秒")
        
    except Exception as e:
        print(f"❌ 生成失败: {str(e)}")

if __name__ == "__main__":
    main() 