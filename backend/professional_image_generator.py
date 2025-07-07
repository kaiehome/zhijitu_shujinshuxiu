"""
专业织机识别图像生成器
基于对专业织机软件的分析，优化图像处理算法以达到专业级别的效果
"""

import cv2
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance, ImageOps, ImageDraw
from sklearn.cluster import KMeans
import os
import logging
import time
from pathlib import Path
from typing import Tuple, List, Optional, Dict, Any
from scipy import ndimage
from scipy.spatial.distance import cdist
import warnings

# 抑制不必要的警告
warnings.filterwarnings('ignore', category=UserWarning)

logger = logging.getLogger(__name__)


class ProfessionalWeavingImageGenerator:
    """
    专业织机识别图像生成器
    
    基于专业织机软件的效果分析，实现：
    1. 高质量颜色区域连贯性
    2. 专业级边缘锐化
    3. 艺术化背景装饰
    4. 织机识别优化
    """
    
    def __init__(self, outputs_dir: str = "outputs"):
        """初始化专业图像生成器"""
        self.outputs_dir = Path(outputs_dir)
        self.outputs_dir.mkdir(exist_ok=True, parents=True)
        
        # 专业织机配置
        self.weaving_config = {
            # 颜色处理
            "color_connectivity_strength": 15,  # 颜色连通性强度
            "color_smoothing_radius": 8,        # 颜色平滑半径
            "dominant_color_threshold": 0.02,   # 主导色阈值
            
            # 边缘处理
            "edge_sharpening_intensity": 2.5,   # 边缘锐化强度
            "contour_enhancement": True,        # 轮廓增强
            "edge_smoothing": True,            # 边缘平滑
            
            # 区域处理
            "region_consolidation": True,       # 区域合并
            "small_region_threshold": 100,      # 小区域阈值
            "region_filling": True,            # 区域填充
            
            # 艺术化处理
            "decorative_border": True,         # 装饰性边框
            "background_enhancement": True,     # 背景增强
            "artistic_pattern": True,          # 艺术图案
            
            # 质量优化
            "anti_aliasing": True,            # 抗锯齿
            "texture_preservation": True,      # 纹理保持
            "color_saturation_boost": 1.3,    # 色彩饱和度提升
        }
        
        logger.info("专业织机识别图像生成器已初始化")
    
    def generate_professional_weaving_image(self, 
                                          input_path: str, 
                                          job_id: str,
                                          color_count: int = 16,
                                          style: str = "professional") -> Tuple[str, str, float]:
        """
        生成专业织机识别图像
        
        Args:
            input_path: 输入图像路径
            job_id: 任务ID
            color_count: 颜色数量
            style: 处理风格 ("professional", "artistic", "technical")
            
        Returns:
            Tuple[professional_png_path, comparison_png_path, processing_time]
        """
        start_time = time.time()
        
        try:
            logger.info(f"🎨 开始生成专业织机识别图像: {job_id}")
            
            # 创建输出目录
            job_dir = self.outputs_dir / job_id
            job_dir.mkdir(exist_ok=True, parents=True)
            
            # 加载和预处理原始图像
            original_image = self._load_image_professionally(input_path)
            logger.info("✓ 原始图像加载完成")
            
            # 专业织机处理流水线
            processed_image = self._professional_weaving_pipeline(
                original_image, color_count, style
            )
            logger.info("✓ 专业织机处理完成")
            
            # 创建对比图像
            comparison_image = self._create_comparison_visualization(
                original_image, processed_image
            )
            logger.info("✓ 对比可视化创建完成")
            
            # 添加专业装饰和边框
            final_professional_image = self._add_professional_decorations(
                processed_image, style
            )
            logger.info("✓ 专业装饰添加完成")
            
            # 保存结果
            professional_path = job_dir / f"{job_id}_professional_weaving.png"
            comparison_path = job_dir / f"{job_id}_comparison.png"
            
            self._save_professional_image(final_professional_image, str(professional_path))
            self._save_professional_image(comparison_image, str(comparison_path))
            
            processing_time = time.time() - start_time
            logger.info(f"🎯 专业织机图像生成完成，耗时: {processing_time:.2f}秒")
            
            return str(professional_path), str(comparison_path), processing_time
            
        except Exception as e:
            error_msg = f"专业织机图像生成失败: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise Exception(error_msg)
    
    def _load_image_professionally(self, input_path: str) -> np.ndarray:
        """专业级图像加载"""
        try:
            with Image.open(input_path) as pil_image:
                if pil_image.mode != 'RGB':
                    pil_image = pil_image.convert('RGB')
                
                # 高质量重采样
                image = np.array(pil_image.resize(
                    pil_image.size, Image.Resampling.LANCZOS
                ))
                
                return image
                
        except Exception as e:
            raise Exception(f"图像加载失败: {str(e)}")
    
    def _professional_weaving_pipeline(self, 
                                     image: np.ndarray, 
                                     color_count: int, 
                                     style: str) -> np.ndarray:
        """专业织机处理流水线"""
        try:
            # 1. 智能颜色降色（基于织机需求）
            color_reduced = self._intelligent_color_reduction(image, color_count)
            logger.info("  ✓ 智能颜色降色完成")
            
            # 2. 颜色区域连贯性增强
            coherent_regions = self._enhance_color_coherence(color_reduced)
            logger.info("  ✓ 颜色区域连贯性增强完成")
            
            # 3. 专业级边缘处理
            sharp_edges = self._professional_edge_enhancement(coherent_regions)
            logger.info("  ✓ 专业级边缘处理完成")
            
            # 4. 织机识别优化
            weaving_optimized = self._weaving_machine_optimization(sharp_edges)
            logger.info("  ✓ 织机识别优化完成")
            
            # 5. 区域合并和清理
            clean_regions = self._region_consolidation(weaving_optimized)
            logger.info("  ✓ 区域合并和清理完成")
            
            # 6. 质量增强
            enhanced_quality = self._quality_enhancement(clean_regions, style)
            logger.info("  ✓ 质量增强完成")
            
            return enhanced_quality
            
        except Exception as e:
            raise Exception(f"专业织机处理流水线失败: {str(e)}")
    
    def _intelligent_color_reduction(self, image: np.ndarray, n_colors: int) -> np.ndarray:
        """智能颜色降色 - 基于织机需求优化"""
        try:
            # 预处理：轻微高斯模糊以减少噪声
            blurred = cv2.GaussianBlur(image, (3, 3), 0.8)
            
            # 重塑为像素向量
            pixels = blurred.reshape(-1, 3)
            
            # 使用改进的K-means聚类
            kmeans = KMeans(
                n_clusters=n_colors,
                init='k-means++',
                n_init=30,  # 增加初始化次数
                max_iter=500,  # 增加迭代次数
                random_state=42,
                algorithm='lloyd'
            )
            
            # 执行聚类
            kmeans.fit(pixels)
            
            # 获取聚类中心并优化颜色
            colors = kmeans.cluster_centers_.astype(np.uint8)
            
            # 颜色优化：增强对比度和饱和度
            colors = self._optimize_weaving_colors(colors)
            
            # 替换像素颜色
            labels = kmeans.labels_
            reduced_pixels = colors[labels]
            
            # 重塑回原始图像形状
            reduced_image = reduced_pixels.reshape(image.shape)
            
            return reduced_image
            
        except Exception as e:
            raise Exception(f"智能颜色降色失败: {str(e)}")
    
    def _optimize_weaving_colors(self, colors: np.ndarray) -> np.ndarray:
        """优化织机颜色 - 增强对比度和饱和度"""
        try:
            optimized_colors = []
            
            for color in colors:
                # 转换为HSV以便于调整饱和度
                hsv = cv2.cvtColor(color.reshape(1, 1, 3), cv2.COLOR_RGB2HSV)[0, 0]
                
                # 增强饱和度（织机需要高饱和度）
                hsv[1] = min(255, int(hsv[1] * self.weaving_config["color_saturation_boost"]))
                
                # 轻微调整亮度以增强对比度
                if hsv[2] > 128:
                    hsv[2] = min(255, int(hsv[2] * 1.1))
                else:
                    hsv[2] = max(0, int(hsv[2] * 0.9))
                
                # 转换回RGB
                rgb = cv2.cvtColor(hsv.reshape(1, 1, 3), cv2.COLOR_HSV2RGB)[0, 0]
                optimized_colors.append(rgb)
            
            return np.array(optimized_colors, dtype=np.uint8)
            
        except Exception as e:
            logger.warning(f"颜色优化失败，使用原始颜色: {str(e)}")
            return colors
    
    def _enhance_color_coherence(self, image: np.ndarray) -> np.ndarray:
        """增强颜色区域连贯性 - 关键改进"""
        try:
            # 1. 获取所有唯一颜色
            unique_colors = self._get_unique_colors(image)
            
            # 2. 为每个颜色创建掩码并增强连贯性
            result = image.copy()
            
            for color in unique_colors:
                # 创建当前颜色的掩码
                mask = np.all(image == color, axis=2)
                
                if np.sum(mask) == 0:
                    continue
                
                # 形态学操作增强连贯性
                kernel_size = self.weaving_config["color_connectivity_strength"]
                kernel = np.ones((kernel_size, kernel_size), np.uint8)
                
                # 闭运算：连接相近区域
                mask_closed = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
                
                # 开运算：去除小噪点
                mask_cleaned = cv2.morphologyEx(mask_closed, cv2.MORPH_OPEN, kernel//2)
                
                # 应用清理后的掩码
                result[mask_cleaned > 0] = color
            
            # 3. 整体平滑处理
            smoothed = cv2.bilateralFilter(
                result, 
                self.weaving_config["color_smoothing_radius"], 
                80, 80
            )
            
            return smoothed
            
        except Exception as e:
            raise Exception(f"颜色连贯性增强失败: {str(e)}")
    
    def _professional_edge_enhancement(self, image: np.ndarray) -> np.ndarray:
        """专业级边缘增强 - 对标专业软件"""
        try:
            # 1. 多尺度边缘检测
            edges = self._multi_scale_edge_detection(image)
            
            # 2. 边缘锐化
            sharpened = self._selective_sharpening(image, edges)
            
            # 3. 轮廓增强
            if self.weaving_config["contour_enhancement"]:
                contour_enhanced = self._enhance_contours(sharpened)
            else:
                contour_enhanced = sharpened
            
            # 4. 边缘平滑
            if self.weaving_config["edge_smoothing"]:
                final_edges = self._smooth_edges(contour_enhanced)
            else:
                final_edges = contour_enhanced
            
            return final_edges
            
        except Exception as e:
            raise Exception(f"专业级边缘增强失败: {str(e)}")
    
    def _multi_scale_edge_detection(self, image: np.ndarray) -> np.ndarray:
        """多尺度边缘检测"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            # 多个尺度的边缘检测
            edges_combined = np.zeros_like(gray)
            
            scales = [1, 2, 3]
            for scale in scales:
                # 高斯模糊
                blurred = cv2.GaussianBlur(gray, (scale*2+1, scale*2+1), scale)
                
                # Canny边缘检测
                edges = cv2.Canny(blurred, 50, 150)
                
                # 合并边缘
                edges_combined = cv2.bitwise_or(edges_combined, edges)
            
            return edges_combined
            
        except Exception as e:
            logger.warning(f"多尺度边缘检测失败: {str(e)}")
            return cv2.Canny(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY), 50, 150)
    
    def _selective_sharpening(self, image: np.ndarray, edges: np.ndarray) -> np.ndarray:
        """选择性锐化 - 只在边缘区域锐化"""
        try:
            # 创建锐化核
            intensity = self.weaving_config["edge_sharpening_intensity"]
            kernel = np.array([
                [-1, -1, -1],
                [-1, 8+intensity, -1],
                [-1, -1, -1]
            ])
            
            # 应用锐化
            sharpened = cv2.filter2D(image, -1, kernel)
            
            # 只在边缘区域应用锐化
            edges_3d = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB) / 255.0
            
            # 边缘区域使用锐化版本，其他区域保持原样
            result = (image * (1 - edges_3d) + sharpened * edges_3d).astype(np.uint8)
            
            return result
            
        except Exception as e:
            logger.warning(f"选择性锐化失败: {str(e)}")
            return image
    
    def _enhance_contours(self, image: np.ndarray) -> np.ndarray:
        """增强轮廓清晰度"""
        try:
            # 找到轮廓
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # 在原图上绘制增强的轮廓
            result = image.copy()
            
            # 绘制轮廓（稍微加粗）
            cv2.drawContours(result, contours, -1, (0, 0, 0), 1)
            
            return result
            
        except Exception as e:
            logger.warning(f"轮廓增强失败: {str(e)}")
            return image
    
    def _smooth_edges(self, image: np.ndarray) -> np.ndarray:
        """边缘平滑处理"""
        try:
            # 使用双边滤波保持边缘的同时平滑
            smoothed = cv2.bilateralFilter(image, 5, 50, 50)
            return smoothed
            
        except Exception as e:
            logger.warning(f"边缘平滑失败: {str(e)}")
            return image
    
    def _weaving_machine_optimization(self, image: np.ndarray) -> np.ndarray:
        """织机识别专项优化"""
        try:
            # 1. 颜色区域合并
            merged_regions = self._merge_similar_regions(image)
            
            # 2. 小区域清理
            cleaned = self._clean_small_regions(merged_regions)
            
            # 3. 边界清理
            boundary_cleaned = self._clean_boundaries(cleaned)
            
            return boundary_cleaned
            
        except Exception as e:
            raise Exception(f"织机识别优化失败: {str(e)}")
    
    def _merge_similar_regions(self, image: np.ndarray) -> np.ndarray:
        """合并相似颜色区域"""
        try:
            # 使用形态学操作合并相似区域
            kernel = np.ones((5, 5), np.uint8)
            
            # 闭运算
            closed = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
            
            # 开运算
            opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)
            
            return opened
            
        except Exception as e:
            logger.warning(f"区域合并失败: {str(e)}")
            return image
    
    def _clean_small_regions(self, image: np.ndarray) -> np.ndarray:
        """清理小区域"""
        try:
            # 获取所有唯一颜色
            unique_colors = self._get_unique_colors(image)
            result = image.copy()
            
            for color in unique_colors:
                # 为每个颜色创建掩码
                mask = np.all(image == color, axis=2).astype(np.uint8)
                
                # 找到连通组件
                num_labels, labels = cv2.connectedComponents(mask)
                
                # 计算每个组件的大小
                for label in range(1, num_labels):
                    component_mask = (labels == label)
                    component_size = np.sum(component_mask)
                    
                    # 如果组件太小，用周围的主要颜色替换
                    if component_size < self.weaving_config["small_region_threshold"]:
                        # 简单处理：用最近邻颜色替换
                        result[component_mask] = self._get_dominant_neighbor_color(
                            image, component_mask
                        )
            
            return result
            
        except Exception as e:
            logger.warning(f"小区域清理失败: {str(e)}")
            return image
    
    def _clean_boundaries(self, image: np.ndarray) -> np.ndarray:
        """清理边界"""
        try:
            # 使用中值滤波清理边界噪声
            cleaned = cv2.medianBlur(image, 3)
            
            return cleaned
            
        except Exception as e:
            logger.warning(f"边界清理失败: {str(e)}")
            return image
    
    def _region_consolidation(self, image: np.ndarray) -> np.ndarray:
        """区域合并"""
        try:
            if not self.weaving_config["region_consolidation"]:
                return image
            
            # 1. 形态学操作
            kernel = np.ones((7, 7), np.uint8)
            
            # 闭运算 - 连接相近区域
            closed = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
            
            # 2. 区域填充
            if self.weaving_config["region_filling"]:
                filled = self._fill_regions(closed)
            else:
                filled = closed
            
            return filled
            
        except Exception as e:
            raise Exception(f"区域合并失败: {str(e)}")
    
    def _fill_regions(self, image: np.ndarray) -> np.ndarray:
        """填充区域内的空洞"""
        try:
            # 对每个颜色通道分别处理
            result = image.copy()
            
            for channel in range(3):
                # 使用形态学闭运算填充空洞
                kernel = np.ones((5, 5), np.uint8)
                filled = cv2.morphologyEx(result[:, :, channel], cv2.MORPH_CLOSE, kernel)
                result[:, :, channel] = filled
            
            return result
            
        except Exception as e:
            logger.warning(f"区域填充失败: {str(e)}")
            return image
    
    def _quality_enhancement(self, image: np.ndarray, style: str) -> np.ndarray:
        """质量增强"""
        try:
            # 1. 抗锯齿处理
            if self.weaving_config["anti_aliasing"]:
                anti_aliased = self._apply_anti_aliasing(image)
            else:
                anti_aliased = image
            
            # 2. 纹理保持
            if self.weaving_config["texture_preservation"]:
                texture_preserved = self._preserve_texture(anti_aliased)
            else:
                texture_preserved = anti_aliased
            
            # 3. 最终色彩调整
            final_enhanced = self._final_color_adjustment(texture_preserved, style)
            
            return final_enhanced
            
        except Exception as e:
            raise Exception(f"质量增强失败: {str(e)}")
    
    def _apply_anti_aliasing(self, image: np.ndarray) -> np.ndarray:
        """应用抗锯齿"""
        try:
            # 使用高质量重采样实现抗锯齿
            pil_image = Image.fromarray(image)
            
            # 先放大再缩小，实现抗锯齿效果
            width, height = pil_image.size
            enlarged = pil_image.resize((width*2, height*2), Image.Resampling.LANCZOS)
            anti_aliased = enlarged.resize((width, height), Image.Resampling.LANCZOS)
            
            return np.array(anti_aliased)
            
        except Exception as e:
            logger.warning(f"抗锯齿处理失败: {str(e)}")
            return image
    
    def _preserve_texture(self, image: np.ndarray) -> np.ndarray:
        """保持纹理"""
        try:
            # 使用保边滤波器保持纹理
            preserved = cv2.edgePreservingFilter(image, flags=1, sigma_s=50, sigma_r=0.4)
            
            return preserved
            
        except Exception as e:
            logger.warning(f"纹理保持失败: {str(e)}")
            return image
    
    def _final_color_adjustment(self, image: np.ndarray, style: str) -> np.ndarray:
        """最终色彩调整"""
        try:
            pil_image = Image.fromarray(image)
            
            # 根据风格调整
            if style == "professional":
                # 专业风格：增强对比度和饱和度
                enhancer = ImageEnhance.Contrast(pil_image)
                enhanced = enhancer.enhance(1.2)
                
                enhancer = ImageEnhance.Color(enhanced)
                enhanced = enhancer.enhance(1.3)
                
            elif style == "artistic":
                # 艺术风格：增强色彩鲜艳度
                enhancer = ImageEnhance.Color(pil_image)
                enhanced = enhancer.enhance(1.4)
                
                enhancer = ImageEnhance.Brightness(enhanced)
                enhanced = enhancer.enhance(1.1)
                
            else:  # technical
                # 技术风格：保持准确性
                enhancer = ImageEnhance.Sharpness(pil_image)
                enhanced = enhancer.enhance(1.1)
            
            return np.array(enhanced)
            
        except Exception as e:
            logger.warning(f"最终色彩调整失败: {str(e)}")
            return image
    
    def _add_professional_decorations(self, image: np.ndarray, style: str) -> np.ndarray:
        """添加专业装饰（模仿专业织机软件）"""
        try:
            if not self.weaving_config["decorative_border"]:
                return image
            
            # 转换为PIL图像以便于绘制
            pil_image = Image.fromarray(image)
            width, height = pil_image.size
            
            # 创建带边框的新图像
            border_width = 20
            new_width = width + 2 * border_width
            new_height = height + 2 * border_width
            
            # 创建背景
            if style == "professional":
                border_color = (255, 192, 203)  # 粉色边框（模仿专业软件）
            elif style == "artistic":
                border_color = (255, 215, 0)    # 金色边框
            else:
                border_color = (128, 128, 128)  # 灰色边框
            
            bordered_image = Image.new('RGB', (new_width, new_height), border_color)
            
            # 粘贴原图像到中心
            bordered_image.paste(pil_image, (border_width, border_width))
            
            # 添加装饰图案（如果启用）
            if self.weaving_config["artistic_pattern"]:
                bordered_image = self._add_decorative_pattern(bordered_image, style)
            
            return np.array(bordered_image)
            
        except Exception as e:
            logger.warning(f"专业装饰添加失败: {str(e)}")
            return image
    
    def _add_decorative_pattern(self, pil_image: Image.Image, style: str) -> Image.Image:
        """添加装饰图案"""
        try:
            draw = ImageDraw.Draw(pil_image)
            width, height = pil_image.size
            
            if style == "professional":
                # 专业风格：简单几何图案
                pattern_color = (255, 255, 255)  # 白色图案
                
                # 在边框上绘制小圆点
                for i in range(10, width-10, 30):
                    draw.ellipse([i-3, 7, i+3, 13], fill=pattern_color)
                    draw.ellipse([i-3, height-13, i+3, height-7], fill=pattern_color)
                
                for i in range(10, height-10, 30):
                    draw.ellipse([7, i-3, 13, i+3], fill=pattern_color)
                    draw.ellipse([width-13, i-3, width-7, i+3], fill=pattern_color)
            
            elif style == "artistic":
                # 艺术风格：更复杂的图案
                pattern_color = (255, 255, 255)
                
                # 绘制角落装饰
                for corner in [(10, 10), (width-30, 10), (10, height-30), (width-30, height-30)]:
                    x, y = corner
                    draw.rectangle([x, y, x+20, y+20], outline=pattern_color, width=2)
                    draw.line([x, y, x+20, y+20], fill=pattern_color, width=1)
                    draw.line([x+20, y, x, y+20], fill=pattern_color, width=1)
            
            return pil_image
            
        except Exception as e:
            logger.warning(f"装饰图案添加失败: {str(e)}")
            return pil_image
    
    def _create_comparison_visualization(self, 
                                       original: np.ndarray, 
                                       processed: np.ndarray) -> np.ndarray:
        """创建对比可视化"""
        try:
            # 确保两个图像尺寸一致
            if original.shape != processed.shape:
                original = cv2.resize(original, (processed.shape[1], processed.shape[0]))
            
            # 创建并排对比
            comparison = np.hstack([original, processed])
            
            # 转换为PIL进行文字标注
            pil_comparison = Image.fromarray(comparison)
            draw = ImageDraw.Draw(pil_comparison)
            
            # 添加标题
            width, height = pil_comparison.size
            
            # 标注原图
            draw.text((width//4, 10), "原始图像", fill=(255, 255, 255), anchor="mm")
            
            # 标注处理后图像
            draw.text((3*width//4, 10), "专业织机处理", fill=(255, 255, 255), anchor="mm")
            
            # 添加分隔线
            draw.line([(width//2, 0), (width//2, height)], fill=(255, 255, 255), width=2)
            
            return np.array(pil_comparison)
            
        except Exception as e:
            logger.warning(f"对比可视化创建失败: {str(e)}")
            # 返回处理后的图像
            return processed
    
    def _save_professional_image(self, image: np.ndarray, output_path: str):
        """保存专业级图像"""
        try:
            pil_image = Image.fromarray(image)
            
            # 超高质量保存设置
            save_kwargs = {
                'format': 'PNG',
                'optimize': False,
                'compress_level': 0,
                'pnginfo': self._create_professional_metadata()
            }
            
            pil_image.save(output_path, **save_kwargs)
            
            file_size = Path(output_path).stat().st_size
            logger.info(f"专业图像已保存: {Path(output_path).name} ({file_size/1024/1024:.2f} MB)")
            
        except Exception as e:
            raise Exception(f"专业图像保存失败: {str(e)}")
    
    def _create_professional_metadata(self) -> Optional[object]:
        """创建专业元数据"""
        try:
            from PIL.PngImagePlugin import PngInfo
            metadata = PngInfo()
            metadata.add_text("Software", "专业织机识别图像生成器")
            metadata.add_text("Description", "Professional Weaving Machine Recognition Image")
            metadata.add_text("Generator", "ProfessionalWeavingImageGenerator v1.0")
            metadata.add_text("Creation Time", time.strftime("%Y-%m-%d %H:%M:%S"))
            return metadata
        except ImportError:
            return None
    
    def _get_unique_colors(self, image: np.ndarray) -> List[Tuple[int, int, int]]:
        """获取图像中的唯一颜色"""
        try:
            pixels = image.reshape(-1, 3)
            unique_colors = np.unique(pixels, axis=0)
            return [tuple(color) for color in unique_colors]
        except Exception as e:
            logger.warning(f"获取唯一颜色失败: {str(e)}")
            return []
    
    def _get_dominant_neighbor_color(self, 
                                   image: np.ndarray, 
                                   mask: np.ndarray) -> Tuple[int, int, int]:
        """获取周围区域的主导颜色"""
        try:
            # 膨胀掩码以获取周围区域
            kernel = np.ones((5, 5), np.uint8)
            dilated_mask = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1)
            
            # 获取边界区域
            boundary = dilated_mask - mask.astype(np.uint8)
            
            if np.sum(boundary) == 0:
                return (128, 128, 128)  # 默认灰色
            
            # 获取边界区域的颜色
            boundary_colors = image[boundary > 0]
            
            if len(boundary_colors) == 0:
                return (128, 128, 128)
            
            # 返回最常见的颜色
            unique_colors, counts = np.unique(boundary_colors, axis=0, return_counts=True)
            dominant_color = unique_colors[np.argmax(counts)]
            
            return tuple(dominant_color)
            
        except Exception as e:
            logger.warning(f"获取主导邻居颜色失败: {str(e)}")
            return (128, 128, 128)
    
    def get_generator_info(self) -> Dict[str, Any]:
        """获取生成器信息"""
        return {
            "generator_version": "1.0.0",
            "output_directory": str(self.outputs_dir),
            "weaving_config": self.weaving_config.copy(),
            "supported_styles": ["professional", "artistic", "technical"],
            "features": [
                "专业织机识别优化",
                "智能颜色降色",
                "颜色区域连贯性增强",
                "专业级边缘处理",
                "多尺度边缘检测",
                "选择性锐化",
                "轮廓增强",
                "区域合并清理",
                "装饰性边框",
                "对比可视化",
                "超高质量输出"
            ]
        } 