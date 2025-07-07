"""
专业织机识别图像生成器
基于专业织机软件效果分析，提供高质量的图像处理功能
"""

import cv2
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance, ImageOps, ImageDraw, ImageFont
from sklearn.cluster import KMeans
import logging
import time
from pathlib import Path
from ai_enhanced_processor import AIEnhancedProcessor
from ai_image_generator import AIImageGenerator
from typing import Tuple, List, Optional, Dict, Any
import warnings

warnings.filterwarnings('ignore', category=UserWarning)
logger = logging.getLogger(__name__)


class ProfessionalWeavingGenerator:
    """专业织机识别图像生成器"""
    
    def __init__(self, outputs_dir: str = "outputs"):
        """初始化生成器"""
        self.outputs_dir = Path(outputs_dir)
        self.outputs_dir.mkdir(exist_ok=True, parents=True)
        
        # 初始化AI增强处理器
        try:
            self.ai_processor = AIEnhancedProcessor()
            logger.info("AI增强处理器初始化成功")
        except Exception as e:
            logger.warning(f"AI增强处理器初始化失败: {str(e)}")
            self.ai_processor = None
        
        # 初始化AI图像生成器
        try:
            self.ai_generator = AIImageGenerator()
            logger.info("AI图像生成器初始化成功")
        except Exception as e:
            logger.warning(f"AI图像生成器初始化失败: {str(e)}")
            self.ai_generator = None
        
        # AI增强织机识别配置 - 向专业软件靠近
        self.config = {
            "color_connectivity_strength": 18,      # 增强连贯性
            "edge_sharpening_intensity": 2.5,       # 增强边缘锐化
            "region_consolidation": True,
            "decorative_border": True,              # 启用专业边框
            "color_saturation_boost": 1.4,          # 增强饱和度
            "contrast_boost": 1.3,                  # 增强对比度
            "anti_aliasing": True,
            "artistic_background": True,             # 启用AI艺术化背景
            "complex_decoration": True,              # 启用专业装饰
            "professional_enhancement": True,        # 启用专业级AI增强
            "pure_weaving_mode": False,             # 专业模式
        }
        
        logger.info("专业织机识别图像生成器已初始化")
    
    def generate_professional_image(self, 
                                  input_path: str, 
                                  job_id: str,
                                  color_count: int = 16) -> Tuple[str, str, float]:
        """生成专业织机识别图像"""
        start_time = time.time()
        
        try:
            logger.info(f"🎨 开始生成专业织机识别图像: {job_id}")
            
            # 创建输出目录
            job_dir = self.outputs_dir / job_id
            job_dir.mkdir(exist_ok=True, parents=True)
            
            # 加载原始图像
            original_image = self._load_image(input_path)
            
            # 专业织机处理流水线
            processed_image = self._professional_pipeline(original_image, color_count)
            
            # 专业模式 - 添加装饰效果
            if self.config["decorative_border"] or self.config["complex_decoration"]:
                final_image = self._add_decorations(processed_image)
                logger.info("  ✓ 专业装饰效果已应用")
            else:
                final_image = processed_image
            
            # 创建对比图像
            comparison_image = self._create_comparison(original_image, final_image)
            
            # 保存结果
            professional_path = job_dir / f"{job_id}_professional.png"
            comparison_path = job_dir / f"{job_id}_comparison.png"
            
            self._save_image(final_image, str(professional_path))
            self._save_image(comparison_image, str(comparison_path))
            
            processing_time = time.time() - start_time
            logger.info(f"🎯 专业织机图像生成完成，耗时: {processing_time:.2f}秒")
            
            return str(professional_path), str(comparison_path), processing_time
            
        except Exception as e:
            error_msg = f"专业织机图像生成失败: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise Exception(error_msg)
    
    def _load_image(self, input_path: str) -> np.ndarray:
        """加载图像"""
        try:
            with Image.open(input_path) as pil_image:
                if pil_image.mode != 'RGB':
                    pil_image = pil_image.convert('RGB')
                return np.array(pil_image)
        except Exception as e:
            raise Exception(f"图像加载失败: {str(e)}")
    
    def _professional_pipeline(self, image: np.ndarray, color_count: int) -> np.ndarray:
        """专业织机处理流水线 - 重新设计：先锐化后简化"""
        try:
            # 1. 预锐化：先增强边缘，防止后续处理模糊化
            pre_sharpened = self._extreme_pre_sharpening(image)
            logger.info("  ✓ 预锐化处理完成")
            
            # 2. 智能颜色降色（保持锐度）
            color_reduced = self._intelligent_color_reduction_preserve_edges(pre_sharpened, color_count)
            logger.info("  ✓ 智能颜色降色完成")
            
            # 3. 颜色区域连贯性增强（轻度处理，保持边缘）
            coherent = self._enhance_color_coherence_preserve_edges(color_reduced)
            logger.info("  ✓ 颜色区域连贯性增强完成")
            
            # 4. 最终极度锐化
            ultra_sharp = self._final_extreme_sharpening(coherent)
            logger.info("  ✓ 最终极度锐化完成")
            
            # 5. 织机识别优化（轻度处理）
            weaving_optimized = self._weaving_optimization_light(ultra_sharp)
            logger.info("  ✓ 织机识别优化完成")
            
            # 6. 质量增强（不使用抗锯齿，保持锐度）
            enhanced = self._quality_enhancement_sharp(weaving_optimized)
            logger.info("  ✓ 质量增强完成")
            
            # 7. 专业级艺术化处理
            if self.config["professional_enhancement"]:
                artistic_enhanced = self._professional_artistic_enhancement(enhanced)
                logger.info("  ✓ 专业级艺术化处理完成")
                return artistic_enhanced
            
            return enhanced
            
        except Exception as e:
            raise Exception(f"专业织机处理失败: {str(e)}")
    
    def _extreme_pre_sharpening(self, image: np.ndarray) -> np.ndarray:
        """预锐化处理 - 在所有处理之前先极度锐化"""
        try:
            # 超强锐化核
            kernel = np.array([
                [-4, -4, -4],
                [-4, 32, -4],
                [-4, -4, -4]
            ], dtype=np.float32)
            
            # 应用锐化
            sharpened = cv2.filter2D(image, -1, kernel)
            
            # 多重Unsharp Mask
            for sigma in [0.3, 0.7, 1.2]:
                gaussian = cv2.GaussianBlur(image, (0, 0), sigma)
                unsharp = cv2.addWeighted(image, 4.0, gaussian, -3.0, 0)
                sharpened = cv2.addWeighted(sharpened, 0.8, unsharp, 0.2, 0)
            
            return np.clip(sharpened, 0, 255).astype(np.uint8)
            
        except Exception as e:
            logger.warning(f"预锐化失败: {str(e)}")
            return image
    
    def _intelligent_color_reduction_preserve_edges(self, image: np.ndarray, n_colors: int) -> np.ndarray:
        """智能颜色降色 - 保持边缘锐度版本"""
        try:
            # 保存原始边缘信息
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 30, 100)
            
            # 极度降色到更少颜色
            reduced_colors = max(4, min(8, n_colors // 3))  # 更激进的降色
            
            # K-means聚类（不进行预模糊，保持锐度）
            pixels = image.reshape(-1, 3)
            kmeans = KMeans(
                n_clusters=reduced_colors,
                init='k-means++',
                n_init=50,
                max_iter=1000,
                random_state=42
            )
            
            kmeans.fit(pixels)
            colors = kmeans.cluster_centers_.astype(np.uint8)
            
            # 极度优化颜色
            colors = self._optimize_colors_extreme(colors)
            
            # 替换像素颜色
            labels = kmeans.labels_
            reduced_pixels = colors[labels]
            result = reduced_pixels.reshape(image.shape)
            
            # 在边缘区域保持原始锐度
            edges_3d = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB) / 255.0
            result = result * (1 - edges_3d * 0.5) + image * (edges_3d * 0.5)
            
            # 色彩量化
            result = (result // 16) * 16  # 更激进的量化
            
            return np.clip(result, 0, 255).astype(np.uint8)
            
        except Exception as e:
            raise Exception(f"颜色降色失败: {str(e)}")
    
    def _enhance_color_coherence_preserve_edges(self, image: np.ndarray) -> np.ndarray:
        """颜色连贯性增强 - 保持边缘版本"""
        try:
            # 保存边缘信息
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 30, 100)
            
            # 轻度连贯性处理
            unique_colors = self._get_unique_colors(image)
            result = image.copy()
            
            for color in unique_colors[:20]:  # 只处理前20种主要颜色
                color_np = np.array(color)
                diff = np.abs(result.astype(np.int16) - color_np.astype(np.int16))
                mask = np.sum(diff, axis=2) < 30
                
                if np.sum(mask) == 0:
                    continue
                
                # 轻度形态学操作
                kernel_size = 7  # 小核心，保持细节
                kernel = np.ones((kernel_size, kernel_size), np.uint8)
                
                mask_processed = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
                result[mask_processed > 0] = color
            
            # 在非边缘区域轻度平滑
            edges_3d = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB) / 255.0
            smoothed = cv2.bilateralFilter(result, 5, 50, 50)  # 轻度平滑
            result = result * edges_3d + smoothed * (1 - edges_3d)
            
            return result.astype(np.uint8)
            
        except Exception as e:
            raise Exception(f"颜色连贯性增强失败: {str(e)}")
    
    def _final_extreme_sharpening(self, image: np.ndarray) -> np.ndarray:
        """最终极度锐化 - 专业软件级别的边缘极化"""
        try:
            # 第一轮：边缘检测和极化
            edges_polarized = self._polarize_edges(image)
            
            # 第二轮：超强锐化核
            kernel1 = np.array([
                [-6, -6, -6],
                [-6, 48, -6],
                [-6, -6, -6]
            ], dtype=np.float32)
            
            sharpened1 = cv2.filter2D(edges_polarized, -1, kernel1)
            
            # 第三轮：多级边缘增强
            kernel2 = np.array([
                [-2, -3, -2],
                [-3, 16, -3],
                [-2, -3, -2]
            ], dtype=np.float32)
            
            sharpened2 = cv2.filter2D(sharpened1, -1, kernel2)
            
            # 第四轮：高通滤波增强
            gaussian = cv2.GaussianBlur(image, (0, 0), 0.3)
            high_pass = cv2.addWeighted(image, 6.0, gaussian, -5.0, 0)
            
            # 第五轮：拉普拉斯锐化
            laplacian_kernel = np.array([
                [0, -1, 0],
                [-1, 5, -1],
                [0, -1, 0]
            ], dtype=np.float32)
            
            laplacian_sharp = cv2.filter2D(sharpened2, -1, laplacian_kernel)
            
            # 混合所有锐化结果
            final_result = cv2.addWeighted(laplacian_sharp, 0.6, high_pass, 0.4, 0)
            
            # 最终边缘强化
            final_result = self._enhance_edge_contrast(final_result)
            
            return np.clip(final_result, 0, 255).astype(np.uint8)
            
        except Exception as e:
            logger.warning(f"最终锐化失败: {str(e)}")
            return image
    
    def _polarize_edges(self, image: np.ndarray) -> np.ndarray:
        """边缘极化处理 - 模拟专业软件的极锐边缘"""
        try:
            # 检测边缘
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            # 多方向边缘检测
            sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            
            # 计算梯度幅值
            gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
            
            # 边缘极化：增强强边缘，抑制弱边缘
            threshold = np.percentile(gradient_magnitude, 70)  # 70%分位数作为阈值
            
            # 创建边缘掩码
            strong_edges = gradient_magnitude > threshold
            weak_edges = (gradient_magnitude > threshold * 0.3) & (gradient_magnitude <= threshold)
            
            # 对原图像进行边缘增强
            result = image.copy().astype(np.float32)
            
            # 在强边缘区域极度锐化
            for i in range(3):  # RGB三个通道
                channel = result[:, :, i]
                # 强边缘区域：极度增强对比度
                channel[strong_edges] = channel[strong_edges] * 1.5
                # 弱边缘区域：轻度增强
                channel[weak_edges] = channel[weak_edges] * 1.2
                result[:, :, i] = channel
            
            return np.clip(result, 0, 255).astype(np.uint8)
            
        except Exception as e:
            logger.warning(f"边缘极化失败: {str(e)}")
            return image
    
    def _enhance_edge_contrast(self, image: np.ndarray) -> np.ndarray:
        """增强边缘对比度 - 最终边缘强化"""
        try:
            # 检测边缘
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 20, 80)
            
            # 膨胀边缘，创建边缘区域
            kernel = np.ones((3, 3), np.uint8)
            edge_regions = cv2.dilate(edges, kernel, iterations=1)
            
            # 在边缘区域增强对比度
            result = image.copy().astype(np.float32)
            edge_mask = edge_regions > 0
            
            for i in range(3):  # RGB三个通道
                channel = result[:, :, i]
                # 在边缘区域应用S型对比度增强
                edge_pixels = channel[edge_mask]
                # S型曲线：暗的更暗，亮的更亮
                normalized = edge_pixels / 255.0
                enhanced = np.where(normalized < 0.5, 
                                  2 * normalized**2, 
                                  1 - 2 * (1 - normalized)**2)
                channel[edge_mask] = enhanced * 255
                result[:, :, i] = channel
            
            return np.clip(result, 0, 255).astype(np.uint8)
            
        except Exception as e:
            logger.warning(f"边缘对比度增强失败: {str(e)}")
            return image
    
    def _weaving_optimization_light(self, image: np.ndarray) -> np.ndarray:
        """轻度织机识别优化 - 保持锐度"""
        try:
            # 只做轻微的噪点清理，不影响边缘
            cleaned = cv2.medianBlur(image, 3)  # 小核心
            return cleaned
            
        except Exception as e:
            logger.warning(f"织机优化失败: {str(e)}")
            return image
    
    def _quality_enhancement_sharp(self, image: np.ndarray) -> np.ndarray:
        """质量增强 - 保持锐度版本"""
        try:
            # 不使用抗锯齿，直接进行色彩调整
            pil_image = Image.fromarray(image)
            
            # 极度增强对比度和饱和度
            enhancer = ImageEnhance.Contrast(pil_image)
            enhanced = enhancer.enhance(self.config["contrast_boost"] * 1.5)
            
            enhancer = ImageEnhance.Color(enhanced)
            enhanced = enhancer.enhance(self.config["color_saturation_boost"] * 1.2)
            
            # 增强锐度
            enhancer = ImageEnhance.Sharpness(enhanced)
            enhanced = enhancer.enhance(2.0)  # 2倍锐度
            
            return np.array(enhanced)
            
        except Exception as e:
            raise Exception(f"质量增强失败: {str(e)}")
    
    def _professional_artistic_enhancement(self, image: np.ndarray) -> np.ndarray:
        """专业级艺术化增强 - 对标专业织机软件"""
        try:
            # 1. 极致边缘锐化
            ultra_sharpened = self._ultra_edge_sharpening(image)
            
            # 2. 艺术化背景生成
            if self.config["artistic_background"]:
                artistic_bg = self._generate_artistic_background(ultra_sharpened)
            else:
                artistic_bg = ultra_sharpened
            
            # 3. 复杂装饰图案
            if self.config["complex_decoration"]:
                decorated = self._add_complex_decorations(artistic_bg)
            else:
                decorated = artistic_bg
            
            return decorated
            
        except Exception as e:
            logger.warning(f"专业级艺术化增强失败: {str(e)}")
            return image
    
    def _ultra_edge_sharpening(self, image: np.ndarray) -> np.ndarray:
        """极致边缘锐化"""
        try:
            # 创建超强锐化核
            intensity = self.config["edge_sharpening_intensity"]
            kernel = np.array([
                [-1, -1, -1, -1, -1],
                [-1, -2, -2, -2, -1],
                [-1, -2, 16+intensity*2, -2, -1],
                [-1, -2, -2, -2, -1],
                [-1, -1, -1, -1, -1]
            ]) / 9
            
            # 应用极致锐化
            ultra_sharp = cv2.filter2D(image, -1, kernel)
            
            # 与原图混合，保持细节
            result = cv2.addWeighted(image, 0.3, ultra_sharp, 0.7, 0)
            
            return result.astype(np.uint8)
            
        except Exception as e:
            logger.warning(f"极致边缘锐化失败: {str(e)}")
            return image
    
    def _generate_artistic_background(self, image: np.ndarray) -> np.ndarray:
        """生成AI增强的艺术化背景"""
        try:
            height, width = image.shape[:2]
            
            # AI增强的主体检测
            main_object_mask = self._detect_main_object_enhanced(image)
            
            # 使用AI生成器创建专业背景
            if self.ai_generator is not None:
                try:
                    # 分析主体类型
                    subject_type = "熊猫"  # 默认，后续可以通过AI分析得出
                    if self.ai_processor is not None:
                        subject_info = self.ai_processor.analyze_image_content(image)
                        subject_type = subject_info.get('main_subject', '熊猫')
                    
                    # 生成专业级背景
                    professional_background = self.ai_generator.generate_professional_background(
                        width, height, subject_type
                    )
                    logger.info("AI专业背景生成成功")
                    
                except Exception as e:
                    logger.warning(f"AI背景生成失败，使用传统方法: {str(e)}")
                    professional_background = self._create_traditional_floral_pattern(height, width)
            else:
                # 回退到传统方法
                professional_background = self._create_traditional_floral_pattern(height, width)
            
            # 智能融合：绝对保护主体，装饰背景
            artistic_image = image.copy()
            background_mask = ~main_object_mask
            
            if np.sum(background_mask) > 0:
                # 背景区域用专业图案
                background_regions = artistic_image[background_mask]
                professional_regions = professional_background[background_mask]
                
                # 20% 专业图案 + 80% 原始色调 (保守融合)
                blended = cv2.addWeighted(professional_regions, 0.2, background_regions, 0.8, 0)
                artistic_image[background_mask] = blended
            
            # 绝对保护主体区域
            artistic_image[main_object_mask] = image[main_object_mask]
            
            return artistic_image
            
        except Exception as e:
            logger.warning(f"AI艺术化背景生成失败: {str(e)}")
            return image
    
    def _detect_main_object_enhanced(self, image: np.ndarray) -> np.ndarray:
        """AI增强的主体对象检测"""
        try:
            # 尝试使用AI增强检测
            if self.ai_processor is not None:
                try:
                    # AI分析图像内容
                    subject_info = self.ai_processor.analyze_image_content(image)
                    logger.info(f"AI分析结果: {subject_info.get('main_subject', 'unknown')}")
                    
                    # 基于AI结果生成智能掩码
                    ai_mask = self.ai_processor.generate_smart_mask(image, subject_info)
                    
                    # 验证掩码有效性
                    if np.sum(ai_mask) > 0:
                        return ai_mask
                    else:
                        logger.warning("AI生成的掩码无效，回退到传统方法")
                        
                except Exception as e:
                    logger.warning(f"AI增强检测失败: {str(e)}")
            
            # 回退到传统检测方法
            return self._detect_main_object_fallback(image)
            
        except Exception as e:
            logger.warning(f"主体检测失败: {str(e)}")
            return self._detect_main_object_fallback(image)
    
    def _detect_texture_similarity(self, image: np.ndarray, center_x: int, center_y: int) -> np.ndarray:
        """基于纹理相似性的区域检测"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            height, width = gray.shape
            
            # 简化的纹理特征（局部标准差）
            kernel_size = 9
            kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)
            mean_filtered = cv2.filter2D(gray.astype(np.float32), -1, kernel)
            sqr_filtered = cv2.filter2D((gray.astype(np.float32))**2, -1, kernel)
            texture = np.sqrt(np.maximum(0, sqr_filtered - mean_filtered**2))
            
            # 获取中心区域的纹理特征
            center_texture = texture[center_y, center_x]
            
            # 相似性掩码
            texture_diff = np.abs(texture - center_texture)
            similarity_mask = texture_diff < (center_texture * 0.5 + 10)
            
            return similarity_mask
            
        except Exception as e:
            logger.warning(f"纹理相似性检测失败: {str(e)}")
            return np.ones(image.shape[:2], dtype=bool)
    
    def _detect_main_object_fallback(self, image: np.ndarray) -> np.ndarray:
        """主体检测回退方法"""
        try:
            height, width = image.shape[:2]
            
            # 基于边缘检测的简单方法
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            
            # 形态学操作连接边缘
            kernel = np.ones((5, 5), np.uint8)
            edges_closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
            
            # 填充内部区域
            contours, _ = cv2.findContours(edges_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            mask = np.zeros((height, width), dtype=np.uint8)
            
            if contours:
                # 选择面积最大的轮廓
                largest_contour = max(contours, key=cv2.contourArea)
                cv2.fillPoly(mask, [largest_contour], 255)
            
            return mask > 0
            
        except Exception as e:
            logger.warning(f"回退主体检测失败: {str(e)}")
            return np.zeros(image.shape[:2], dtype=bool)
    
    def _create_traditional_floral_pattern(self, height: int, width: int) -> np.ndarray:
        """创建传统花卉图案"""
        try:
            # 创建花卉背景图案
            pattern = np.zeros((height, width, 3), dtype=np.uint8)
            
            # 定义传统色彩调色板
            colors = [
                (139, 69, 19),    # 褐色
                (160, 82, 45),    # 鞍褐色  
                (210, 180, 140),  # 棕褐色
                (222, 184, 135),  # 浅褐色
                (245, 245, 220),  # 米色
                (255, 228, 196),  # 浅棕色
                (205, 133, 63),   # 秘鲁色
                (188, 143, 143),  # 玫瑰褐色
            ]
            
            # 生成基础渐变背景
            for y in range(height):
                for x in range(width):
                    # 基于位置的颜色变化
                    color_idx = int((x / width + y / height) * len(colors) / 2) % len(colors)
                    pattern[y, x] = colors[color_idx]
            
            # 添加花卉图案纹理
            pattern = self._add_floral_texture(pattern)
            
            # 添加细节装饰
            pattern = self._add_traditional_details(pattern)
            
            return pattern
            
        except Exception as e:
            logger.warning(f"传统花卉图案创建失败: {str(e)}")
            return np.full((height, width, 3), (139, 69, 19), dtype=np.uint8)
    
    def _add_floral_texture(self, pattern: np.ndarray) -> np.ndarray:
        """添加花卉纹理"""
        try:
            height, width = pattern.shape[:2]
            
            # 使用正弦波生成花卉图案
            for y in range(0, height, 25):
                for x in range(0, width, 25):
                    # 生成花朵中心
                    center_x, center_y = x + 12, y + 12
                    if center_x < width and center_y < height:
                        
                        # 花朵主体（圆形渐变）
                        for dy in range(-8, 9):
                            for dx in range(-8, 9):
                                px, py = center_x + dx, center_y + dy
                                if 0 <= px < width and 0 <= py < height:
                                    distance = np.sqrt(dx*dx + dy*dy)
                                    if distance <= 8:
                                        # 花朵颜色（更亮的色调）
                                        intensity = max(0, 1 - distance/8)
                                        flower_color = np.array([200, 150, 100]) * intensity
                                        pattern[py, px] = np.clip(pattern[py, px] + flower_color, 0, 255)
                        
                        # 花瓣（椭圆形）
                        for angle in [0, 60, 120, 180, 240, 300]:
                            rad = np.radians(angle)
                            petal_x = center_x + int(12 * np.cos(rad))
                            petal_y = center_y + int(12 * np.sin(rad))
                            
                            if 0 <= petal_x < width and 0 <= petal_y < height:
                                # 花瓣椭圆
                                cv2.ellipse(pattern, (petal_x, petal_y), (6, 3), angle, 0, 360, 
                                          (180, 120, 80), -1)
            
            return pattern
            
        except Exception as e:
            logger.warning(f"花卉纹理添加失败: {str(e)}")
            return pattern
    
    def _add_traditional_details(self, pattern: np.ndarray) -> np.ndarray:
        """添加传统装饰细节"""
        try:
            height, width = pattern.shape[:2]
            
            # 添加藤蔓连接线
            for y in range(0, height, 50):
                for x in range(0, width-50, 50):
                    # S形藤蔓
                    points = []
                    for t in range(0, 51, 5):
                        curve_x = x + t
                        curve_y = y + int(10 * np.sin(t * 0.2))
                        if 0 <= curve_x < width and 0 <= curve_y < height:
                            points.append([curve_x, curve_y])
                    
                    if len(points) > 1:
                        points = np.array(points, dtype=np.int32)
                        cv2.polylines(pattern, [points], False, (100, 80, 60), 2)
            
            # 添加小叶子
            for y in range(15, height, 40):
                for x in range(15, width, 40):
                    if x < width and y < height:
                        # 叶子形状
                        leaf_points = np.array([
                            [x, y-5], [x+3, y-2], [x+5, y], 
                            [x+3, y+2], [x, y+5], [x-3, y+2],
                            [x-5, y], [x-3, y-2]
                        ], dtype=np.int32)
                        
                        cv2.fillPoly(pattern, [leaf_points], (120, 100, 70))
            
            return pattern
            
        except Exception as e:
            logger.warning(f"传统装饰细节添加失败: {str(e)}")
            return pattern
    
    def _add_mini_pattern(self, image: np.ndarray, x: int, y: int):
        """添加小型艺术图案"""
        try:
            height, width = image.shape[:2]
            pattern_size = 15
            
            if x + pattern_size >= width or y + pattern_size >= height:
                return
            
            # 获取当前区域的主导色
            region = image[y:y+pattern_size, x:x+pattern_size]
            main_color = np.mean(region.reshape(-1, 3), axis=0).astype(np.uint8)
            
            # 生成对比色
            contrast_color = 255 - main_color
            
            # 添加小型几何图案
            pattern_type = (x + y) % 4
            
            if pattern_type == 0:
                # 小圆点
                cv2.circle(image, (x+7, y+7), 3, contrast_color.tolist(), -1)
            elif pattern_type == 1:
                # 小菱形
                pts = np.array([[x+7, y+2], [x+12, y+7], [x+7, y+12], [x+2, y+7]], np.int32)
                cv2.fillPoly(image, [pts], contrast_color.tolist())
            elif pattern_type == 2:
                # 小十字
                cv2.line(image, (x+3, y+7), (x+11, y+7), contrast_color.tolist(), 2)
                cv2.line(image, (x+7, y+3), (x+7, y+11), contrast_color.tolist(), 2)
            else:
                # 小方形
                cv2.rectangle(image, (x+4, y+4), (x+10, y+10), contrast_color.tolist(), 1)
                
        except Exception as e:
            logger.warning(f"小型图案添加失败: {str(e)}")
    
    def _add_complex_decorations(self, image: np.ndarray) -> np.ndarray:
        """添加复杂装饰图案"""
        try:
            height, width = image.shape[:2]
            decorated = image.copy()
            
            # 在边缘区域添加更复杂的装饰
            border_width = 30
            
            # 顶部和底部装饰
            for y in [10, height-40]:
                for x in range(border_width, width-border_width, 25):
                    self._add_decorative_motif(decorated, x, y)
            
            # 左侧和右侧装饰
            for x in [10, width-40]:
                for y in range(border_width, height-border_width, 25):
                    self._add_decorative_motif(decorated, x, y)
            
            return decorated
            
        except Exception as e:
            logger.warning(f"复杂装饰添加失败: {str(e)}")
            return image
    
    def _add_decorative_motif(self, image: np.ndarray, x: int, y: int):
        """添加装饰主题图案"""
        try:
            height, width = image.shape[:2]
            if x + 20 >= width or y + 20 >= height or x < 0 or y < 0:
                return
            
            # 蜀锦风格装饰色彩
            colors = [
                (255, 215, 0),    # 金色
                (220, 20, 60),    # 深红
                (0, 100, 0),      # 深绿
                (139, 69, 19),    # 棕色
                (75, 0, 130),     # 紫色
            ]
            
            color = colors[(x + y) % len(colors)]
            
            # 绘制复杂图案
            # 外圈
            cv2.circle(image, (x+10, y+10), 8, color, 2)
            # 内部装饰
            cv2.circle(image, (x+10, y+10), 4, color, 1)
            # 十字装饰
            cv2.line(image, (x+6, y+10), (x+14, y+10), color, 1)
            cv2.line(image, (x+10, y+6), (x+10, y+14), color, 1)
            # 角落点缀
            for dx, dy in [(2, 2), (18, 2), (2, 18), (18, 18)]:
                cv2.circle(image, (x+dx, y+dy), 1, color, -1)
                
        except Exception as e:
            logger.warning(f"装饰图案添加失败: {str(e)}")

    def _add_decorations(self, image: np.ndarray) -> np.ndarray:
        """添加专业装饰"""
        try:
            if not self.config["decorative_border"]:
                return image
            
            pil_image = Image.fromarray(image)
            width, height = pil_image.size
            
            # 创建带边框的图像
            border_width = 20
            new_width = width + 2 * border_width
            new_height = height + 2 * border_width
            
            # 粉色边框（模仿专业软件）
            border_color = (255, 192, 203)
            bordered_image = Image.new('RGB', (new_width, new_height), border_color)
            
            # 粘贴原图像
            bordered_image.paste(pil_image, (border_width, border_width))
            
            # 添加装饰图案
            bordered_image = self._add_decorative_pattern(bordered_image)
            
            return np.array(bordered_image)
            
        except Exception as e:
            logger.warning(f"装饰添加失败: {str(e)}")
            return image
    
    def _add_decorative_pattern(self, pil_image: Image.Image) -> Image.Image:
        """添加装饰图案"""
        try:
            draw = ImageDraw.Draw(pil_image)
            width, height = pil_image.size
            pattern_color = (255, 255, 255)
            
            # 在边框上绘制装饰圆点
            for i in range(10, width-10, 30):
                draw.ellipse([i-3, 7, i+3, 13], fill=pattern_color)
                draw.ellipse([i-3, height-13, i+3, height-7], fill=pattern_color)
            
            for i in range(10, height-10, 30):
                draw.ellipse([7, i-3, 13, i+3], fill=pattern_color)
                draw.ellipse([width-13, i-3, width-7, i+3], fill=pattern_color)
            
            return pil_image
            
        except Exception as e:
            logger.warning(f"装饰图案失败: {str(e)}")
            return pil_image
    
    def _create_comparison(self, original: np.ndarray, processed: np.ndarray) -> np.ndarray:
        """创建对比图像"""
        try:
            # 确保尺寸一致
            if original.shape != processed.shape:
                original = cv2.resize(original, (processed.shape[1], processed.shape[0]))
            
            # 创建并排对比
            comparison = np.hstack([original, processed])
            
            # 添加标注
            pil_comparison = Image.fromarray(comparison)
            draw = ImageDraw.Draw(pil_comparison)
            
            width, height = pil_comparison.size
            
            # 标注文字
            try:
                # 尝试使用默认字体
                font = ImageFont.load_default()
            except:
                font = None
            
            draw.text((width//4, 10), "原始图像", fill=(255, 255, 255), font=font, anchor="mm")
            draw.text((3*width//4, 10), "专业织机处理", fill=(255, 255, 255), font=font, anchor="mm")
            
            # 分隔线
            draw.line([(width//2, 0), (width//2, height)], fill=(255, 255, 255), width=2)
            
            return np.array(pil_comparison)
            
        except Exception as e:
            logger.warning(f"对比图像创建失败: {str(e)}")
            return processed
    
    def _save_image(self, image: np.ndarray, output_path: str):
        """保存图像"""
        try:
            pil_image = Image.fromarray(image)
            
            save_kwargs = {
                'format': 'PNG',
                'optimize': False,
                'compress_level': 0,
            }
            
            pil_image.save(output_path, **save_kwargs)
            
            file_size = Path(output_path).stat().st_size
            logger.info(f"图像已保存: {Path(output_path).name} ({file_size/1024/1024:.2f} MB)")
            
        except Exception as e:
            raise Exception(f"图像保存失败: {str(e)}")
    
    def _get_unique_colors(self, image: np.ndarray) -> List[Tuple[int, int, int]]:
        """获取唯一颜色"""
        try:
            pixels = image.reshape(-1, 3)
            unique_colors = np.unique(pixels, axis=0)
            return [tuple(color) for color in unique_colors]
        except Exception as e:
            logger.warning(f"获取颜色失败: {str(e)}")
            return []

    def _intelligent_color_reduction(self, image: np.ndarray, n_colors: int) -> np.ndarray:
        """智能颜色降色 - 原版本保留作为备用"""
        try:
            # 预处理：轻微模糊减少噪声
            blurred = cv2.GaussianBlur(image, (3, 3), 0.8)
            
            # K-means聚类
            pixels = blurred.reshape(-1, 3)
            kmeans = KMeans(
                n_clusters=n_colors,
                init='k-means++',
                n_init=30,
                max_iter=500,
                random_state=42
            )
            
            kmeans.fit(pixels)
            colors = kmeans.cluster_centers_.astype(np.uint8)
            
            # 颜色优化：增强饱和度
            colors = self._optimize_colors(colors)
            
            # 替换像素颜色
            labels = kmeans.labels_
            reduced_pixels = colors[labels]
            
            return reduced_pixels.reshape(image.shape)
            
        except Exception as e:
            raise Exception(f"颜色降色失败: {str(e)}")
    
    def _optimize_colors(self, colors: np.ndarray) -> np.ndarray:
        """优化颜色"""
        try:
            optimized = []
            
            for color in colors:
                # 转换为HSV调整饱和度
                hsv = cv2.cvtColor(color.reshape(1, 1, 3), cv2.COLOR_RGB2HSV)[0, 0]
                
                # 增强饱和度
                hsv[1] = min(255, int(hsv[1] * self.config["color_saturation_boost"]))
                
                # 调整亮度增强对比度
                if hsv[2] > 128:
                    hsv[2] = min(255, int(hsv[2] * 1.1))
                else:
                    hsv[2] = max(0, int(hsv[2] * 0.9))
                
                # 转换回RGB
                rgb = cv2.cvtColor(hsv.reshape(1, 1, 3), cv2.COLOR_HSV2RGB)[0, 0]
                optimized.append(rgb)
            
            return np.array(optimized, dtype=np.uint8)
            
        except Exception as e:
            logger.warning(f"颜色优化失败: {str(e)}")
            return colors
    
    def _optimize_colors_extreme(self, colors: np.ndarray) -> np.ndarray:
        """极度颜色优化 - 模拟专业织机识别软件效果"""
        try:
            optimized = []
            
            for color in colors:
                # 转换为HSV进行极度调整
                hsv = cv2.cvtColor(color.reshape(1, 1, 3), cv2.COLOR_RGB2HSV)[0, 0]
                
                # 极度增强饱和度（模拟专业软件的高饱和度）
                hsv[1] = min(255, int(hsv[1] * 2.5))  # 2.5倍饱和度
                
                # 极度对比度调整（模拟专业软件的黑白分明效果）
                if hsv[2] > 128:
                    hsv[2] = min(255, int(hsv[2] * 1.3))  # 亮部更亮
                else:
                    hsv[2] = max(20, int(hsv[2] * 0.7))   # 暗部更暗，但保留一定可见度
                
                # 色相量化，减少色相变化（模拟专业软件的纯色效果）
                hsv[0] = (hsv[0] // 20) * 20  # 色相量化到20度的倍数
                
                # 转换回RGB
                rgb = cv2.cvtColor(hsv.reshape(1, 1, 3), cv2.COLOR_HSV2RGB)[0, 0]
                
                # 进一步RGB量化，确保纯色效果
                rgb = (rgb // 32) * 32  # 量化到32的倍数
                
                optimized.append(rgb)
            
            return np.array(optimized, dtype=np.uint8)
            
        except Exception as e:
            logger.warning(f"极度颜色优化失败: {str(e)}")
            return colors