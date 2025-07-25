"""
刺绣优化器
改进颜色量化、边缘检测、纹理细节和图案连续性
"""

import cv2
import numpy as np
import logging
import time
from typing import Dict, Any, Tuple, List, Optional
from dataclasses import dataclass
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import pairwise_distances
import colorsys
from scipy import ndimage
from scipy.spatial import cKDTree
import json
import os

logger = logging.getLogger(__name__)


@dataclass
class OptimizationParams:
    """优化参数"""
    color_clusters: int = 16
    edge_smoothness: float = 1.5
    texture_detail_level: float = 0.8
    pattern_continuity_weight: float = 0.6
    noise_reduction: float = 0.3
    contrast_enhancement: float = 1.2


@dataclass
class OptimizationResult:
    """优化结果"""
    optimized_image: np.ndarray
    original_image: np.ndarray
    optimization_params: OptimizationParams
    quality_improvements: Dict[str, float]
    processing_time: float


class EmbroideryOptimizer:
    """刺绣优化器"""
    
    def __init__(self):
        self.optimization_history = []
        
    def optimize_embroidery(self, 
                          image: np.ndarray, 
                          params: OptimizationParams = None,
                          target_style: str = "realistic") -> OptimizationResult:
        """优化刺绣图像"""
        logger.info("开始刺绣优化...")
        
        if params is None:
            params = OptimizationParams()
        
        start_time = time.time()
        original_image = image.copy()
        
        # 记录原始质量指标
        original_metrics = self._calculate_quality_metrics(original_image)
        
        # 执行优化步骤
        optimized_image = self._apply_optimization_pipeline(original_image, params, target_style)
        
        # 计算优化后的质量指标
        optimized_metrics = self._calculate_quality_metrics(optimized_image)
        
        # 计算改进程度
        improvements = self._calculate_improvements(original_metrics, optimized_metrics)
        
        processing_time = time.time() - start_time
        
        # 创建优化结果
        result = OptimizationResult(
            optimized_image=optimized_image,
            original_image=original_image,
            optimization_params=params,
            quality_improvements=improvements,
            processing_time=processing_time
        )
        
        self.optimization_history.append(result)
        logger.info(f"优化完成，处理时间: {processing_time:.2f}秒")
        
        return result
    
    def _apply_optimization_pipeline(self, 
                                   image: np.ndarray, 
                                   params: OptimizationParams,
                                   target_style: str) -> np.ndarray:
        """应用优化流水线"""
        optimized = image.copy()
        
        # 1. 颜色优化
        optimized = self._optimize_colors(optimized, params.color_clusters, target_style)
        
        # 2. 边缘优化
        optimized = self._optimize_edges(optimized, params.edge_smoothness)
        
        # 3. 纹理优化
        optimized = self._optimize_texture(optimized, params.texture_detail_level)
        
        # 4. 图案连续性优化
        optimized = self._optimize_pattern_continuity(optimized, params.pattern_continuity_weight)
        
        # 5. 噪声减少
        optimized = self._reduce_noise(optimized, params.noise_reduction)
        
        # 6. 对比度增强
        optimized = self._enhance_contrast(optimized, params.contrast_enhancement)
        
        return optimized
    
    def _optimize_colors(self, image: np.ndarray, n_clusters: int, target_style: str) -> np.ndarray:
        """优化颜色量化"""
        logger.info(f"优化颜色量化，聚类数: {n_clusters}")
        
        # 转换到LAB色彩空间
        lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        
        # 重塑图像数据
        pixels = lab_image.reshape(-1, 3)
        
        # 使用K-means聚类
        kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, batch_size=1000)
        labels = kmeans.fit_predict(pixels)
        
        # 获取聚类中心
        centers = kmeans.cluster_centers_
        
        # 根据目标风格调整颜色
        if target_style == "realistic":
            centers = self._adjust_colors_for_realism(centers)
        elif target_style == "artistic":
            centers = self._adjust_colors_for_artistic(centers)
        
        # 重建图像
        quantized_pixels = centers[labels]
        quantized_lab = quantized_pixels.reshape(lab_image.shape)
        
        # 转换回BGR
        optimized_image = cv2.cvtColor(quantized_lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
        
        return optimized_image
    
    def _optimize_edges(self, image: np.ndarray, smoothness: float) -> np.ndarray:
        """优化边缘质量"""
        logger.info(f"优化边缘质量，平滑度: {smoothness}")
        
        # 转换为灰度图
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 边缘检测
        edges = cv2.Canny(gray, 50, 150)
        
        # 边缘平滑
        kernel_size = int(3 + smoothness * 2)
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        smoothed_edges = cv2.GaussianBlur(edges, (kernel_size, kernel_size), smoothness)
        
        # 形态学操作改善边缘连续性
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        closed_edges = cv2.morphologyEx(smoothed_edges, cv2.MORPH_CLOSE, kernel)
        
        # 将边缘信息融合到原图像
        edge_weight = 0.3
        enhanced_image = image.copy()
        
        for i in range(3):  # BGR通道
            enhanced_image[:, :, i] = cv2.addWeighted(
                enhanced_image[:, :, i], 1 - edge_weight,
                closed_edges, edge_weight, 0
            )
        
        return enhanced_image
    
    def _optimize_texture(self, image: np.ndarray, detail_level: float) -> np.ndarray:
        """优化纹理细节"""
        logger.info(f"优化纹理细节，细节级别: {detail_level}")
        
        # 转换为灰度图
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 计算局部方差作为纹理指标
        kernel_size = 5
        mean_filtered = cv2.blur(gray, (kernel_size, kernel_size))
        variance = cv2.blur(gray.astype(float)**2, (kernel_size, kernel_size)) - mean_filtered.astype(float)**2
        
        # 归一化方差
        variance = (variance - variance.min()) / (variance.max() - variance.min() + 1e-8)
        
        # 根据细节级别调整纹理
        texture_enhanced = gray + detail_level * 50 * variance
        
        # 限制像素值范围
        texture_enhanced = np.clip(texture_enhanced, 0, 255).astype(np.uint8)
        
        # 将纹理信息融合到原图像
        enhanced_image = image.copy()
        for i in range(3):
            enhanced_image[:, :, i] = cv2.addWeighted(
                enhanced_image[:, :, i], 0.7,
                texture_enhanced, 0.3, 0
            )
        
        return enhanced_image
    
    def _optimize_pattern_continuity(self, image: np.ndarray, continuity_weight: float) -> np.ndarray:
        """优化图案连续性"""
        logger.info(f"优化图案连续性，权重: {continuity_weight}")
        
        # 转换为灰度图
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 计算连通区域
        _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
        
        # 形态学操作改善连通性
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        connected = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        # 计算区域属性
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(connected, connectivity=8)
        
        # 改善小区域的连续性
        min_area = 50
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] < min_area:
                # 填充小区域
                mask = (labels == i).astype(np.uint8)
                filled = cv2.dilate(mask, kernel, iterations=2)
                connected = cv2.bitwise_or(connected, filled)
        
        # 将连续性信息融合到原图像
        enhanced_image = image.copy()
        continuity_mask = connected.astype(float) / 255.0
        
        for i in range(3):
            enhanced_image[:, :, i] = cv2.addWeighted(
                enhanced_image[:, :, i], 1 - continuity_weight,
                (enhanced_image[:, :, i] * continuity_mask).astype(np.uint8), continuity_weight, 0
            )
        
        return enhanced_image
    
    def _reduce_noise(self, image: np.ndarray, reduction_level: float) -> np.ndarray:
        """减少噪声"""
        logger.info(f"减少噪声，级别: {reduction_level}")
        
        # 双边滤波保持边缘的同时减少噪声
        kernel_size = int(5 + reduction_level * 10)
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        sigma_color = 75 * reduction_level
        sigma_space = 75 * reduction_level
        
        denoised = cv2.bilateralFilter(image, kernel_size, sigma_color, sigma_space)
        
        # 混合原图像和去噪图像
        alpha = 0.7
        result = cv2.addWeighted(image, alpha, denoised, 1 - alpha, 0)
        
        return result
    
    def _enhance_contrast(self, image: np.ndarray, enhancement_level: float) -> np.ndarray:
        """增强对比度"""
        logger.info(f"增强对比度，级别: {enhancement_level}")
        
        # CLAHE (Contrast Limited Adaptive Histogram Equalization)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        
        # 对L通道应用CLAHE
        clahe = cv2.createCLAHE(clipLimit=enhancement_level, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        
        # 转换回BGR
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        # 混合原图像和增强图像
        alpha = 0.8
        result = cv2.addWeighted(image, alpha, enhanced, 1 - alpha, 0)
        
        return result
    
    def _adjust_colors_for_realism(self, centers: np.ndarray) -> np.ndarray:
        """调整颜色以更接近真实刺绣"""
        # 增加饱和度
        adjusted_centers = centers.copy()
        
        for i in range(len(centers)):
            # 转换到HSV
            hsv = colorsys.rgb_to_hsv(centers[i][2]/255, centers[i][1]/255, centers[i][0]/255)
            
            # 增加饱和度
            hsv = (hsv[0], min(1.0, hsv[1] * 1.2), hsv[2])
            
            # 转换回RGB
            rgb = colorsys.hsv_to_rgb(hsv[0], hsv[1], hsv[2])
            
            # 更新中心点
            adjusted_centers[i] = [rgb[2]*255, rgb[1]*255, rgb[0]*255]
        
        return adjusted_centers
    
    def _adjust_colors_for_artistic(self, centers: np.ndarray) -> np.ndarray:
        """调整颜色以更艺术化"""
        # 增加对比度和饱和度
        adjusted_centers = centers.copy()
        
        for i in range(len(centers)):
            # 转换到HSV
            hsv = colorsys.rgb_to_hsv(centers[i][2]/255, centers[i][1]/255, centers[i][0]/255)
            
            # 增加饱和度和亮度
            hsv = (hsv[0], min(1.0, hsv[1] * 1.5), min(1.0, hsv[2] * 1.1))
            
            # 转换回RGB
            rgb = colorsys.hsv_to_rgb(hsv[0], hsv[1], hsv[2])
            
            # 更新中心点
            adjusted_centers[i] = [rgb[2]*255, rgb[1]*255, rgb[0]*255]
        
        return adjusted_centers
    
    def _calculate_quality_metrics(self, image: np.ndarray) -> Dict[str, float]:
        """计算质量指标"""
        # 简化的质量指标计算
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 颜色多样性
        unique_colors = len(np.unique(image.reshape(-1, 3), axis=0))
        color_diversity = min(1.0, unique_colors / 1000)
        
        # 边缘清晰度
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        edge_clarity = min(1.0, edge_density * 10)
        
        # 纹理复杂度
        gradient_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gradient_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
        texture_complexity = min(1.0, np.mean(gradient_magnitude) / 50)
        
        # 对比度
        contrast = np.std(gray) / 128
        
        return {
            'color_diversity': color_diversity,
            'edge_clarity': edge_clarity,
            'texture_complexity': texture_complexity,
            'contrast': contrast
        }
    
    def _calculate_improvements(self, 
                              original_metrics: Dict[str, float], 
                              optimized_metrics: Dict[str, float]) -> Dict[str, float]:
        """计算改进程度"""
        improvements = {}
        
        for key in original_metrics:
            if original_metrics[key] > 0:
                improvement = (optimized_metrics[key] - original_metrics[key]) / original_metrics[key]
                improvements[key] = improvement
            else:
                improvements[key] = 0.0
        
        return improvements
    
    def auto_optimize(self, image: np.ndarray, target_score: float = 0.8) -> OptimizationResult:
        """自动优化到目标质量"""
        logger.info(f"开始自动优化，目标评分: {target_score}")
        
        best_result = None
        best_score = 0.0
        
        # 参数搜索空间
        param_combinations = [
            OptimizationParams(16, 1.0, 0.6, 0.5, 0.2, 1.0),
            OptimizationParams(18, 1.2, 0.7, 0.6, 0.3, 1.1),
            OptimizationParams(20, 1.5, 0.8, 0.7, 0.4, 1.2),
            OptimizationParams(16, 1.8, 0.9, 0.8, 0.5, 1.3),
            OptimizationParams(18, 2.0, 1.0, 0.9, 0.6, 1.4),
        ]
        
        for params in param_combinations:
            try:
                result = self.optimize_embroidery(image, params)
                
                # 计算总体评分
                improvements = result.quality_improvements
                overall_score = sum(improvements.values()) / len(improvements)
                
                if overall_score > best_score:
                    best_score = overall_score
                    best_result = result
                
                if overall_score >= target_score:
                    logger.info(f"达到目标评分: {overall_score:.3f}")
                    break
                    
            except Exception as e:
                logger.warning(f"参数组合失败: {e}")
                continue
        
        if best_result is None:
            # 使用默认参数
            best_result = self.optimize_embroidery(image, OptimizationParams())
        
        return best_result
    
    def save_optimization_result(self, result: OptimizationResult, output_path: str):
        """保存优化结果"""
        # 保存优化后的图像
        cv2.imwrite(output_path, result.optimized_image)
        
        # 保存优化报告
        report_path = output_path.replace('.png', '_report.json')
        report = {
            'optimization_params': {
                'color_clusters': result.optimization_params.color_clusters,
                'edge_smoothness': result.optimization_params.edge_smoothness,
                'texture_detail_level': result.optimization_params.texture_detail_level,
                'pattern_continuity_weight': result.optimization_params.pattern_continuity_weight,
                'noise_reduction': result.optimization_params.noise_reduction,
                'contrast_enhancement': result.optimization_params.contrast_enhancement
            },
            'quality_improvements': result.quality_improvements,
            'processing_time': result.processing_time
        }
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"优化结果已保存: {output_path}")


if __name__ == "__main__":
    import time
    
    # 测试优化器
    optimizer = EmbroideryOptimizer()
    
    print("刺绣优化器测试")
    print("=" * 50)
    
    # 创建测试图像
    test_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    
    # 测试优化
    params = OptimizationParams(
        color_clusters=16,
        edge_smoothness=1.5,
        texture_detail_level=0.8,
        pattern_continuity_weight=0.6,
        noise_reduction=0.3,
        contrast_enhancement=1.2
    )
    
    result = optimizer.optimize_embroidery(test_image, params)
    
    print(f"优化完成")
    print(f"处理时间: {result.processing_time:.2f}秒")
    print(f"质量改进:")
    for metric, improvement in result.quality_improvements.items():
        print(f"  {metric}: {improvement:+.3f}")
    
    print("优化器测试完成") 