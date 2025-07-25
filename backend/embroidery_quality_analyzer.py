"""
刺绣质量分析器
分析生成图像与真实刺绣的差异，提供详细的优化建议
"""

import cv2
import numpy as np
import logging
from typing import Dict, Any, Tuple, List, Optional
from dataclasses import dataclass
import matplotlib.pyplot as plt
from skimage import measure, filters, morphology
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import json
import os

logger = logging.getLogger(__name__)


@dataclass
class QualityMetrics:
    """质量指标"""
    color_accuracy: float  # 颜色准确度
    edge_quality: float    # 边缘质量
    texture_detail: float  # 纹理细节
    pattern_continuity: float  # 图案连续性
    overall_score: float   # 总体评分
    recommendations: List[str]  # 优化建议


@dataclass
class EmbroideryAnalysis:
    """刺绣分析结果"""
    generated_image: np.ndarray
    reference_image: np.ndarray
    quality_metrics: QualityMetrics
    color_palette_diff: Dict[str, Any]
    edge_analysis: Dict[str, Any]
    texture_analysis: Dict[str, Any]
    pattern_analysis: Dict[str, Any]


class EmbroideryQualityAnalyzer:
    """刺绣质量分析器"""
    
    def __init__(self):
        self.analysis_results = []
        
    def analyze_embroidery_quality(self, 
                                 generated_path: str, 
                                 reference_path: str) -> EmbroideryAnalysis:
        """分析刺绣质量"""
        logger.info("开始刺绣质量分析...")
        
        # 加载图像
        generated_img = cv2.imread(generated_path)
        reference_img = cv2.imread(reference_path)
        
        if generated_img is None or reference_img is None:
            raise ValueError("无法加载图像文件")
        
        # 统一图像尺寸
        generated_img, reference_img = self._normalize_images(generated_img, reference_img)
        
        # 执行各项分析
        color_analysis = self._analyze_color_accuracy(generated_img, reference_img)
        edge_analysis = self._analyze_edge_quality(generated_img, reference_img)
        texture_analysis = self._analyze_texture_detail(generated_img, reference_img)
        pattern_analysis = self._analyze_pattern_continuity(generated_img, reference_img)
        
        # 计算总体评分
        overall_score = self._calculate_overall_score(color_analysis, edge_analysis, 
                                                    texture_analysis, pattern_analysis)
        
        # 生成优化建议
        recommendations = self._generate_recommendations(color_analysis, edge_analysis,
                                                       texture_analysis, pattern_analysis)
        
        # 创建质量指标
        quality_metrics = QualityMetrics(
            color_accuracy=color_analysis['score'],
            edge_quality=edge_analysis['score'],
            texture_detail=texture_analysis['score'],
            pattern_continuity=pattern_analysis['score'],
            overall_score=overall_score,
            recommendations=recommendations
        )
        
        # 创建分析结果
        analysis = EmbroideryAnalysis(
            generated_image=generated_img,
            reference_image=reference_img,
            quality_metrics=quality_metrics,
            color_palette_diff=color_analysis,
            edge_analysis=edge_analysis,
            texture_analysis=texture_analysis,
            pattern_analysis=pattern_analysis
        )
        
        self.analysis_results.append(analysis)
        logger.info(f"分析完成，总体评分: {overall_score:.2f}")
        
        return analysis
    
    def _normalize_images(self, img1: np.ndarray, img2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """统一图像尺寸和格式"""
        # 统一尺寸
        target_size = (512, 512)
        img1_resized = cv2.resize(img1, target_size)
        img2_resized = cv2.resize(img2, target_size)
        
        return img1_resized, img2_resized
    
    def _analyze_color_accuracy(self, generated: np.ndarray, reference: np.ndarray) -> Dict[str, Any]:
        """分析颜色准确度"""
        logger.info("分析颜色准确度...")
        
        # 转换到LAB色彩空间
        generated_lab = cv2.cvtColor(generated, cv2.COLOR_BGR2LAB)
        reference_lab = cv2.cvtColor(reference, cv2.COLOR_BGR2LAB)
        
        # 计算颜色直方图
        generated_hist = cv2.calcHist([generated_lab], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        reference_hist = cv2.calcHist([reference_lab], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        
        # 归一化直方图
        generated_hist = cv2.normalize(generated_hist, generated_hist).flatten()
        reference_hist = cv2.normalize(reference_hist, reference_hist).flatten()
        
        # 计算直方图相似度
        hist_similarity = cv2.compareHist(generated_hist, reference_hist, cv2.HISTCMP_CORREL)
        
        # 计算平均颜色差异
        color_diff = np.mean(np.abs(generated_lab.astype(float) - reference_lab.astype(float)))
        max_diff = 255.0
        color_accuracy = 1.0 - (color_diff / max_diff)
        
        # 计算颜色数量
        generated_colors = len(np.unique(generated.reshape(-1, 3), axis=0))
        reference_colors = len(np.unique(reference.reshape(-1, 3), axis=0))
        
        # 综合评分
        score = (hist_similarity + color_accuracy) / 2.0
        
        return {
            'score': score,
            'histogram_similarity': hist_similarity,
            'color_accuracy': color_accuracy,
            'color_diff': color_diff,
            'generated_colors': generated_colors,
            'reference_colors': reference_colors,
            'color_efficiency': generated_colors / reference_colors if reference_colors > 0 else 0
        }
    
    def _analyze_edge_quality(self, generated: np.ndarray, reference: np.ndarray) -> Dict[str, Any]:
        """分析边缘质量"""
        logger.info("分析边缘质量...")
        
        # 转换为灰度图
        generated_gray = cv2.cvtColor(generated, cv2.COLOR_BGR2GRAY)
        reference_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
        
        # 边缘检测
        generated_edges = cv2.Canny(generated_gray, 50, 150)
        reference_edges = cv2.Canny(reference_gray, 50, 150)
        
        # 计算边缘密度
        generated_edge_density = np.sum(generated_edges > 0) / generated_edges.size
        reference_edge_density = np.sum(reference_edges > 0) / reference_edges.size
        
        # 计算边缘连续性
        generated_continuity = self._calculate_edge_continuity(generated_edges)
        reference_continuity = self._calculate_edge_continuity(reference_edges)
        
        # 计算边缘平滑度
        generated_smoothness = self._calculate_edge_smoothness(generated_edges)
        reference_smoothness = self._calculate_edge_smoothness(reference_edges)
        
        # 综合评分
        density_score = 1.0 - abs(generated_edge_density - reference_edge_density)
        continuity_score = generated_continuity / reference_continuity if reference_continuity > 0 else 0
        smoothness_score = generated_smoothness / reference_smoothness if reference_smoothness > 0 else 0
        
        score = (density_score + continuity_score + smoothness_score) / 3.0
        
        return {
            'score': score,
            'edge_density': generated_edge_density,
            'reference_edge_density': reference_edge_density,
            'edge_continuity': generated_continuity,
            'reference_continuity': reference_continuity,
            'edge_smoothness': generated_smoothness,
            'reference_smoothness': reference_smoothness,
            'density_score': density_score,
            'continuity_score': continuity_score,
            'smoothness_score': smoothness_score
        }
    
    def _analyze_texture_detail(self, generated: np.ndarray, reference: np.ndarray) -> Dict[str, Any]:
        """分析纹理细节"""
        logger.info("分析纹理细节...")
        
        # 转换为灰度图
        generated_gray = cv2.cvtColor(generated, cv2.COLOR_BGR2GRAY)
        reference_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
        
        # 计算局部二值模式 (LBP)
        generated_lbp = self._calculate_lbp(generated_gray)
        reference_lbp = self._calculate_lbp(reference_gray)
        
        # 计算纹理复杂度
        generated_complexity = self._calculate_texture_complexity(generated_gray)
        reference_complexity = self._calculate_texture_complexity(reference_gray)
        
        # 计算梯度幅值
        generated_gradient = self._calculate_gradient_magnitude(generated_gray)
        reference_gradient = self._calculate_gradient_magnitude(reference_gray)
        
        # 计算细节保留度
        detail_retention = generated_gradient / reference_gradient if reference_gradient > 0 else 0
        
        # 综合评分
        complexity_score = generated_complexity / reference_complexity if reference_complexity > 0 else 0
        detail_score = min(detail_retention, 1.0)
        
        score = (complexity_score + detail_score) / 2.0
        
        return {
            'score': score,
            'texture_complexity': generated_complexity,
            'reference_complexity': reference_complexity,
            'gradient_magnitude': generated_gradient,
            'reference_gradient': reference_gradient,
            'detail_retention': detail_retention,
            'complexity_score': complexity_score,
            'detail_score': detail_score
        }
    
    def _analyze_pattern_continuity(self, generated: np.ndarray, reference: np.ndarray) -> Dict[str, Any]:
        """分析图案连续性"""
        logger.info("分析图案连续性...")
        
        # 转换为灰度图
        generated_gray = cv2.cvtColor(generated, cv2.COLOR_BGR2GRAY)
        reference_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
        
        # 计算连通区域
        generated_labels = measure.label(generated_gray > 128)
        reference_labels = measure.label(reference_gray > 128)
        
        # 计算区域数量
        generated_regions = len(np.unique(generated_labels)) - 1
        reference_regions = len(np.unique(reference_labels)) - 1
        
        # 计算区域大小分布
        generated_props = measure.regionprops(generated_labels)
        reference_props = measure.regionprops(reference_labels)
        
        generated_areas = [prop.area for prop in generated_props]
        reference_areas = [prop.area for prop in reference_props]
        
        # 计算连续性指标
        generated_continuity = self._calculate_pattern_continuity(generated_gray)
        reference_continuity = self._calculate_pattern_continuity(reference_gray)
        
        # 计算区域分布相似度
        area_similarity = self._calculate_area_distribution_similarity(generated_areas, reference_areas)
        
        # 综合评分
        region_score = 1.0 - abs(generated_regions - reference_regions) / max(reference_regions, 1)
        continuity_score = generated_continuity / reference_continuity if reference_continuity > 0 else 0
        
        score = (region_score + continuity_score + area_similarity) / 3.0
        
        return {
            'score': score,
            'generated_regions': generated_regions,
            'reference_regions': reference_regions,
            'pattern_continuity': generated_continuity,
            'reference_continuity': reference_continuity,
            'area_similarity': area_similarity,
            'region_score': region_score,
            'continuity_score': continuity_score
        }
    
    def _calculate_overall_score(self, color_analysis: Dict, edge_analysis: Dict, 
                               texture_analysis: Dict, pattern_analysis: Dict) -> float:
        """计算总体评分"""
        weights = {
            'color': 0.3,
            'edge': 0.25,
            'texture': 0.25,
            'pattern': 0.2
        }
        
        overall_score = (
            color_analysis['score'] * weights['color'] +
            edge_analysis['score'] * weights['edge'] +
            texture_analysis['score'] * weights['texture'] +
            pattern_analysis['score'] * weights['pattern']
        )
        
        return overall_score
    
    def _generate_recommendations(self, color_analysis: Dict, edge_analysis: Dict,
                                texture_analysis: Dict, pattern_analysis: Dict) -> List[str]:
        """生成优化建议"""
        recommendations = []
        
        # 颜色优化建议
        if color_analysis['score'] < 0.7:
            if color_analysis['color_efficiency'] < 0.8:
                recommendations.append("增加颜色数量以提高细节表现")
            if color_analysis['color_accuracy'] < 0.6:
                recommendations.append("优化颜色量化算法以提高颜色准确度")
        
        # 边缘优化建议
        if edge_analysis['score'] < 0.7:
            if edge_analysis['edge_smoothness'] < edge_analysis['reference_smoothness'] * 0.8:
                recommendations.append("改善边缘平滑度处理")
            if edge_analysis['edge_continuity'] < edge_analysis['reference_continuity'] * 0.8:
                recommendations.append("增强边缘连续性检测")
        
        # 纹理优化建议
        if texture_analysis['score'] < 0.7:
            if texture_analysis['detail_retention'] < 0.8:
                recommendations.append("增强纹理细节保留")
            if texture_analysis['complexity_score'] < 0.8:
                recommendations.append("提高纹理复杂度")
        
        # 图案优化建议
        if pattern_analysis['score'] < 0.7:
            if pattern_analysis['continuity_score'] < 0.8:
                recommendations.append("改善图案连续性")
            if pattern_analysis['area_similarity'] < 0.8:
                recommendations.append("优化区域分布")
        
        if not recommendations:
            recommendations.append("图像质量良好，可考虑微调参数")
        
        return recommendations
    
    def _calculate_edge_continuity(self, edges: np.ndarray) -> float:
        """计算边缘连续性"""
        # 使用形态学操作计算连通性
        kernel = np.ones((3, 3), np.uint8)
        closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        return np.sum(closed_edges > 0) / np.sum(edges > 0) if np.sum(edges > 0) > 0 else 0
    
    def _calculate_edge_smoothness(self, edges: np.ndarray) -> float:
        """计算边缘平滑度"""
        # 计算边缘的梯度变化
        gradient_x = cv2.Sobel(edges, cv2.CV_64F, 1, 0, ksize=3)
        gradient_y = cv2.Sobel(edges, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
        return np.mean(gradient_magnitude)
    
    def _calculate_lbp(self, image: np.ndarray) -> np.ndarray:
        """计算局部二值模式"""
        # 简化的LBP实现
        lbp = np.zeros_like(image)
        for i in range(1, image.shape[0]-1):
            for j in range(1, image.shape[1]-1):
                center = image[i, j]
                code = 0
                code |= (image[i-1, j-1] > center) << 7
                code |= (image[i-1, j] > center) << 6
                code |= (image[i-1, j+1] > center) << 5
                code |= (image[i, j+1] > center) << 4
                code |= (image[i+1, j+1] > center) << 3
                code |= (image[i+1, j] > center) << 2
                code |= (image[i+1, j-1] > center) << 1
                code |= (image[i, j-1] > center) << 0
                lbp[i, j] = code
        return lbp
    
    def _calculate_texture_complexity(self, image: np.ndarray) -> float:
        """计算纹理复杂度"""
        # 使用标准差作为复杂度指标
        return np.std(image)
    
    def _calculate_gradient_magnitude(self, image: np.ndarray) -> float:
        """计算梯度幅值"""
        gradient_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        gradient_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
        return np.mean(gradient_magnitude)
    
    def _calculate_pattern_continuity(self, image: np.ndarray) -> float:
        """计算图案连续性"""
        # 计算水平方向的连续性
        horizontal_continuity = np.mean(np.abs(np.diff(image, axis=1)))
        # 计算垂直方向的连续性
        vertical_continuity = np.mean(np.abs(np.diff(image, axis=0)))
        return 1.0 / (1.0 + (horizontal_continuity + vertical_continuity) / 2.0)
    
    def _calculate_area_distribution_similarity(self, areas1: List[int], areas2: List[int]) -> float:
        """计算区域分布相似度"""
        if not areas1 or not areas2:
            return 0.0
        
        # 计算面积分布的统计特征
        mean1, std1 = np.mean(areas1), np.std(areas1)
        mean2, std2 = np.mean(areas2), np.std(areas2)
        
        # 计算相似度
        mean_similarity = 1.0 / (1.0 + abs(mean1 - mean2) / max(mean1, mean2, 1))
        std_similarity = 1.0 / (1.0 + abs(std1 - std2) / max(std1, std2, 1))
        
        return (mean_similarity + std_similarity) / 2.0
    
    def generate_analysis_report(self, analysis: EmbroideryAnalysis, output_path: str = None):
        """生成分析报告"""
        report = {
            'overall_score': analysis.quality_metrics.overall_score,
            'detailed_metrics': {
                'color_accuracy': analysis.quality_metrics.color_accuracy,
                'edge_quality': analysis.quality_metrics.edge_quality,
                'texture_detail': analysis.quality_metrics.texture_detail,
                'pattern_continuity': analysis.quality_metrics.pattern_continuity
            },
            'recommendations': analysis.quality_metrics.recommendations,
            'color_analysis': analysis.color_palette_diff,
            'edge_analysis': analysis.edge_analysis,
            'texture_analysis': analysis.texture_analysis,
            'pattern_analysis': analysis.pattern_analysis
        }
        
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
        
        return report
    
    def visualize_analysis(self, analysis: EmbroideryAnalysis, output_path: str = None):
        """可视化分析结果"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # 原始图像对比
        axes[0, 0].imshow(cv2.cvtColor(analysis.generated_image, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title('生成图像')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(cv2.cvtColor(analysis.reference_image, cv2.COLOR_BGR2RGB))
        axes[0, 1].set_title('参考图像')
        axes[0, 1].axis('off')
        
        # 质量指标雷达图
        metrics = ['颜色准确度', '边缘质量', '纹理细节', '图案连续性']
        scores = [
            analysis.quality_metrics.color_accuracy,
            analysis.quality_metrics.edge_quality,
            analysis.quality_metrics.texture_detail,
            analysis.quality_metrics.pattern_continuity
        ]
        
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        scores += scores[:1]  # 闭合图形
        angles += angles[:1]
        
        axes[0, 2].plot(angles, scores, 'o-', linewidth=2)
        axes[0, 2].fill(angles, scores, alpha=0.25)
        axes[0, 2].set_xticks(angles[:-1])
        axes[0, 2].set_xticklabels(metrics)
        axes[0, 2].set_ylim(0, 1)
        axes[0, 2].set_title('质量指标雷达图')
        axes[0, 2].grid(True)
        
        # 详细分析
        axes[1, 0].text(0.1, 0.9, f'总体评分: {analysis.quality_metrics.overall_score:.3f}', 
                       transform=axes[1, 0].transAxes, fontsize=12, fontweight='bold')
        axes[1, 0].text(0.1, 0.8, f'颜色准确度: {analysis.quality_metrics.color_accuracy:.3f}', 
                       transform=axes[1, 0].transAxes, fontsize=10)
        axes[1, 0].text(0.1, 0.7, f'边缘质量: {analysis.quality_metrics.edge_quality:.3f}', 
                       transform=axes[1, 0].transAxes, fontsize=10)
        axes[1, 0].text(0.1, 0.6, f'纹理细节: {analysis.quality_metrics.texture_detail:.3f}', 
                       transform=axes[1, 0].transAxes, fontsize=10)
        axes[1, 0].text(0.1, 0.5, f'图案连续性: {analysis.quality_metrics.pattern_continuity:.3f}', 
                       transform=axes[1, 0].transAxes, fontsize=10)
        axes[1, 0].set_title('详细评分')
        axes[1, 0].axis('off')
        
        # 优化建议
        recommendations_text = '\n'.join(analysis.quality_metrics.recommendations)
        axes[1, 1].text(0.1, 0.9, '优化建议:', transform=axes[1, 1].transAxes, 
                       fontsize=12, fontweight='bold')
        axes[1, 1].text(0.1, 0.8, recommendations_text, transform=axes[1, 1].transAxes, 
                       fontsize=10, verticalalignment='top')
        axes[1, 1].set_title('优化建议')
        axes[1, 1].axis('off')
        
        # 颜色分析
        color_data = analysis.color_palette_diff
        axes[1, 2].bar(['生成颜色数', '参考颜色数'], 
                      [color_data['generated_colors'], color_data['reference_colors']])
        axes[1, 2].set_title('颜色数量对比')
        axes[1, 2].set_ylabel('颜色数量')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
        
        plt.show()


if __name__ == "__main__":
    # 测试分析器
    analyzer = EmbroideryQualityAnalyzer()
    
    # 示例用法
    print("刺绣质量分析器测试")
    print("=" * 50)
    
    # 这里需要实际的图像文件路径
    # analysis = analyzer.analyze_embroidery_quality("generated.png", "reference.png")
    # report = analyzer.generate_analysis_report(analysis, "quality_report.json")
    # analyzer.visualize_analysis(analysis, "quality_analysis.png")
    
    print("分析器初始化完成，等待图像文件进行分析...") 