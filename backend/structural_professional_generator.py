#!/usr/bin/env python3
"""
结构化专业识别图生成器
基于"结构规则"为主轴，AI为辅助的技术路径
目标：生成具备机器可读结构特征的专业识别图像

核心原则：
1. 区域连续性：每个色块为闭合区域，可用于填充
2. 边界清晰性：路径可提取、点数可控  
3. 颜色可控性：控制颜色总数、避免相邻干扰、符合绣线规范
"""

import os
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageDraw, ImageFont
from sklearn.cluster import KMeans
from scipy import ndimage
from scipy.spatial.distance import cdist
import logging
from typing import Tuple, List, Dict, Optional
import time
import json
from structural_generator_core import StructuralCore

logger = logging.getLogger(__name__)

class StructuralProfessionalGenerator:
    """结构化专业识别图生成器"""
    
    def __init__(self):
        """初始化结构化生成器"""
        self.core = StructuralCore()
        logger.info("🔧 结构化专业识别图生成器已初始化")
    
    def generate_structural_professional_image(self, 
                                             input_path: str, 
                                             job_id: str,
                                             color_count: int = 12) -> Tuple[str, str, str, float]:
        """
        生成结构化专业识别图像
        
        Returns:
            Tuple[professional_path, comparison_path, structure_info_path, processing_time]
        """
        start_time = time.time()
        
        try:
            logger.info(f"🎯 开始生成结构化专业识别图像: {job_id}")
            
            # 创建输出目录
            output_dir = f"outputs/{job_id}"
            os.makedirs(output_dir, exist_ok=True)
            
            # 1. 加载和预分析图像结构
            original_image = self._load_and_analyze_structure(input_path)
            logger.info("✓ 图像结构分析完成")
            
            # 2. 结构化颜色分离（硬约束）
            color_regions = self.core.structural_color_separation(original_image, color_count)
            logger.info("✓ 结构化颜色分离完成")
            
            # 3. 区域连通性分析和优化
            connected_regions = self.core.optimize_region_connectivity(color_regions)
            logger.info("✓ 区域连通性优化完成")
            
            # 4. 边界矢量化和路径提取
            vector_boundaries = self.core.extract_vector_boundaries(connected_regions)
            logger.info("✓ 边界矢量化完成")
            
            # 5. 生成最终专业识别图
            professional_image = self.core.generate_final_image(connected_regions, vector_boundaries)
            logger.info("✓ 专业识别图生成完成")
            
            # 6. 创建结构信息和对比图
            structure_info = self._generate_structure_info(vector_boundaries, connected_regions)
            comparison_image = self._create_structural_comparison(original_image, professional_image, structure_info)
            
            # 7. 保存结果
            professional_path = os.path.join(output_dir, f"{job_id}_structural_professional.png")
            comparison_path = os.path.join(output_dir, f"{job_id}_structural_comparison.png")
            structure_info_path = os.path.join(output_dir, f"{job_id}_structure_info.json")
            
            self._save_image(professional_image, professional_path)
            self._save_image(comparison_image, comparison_path)
            self._save_structure_info(structure_info, structure_info_path)
            
            processing_time = time.time() - start_time
            logger.info(f"🎉 结构化专业识别图生成完成，耗时: {processing_time:.2f}秒")
            
            return professional_path, comparison_path, structure_info_path, processing_time
            
        except Exception as e:
            error_msg = f"结构化专业识别图生成失败: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise Exception(error_msg)
    
    def _load_and_analyze_structure(self, input_path: str) -> np.ndarray:
        """加载图像并进行结构预分析"""
        try:
            # 加载图像
            if not os.path.isabs(input_path):
                input_path = os.path.abspath(input_path)
            
            if not os.path.exists(input_path):
                raise ValueError(f"图像文件不存在: {input_path}")
            
            # 使用PIL加载以确保颜色空间正确
            with Image.open(input_path) as pil_image:
                if pil_image.mode != 'RGB':
                    pil_image = pil_image.convert('RGB')
                image = np.array(pil_image)
            
            # 结构预分析
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / edges.size
            unique_colors = len(np.unique(image.reshape(-1, 3), axis=0))
            
            logger.info(f"图像结构指标 - 边缘密度: {edge_density:.3f}, 颜色数: {unique_colors}")
            
            return image
            
        except Exception as e:
            raise Exception(f"图像加载和结构分析失败: {str(e)}")
    
    def _generate_structure_info(self, boundaries: Dict, regions: Dict) -> Dict:
        """生成结构信息"""
        try:
            structure_info = {
                'metadata': {
                    'generator': 'StructuralProfessionalGenerator',
                    'version': '1.0',
                    'timestamp': time.time(),
                    'total_regions': len(regions),
                    'total_boundaries': len(boundaries)
                },
                'regions': {},
                'boundaries': {},
                'color_palette': [],
                'structure_metrics': {}
            }
            
            # 收集颜色调色板
            colors = []
            for region_data in regions.values():
                color = region_data['color'].tolist()
                if color not in colors:
                    colors.append(color)
            structure_info['color_palette'] = colors
            
            # 区域信息
            for region_id, region_data in regions.items():
                structure_info['regions'][region_id] = {
                    'color': region_data['color'].tolist(),
                    'area': int(region_data['area']),
                    'centroid': region_data['centroid'],
                    'is_closed': region_data.get('is_closed', False)
                }
            
            # 边界信息
            for boundary_id, boundary_data in boundaries.items():
                structure_info['boundaries'][boundary_id] = {
                    'path_points': boundary_data['path_points'],
                    'area': float(boundary_data['area']),
                    'perimeter': float(boundary_data['perimeter']),
                    'color': boundary_data['color'].tolist(),
                    'point_count': len(boundary_data['path_points'])
                }
            
            return structure_info
            
        except Exception as e:
            logger.warning(f"结构信息生成失败: {str(e)}")
            return {'error': str(e)}
    
    def _create_structural_comparison(self, original: np.ndarray, professional: np.ndarray, structure_info: Dict) -> np.ndarray:
        """创建结构化对比图"""
        try:
            # 调整图像尺寸
            height, width = original.shape[:2]
            
            # 创建对比画布
            comparison_width = width * 2 + 40
            comparison_height = height + 100
            comparison = np.ones((comparison_height, comparison_width, 3), dtype=np.uint8) * 240
            
            # 放置原图和处理图
            comparison[50:50+height, 20:20+width] = original
            comparison[50:50+height, 40+width:40+2*width] = professional
            
            # 添加标题和信息
            pil_comparison = Image.fromarray(comparison)
            draw = ImageDraw.Draw(pil_comparison)
            
            # 标题
            draw.text((20, 10), "原图", fill=(0, 0, 0))
            draw.text((40+width, 10), f"结构化专业识别图 ({len(structure_info.get('regions', {}))}区域)", fill=(0, 0, 0))
            
            # 底部信息
            info_text = f"颜色数: {len(structure_info.get('color_palette', []))} | 区域数: {len(structure_info.get('regions', {}))}"
            draw.text((20, height + 60), info_text, fill=(0, 0, 0))
            
            return np.array(pil_comparison)
            
        except Exception as e:
            logger.warning(f"结构化对比图创建失败: {str(e)}")
            # 简单对比图
            return np.hstack([original, professional])
    
    def _save_image(self, image: np.ndarray, output_path: str):
        """保存图像"""
        try:
            pil_image = Image.fromarray(image)
            pil_image.save(output_path, 'PNG', quality=100)
            logger.info(f"图像已保存: {output_path}")
        except Exception as e:
            logger.error(f"图像保存失败: {str(e)}")
    
    def _save_structure_info(self, structure_info: Dict, output_path: str):
        """保存结构信息"""
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(structure_info, f, indent=2, ensure_ascii=False)
            logger.info(f"结构信息已保存: {output_path}")
        except Exception as e:
            logger.error(f"结构信息保存失败: {str(e)}")


# 主函数用于测试
def main():
    """测试函数"""
    try:
        generator = StructuralProfessionalGenerator()
        
        # 测试图像路径
        test_image = "uploads/test_image.jpg"
        job_id = "test_structural"
        
        if os.path.exists(test_image):
            professional_path, comparison_path, structure_info_path, processing_time = generator.generate_structural_professional_image(
                test_image, job_id, color_count=12
            )
            
            print(f"✅ 结构化专业识别图生成成功！")
            print(f"📁 专业图像: {professional_path}")
            print(f"📊 对比图像: {comparison_path}")
            print(f"📋 结构信息: {structure_info_path}")
            print(f"⏱️  处理时间: {processing_time:.2f}秒")
        else:
            print(f"❌ 测试图像不存在: {test_image}")
            
    except Exception as e:
        print(f"❌ 测试失败: {str(e)}")


if __name__ == "__main__":
    main() 