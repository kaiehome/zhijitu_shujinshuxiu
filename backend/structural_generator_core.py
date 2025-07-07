#!/usr/bin/env python3
"""
结构化专业识别图生成器 - 核心模块
基于"结构规则"为主轴，AI为辅助的技术路径
"""

import os
import cv2
import numpy as np
from PIL import Image, ImageDraw
from sklearn.cluster import KMeans
from scipy import ndimage
import logging
from typing import Tuple, List, Dict
import time
import json
from skimage.segmentation import slic
from skimage.util import img_as_float
from scipy.ndimage import label, find_objects
from collections import Counter

logger = logging.getLogger(__name__)

class StructuralCore:
    """结构化处理核心"""
    
    def __init__(self):
        self.config = {
            "min_region_size": 100,
            "color_separation_distance": 30,
            "saturation_boost": 1.3,
            "boundary_smoothness": 0.02,
            "contrast_enhancement": 1.5
        }
    
    def slic_superpixel_color_separation(self, image: np.ndarray, color_count: int, n_segments=450, compactness=12) -> Dict:
        """SLIC超像素+主色赋值分色"""
        try:
            segments = slic(img_as_float(image), n_segments=n_segments, compactness=compactness, start_label=0)
            color_regions = {}
            for label in np.unique(segments):
                mask = (segments == label)
                pixels = image[mask].reshape(-1, 3)
                if len(pixels) < 10:
                    color = np.mean(pixels, axis=0)
                else:
                    kmeans = KMeans(n_clusters=1, random_state=0).fit(pixels)
                    color = kmeans.cluster_centers_[0]
                color = color.astype(np.uint8)
                if np.sum(mask) > self.config["min_region_size"]:
                    color_regions[label] = {
                        'color': color,
                        'mask': mask,
                        'area': np.sum(mask),
                        'centroid': self._calculate_centroid(mask)
                    }
            logger.info(f"SLIC分割+主色赋值生成 {len(color_regions)} 个色块区域")
            return color_regions
        except Exception as e:
            raise Exception(f"SLIC超像素分色失败: {str(e)}")

    def structural_color_separation(self, image: np.ndarray, color_count: int, n_segments=450, compactness=12, detail_mask=None) -> dict:
        """
        SLIC超像素+主色赋值+小色块合并+边界平滑+细节增强
        :param image: 输入RGB图像
        :param color_count: 目标色彩数量
        :param n_segments: SLIC超像素数量
        :param compactness: SLIC紧致度
        :param detail_mask: 细节区域mask（可选）
        :return: 结构化分色结果dict
        """
        # 1. SLIC超像素分割
        segments = slic(img_as_float(image), n_segments=n_segments, compactness=compactness, start_label=0)
        out_img = np.zeros_like(image)
        label_map = np.zeros(segments.shape, dtype=np.int32)
        color_palette = []
        region_color = {}
        region_area = {}
        region_pixels = {}
        region_id = 1
        for label_val in np.unique(segments):
            mask = (segments == label_val)
            pixels = image[mask].reshape(-1, 3)
            if len(pixels) < 10:
                color = np.mean(pixels, axis=0)
            else:
                kmeans = KMeans(n_clusters=1, n_init=3, random_state=42)
                color = kmeans.fit(pixels).cluster_centers_[0]
            color = np.round(color).astype(np.uint8)
            color_tuple = tuple(color.tolist())
            color_palette.append(color_tuple)
            out_img[mask] = color
            label_map[mask] = region_id
            region_color[region_id] = color_tuple
            region_area[region_id] = np.sum(mask)
            region_pixels[region_id] = np.argwhere(mask)
            region_id += 1

        # 2. 小色块合并
        min_area = max(30, image.shape[0]*image.shape[1]//(n_segments*3))
        for rid, area in region_area.items():
            if area < min_area:
                coords = region_pixels[rid]
                for y, x in coords:
                    neighbors = label_map[max(0,y-1):y+2, max(0,x-1):x+2].flatten()
                    neighbor_ids = [nid for nid in neighbors if nid != rid and nid > 0]
                    if neighbor_ids:
                        target = Counter(neighbor_ids).most_common(1)[0][0]
                        label_map[y, x] = target
                        out_img[y, x] = region_color[target]

        # 3. 边界平滑
        out_img = cv2.bilateralFilter(out_img, d=7, sigmaColor=50, sigmaSpace=7)

        # 4. 细节区域增强（可选）
        if detail_mask is not None:
            detail_segments = slic(img_as_float(image), n_segments=n_segments*2, compactness=compactness/2, mask=detail_mask, start_label=0)
            for label_val in np.unique(detail_segments):
                mask = (detail_segments == label_val) & (detail_mask > 0)
                if np.sum(mask) < 5:
                    continue
                pixels = image[mask].reshape(-1, 3)
                kmeans = KMeans(n_clusters=1, n_init=3, random_state=42)
                color = kmeans.fit(pixels).cluster_centers_[0]
                color = np.round(color).astype(np.uint8)
                out_img[mask] = color

        # 5. 色彩数量控制
        all_pixels = out_img.reshape(-1, 3)
        kmeans_final = KMeans(n_clusters=color_count, n_init=5, random_state=42)
        labels = kmeans_final.fit_predict(all_pixels)
        palette = np.round(kmeans_final.cluster_centers_).astype(np.uint8)
        out_img = palette[labels].reshape(image.shape)

        # 6. 返回结构化结果
        return {
            'structural_image': out_img,
            'palette': palette.tolist(),
            'label_map': label_map,
        }
    
    def optimize_region_connectivity(self, color_regions: Dict) -> Dict:
        """优化区域连通性，确保每个色块为闭合区域"""
        try:
            optimized_regions = {}
            
            for region_id, region_data in color_regions.items():
                mask = region_data['mask']
                color = region_data['color']
                
                # 形态学闭运算：连接断开的区域
                kernel = np.ones((7, 7), np.uint8)
                closed_mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
                
                # 填充孔洞：确保区域完整性
                filled_mask = ndimage.binary_fill_holes(closed_mask).astype(np.uint8)
                
                # 连通组件分析
                num_labels, labels = cv2.connectedComponents(filled_mask)
                
                # 保留面积足够大的组件
                for label in range(1, num_labels):
                    component_mask = (labels == label)
                    component_area = np.sum(component_mask)
                    
                    if component_area >= self.config["min_region_size"]:
                        new_region_id = f"{region_id}_{label}"
                        optimized_regions[new_region_id] = {
                            'color': color,
                            'mask': component_mask,
                            'area': component_area,
                            'centroid': self._calculate_centroid(component_mask),
                            'is_closed': True
                        }
            
            logger.info(f"区域连通性优化完成，生成 {len(optimized_regions)} 个闭合区域")
            return optimized_regions
            
        except Exception as e:
            raise Exception(f"区域连通性优化失败: {str(e)}")
    
    def extract_vector_boundaries(self, regions: Dict) -> Dict:
        """提取矢量边界和路径信息"""
        try:
            vector_boundaries = {}
            
            for region_id, region_data in regions.items():
                mask = region_data['mask'].astype(np.uint8)
                
                # 边界提取
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                if not contours:
                    continue
                
                # 选择最大轮廓
                main_contour = max(contours, key=cv2.contourArea)
                
                # 轮廓简化
                epsilon = self.config["boundary_smoothness"] * cv2.arcLength(main_contour, True)
                simplified_contour = cv2.approxPolyDP(main_contour, epsilon, True)
                
                # 生成路径信息
                path_points = simplified_contour.reshape(-1, 2).tolist()
                
                vector_boundaries[region_id] = {
                    'contour': main_contour,
                    'simplified_contour': simplified_contour,
                    'path_points': path_points,
                    'area': cv2.contourArea(main_contour),
                    'perimeter': cv2.arcLength(main_contour, True),
                    'color': region_data['color'],
                    'is_closed': True
                }
            
            logger.info(f"矢量边界提取完成，生成 {len(vector_boundaries)} 个路径")
            return vector_boundaries
            
        except Exception as e:
            raise Exception(f"矢量边界提取失败: {str(e)}")
    
    def generate_final_image(self, regions: Dict, boundaries: Dict) -> np.ndarray:
        """生成最终的专业识别图像"""
        try:
            # 获取图像尺寸
            first_region = next(iter(regions.values()))
            height, width = first_region['mask'].shape
            
            # 创建空白图像
            professional_image = np.zeros((height, width, 3), dtype=np.uint8)
            
            # 按区域面积排序，先绘制大区域
            sorted_regions = sorted(
                regions.items(),
                key=lambda x: x[1]['area'],
                reverse=True
            )
            
            for region_id, region_data in sorted_regions:
                mask = region_data['mask']
                color = region_data['color']
                
                # 填充区域
                professional_image[mask] = color
            
            # 边界增强
            professional_image = self._enhance_boundaries(professional_image, boundaries)
            
            # 最终质量增强
            professional_image = self._final_enhancement(professional_image)
            
            return professional_image
            
        except Exception as e:
            raise Exception(f"最终专业图像生成失败: {str(e)}")
    
    def _optimize_colors(self, colors: np.ndarray) -> np.ndarray:
        """优化颜色调色板"""
        try:
            optimized_colors = []
            
            for color in colors:
                # HSV优化
                hsv = cv2.cvtColor(color.reshape(1, 1, 3), cv2.COLOR_RGB2HSV)[0, 0]
                
                # 饱和度增强
                hsv[1] = min(255, max(100, int(hsv[1] * self.config["saturation_boost"])))
                
                # 亮度调整
                if hsv[2] < 50:
                    hsv[2] = 50
                elif hsv[2] > 230:
                    hsv[2] = 230
                
                # 色相量化
                hsv[0] = (hsv[0] // 15) * 15
                
                # 转换回RGB
                rgb = cv2.cvtColor(hsv.reshape(1, 1, 3), cv2.COLOR_HSV2RGB)[0, 0]
                optimized_colors.append(rgb)
            
            return np.array(optimized_colors, dtype=np.uint8)
            
        except Exception as e:
            logger.warning(f"颜色优化失败: {str(e)}")
            return colors
    
    def _calculate_centroid(self, mask: np.ndarray) -> Tuple[int, int]:
        """计算区域质心"""
        try:
            moments = cv2.moments(mask.astype(np.uint8))
            if moments['m00'] != 0:
                cx = int(moments['m10'] / moments['m00'])
                cy = int(moments['m01'] / moments['m00'])
                return (cx, cy)
            else:
                return (0, 0)
        except:
            return (0, 0)
    
    def _enhance_boundaries(self, image: np.ndarray, boundaries: Dict) -> np.ndarray:
        """增强边界清晰度"""
        try:
            enhanced_image = image.copy()
            
            for boundary_data in boundaries.values():
                contour = boundary_data['contour']
                color = boundary_data['color']
                
                # 绘制边界线
                cv2.drawContours(enhanced_image, [contour], -1, color.tolist(), thickness=2)
            
            return enhanced_image
            
        except Exception as e:
            logger.warning(f"边界增强失败: {str(e)}")
            return image
    
    def _final_enhancement(self, image: np.ndarray) -> np.ndarray:
        """最终质量增强"""
        try:
            # 对比度增强
            enhanced = cv2.convertScaleAbs(image, alpha=self.config["contrast_enhancement"], beta=0)
            
            # 轻度锐化
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            sharpened = cv2.filter2D(enhanced, -1, kernel)
            
            # 混合
            final_image = cv2.addWeighted(enhanced, 0.7, sharpened, 0.3, 0)
            
            return final_image
            
        except Exception as e:
            logger.warning(f"最终质量增强失败: {str(e)}")
            return image 