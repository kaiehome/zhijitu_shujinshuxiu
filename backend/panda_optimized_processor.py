# panda_optimized_processor.py
# 专门针对熊猫图像的优化处理器

import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional
from skimage.segmentation import slic
from skimage.color import label2rgb
import json

class PandaOptimizedProcessor:
    def __init__(self):
        self.panda_colors = {
            'black': [0, 0, 0],
            'white': [255, 255, 255],
            'gray': [128, 128, 128],
            'light_gray': [192, 192, 192]
        }
        
    def detect_panda_features(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """检测熊猫特征区域"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 检测黑色区域（熊猫特征）
        _, black_mask = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
        
        # 检测白色区域
        _, white_mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        
        # 检测眼睛区域（圆形检测）
        circles = cv2.HoughCircles(
            gray, cv2.HOUGH_GRADIENT, 1, 20,
            param1=50, param2=30, minRadius=5, maxRadius=30
        )
        
        eye_mask = np.zeros_like(gray)
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                cv2.circle(eye_mask, (i[0], i[1]), i[2], 255, -1)
        
        return {
            'black_mask': black_mask,
            'white_mask': white_mask,
            'eye_mask': eye_mask,
            'gray': gray
        }
    
    def preserve_panda_colors(self, image: np.ndarray, features: Dict[str, np.ndarray]) -> np.ndarray:
        """保留熊猫的黑色和白色特征"""
        result = image.copy()
        
        # 保留黑色区域
        black_areas = features['black_mask'] > 0
        result[black_areas] = self.panda_colors['black']
        
        # 保留白色区域
        white_areas = features['white_mask'] > 0
        result[white_areas] = self.panda_colors['white']
        
        return result
    
    def enhance_background_colors(self, image: np.ndarray, features: Dict[str, np.ndarray]) -> np.ndarray:
        """增强背景色彩，添加红色、绿色、黄色等"""
        result = image.copy()
        
        # 创建背景掩码（非熊猫区域）
        background_mask = ~((features['black_mask'] > 0) | (features['white_mask'] > 0))
        
        # 使用SLIC超像素分割背景
        if np.sum(background_mask) > 1000:  # 确保有足够的背景区域
            background_img = image.copy()
            background_img[~background_mask] = [128, 128, 128]  # 非背景区域设为灰色
            
            # SLIC分割
            segments = slic(background_img, n_segments=50, compactness=10, start_label=1)
            
            # 为不同区域分配颜色
            colors = [
                [255, 0, 0],    # 红色
                [0, 255, 0],    # 绿色
                [255, 255, 0],  # 黄色
                [0, 0, 255],    # 蓝色
                [255, 0, 255],  # 紫色
                [0, 255, 255],  # 青色
                [255, 128, 0],  # 橙色
                [128, 0, 255],  # 紫罗兰
                [255, 192, 203], # 粉色
                [128, 128, 0],  # 橄榄色
            ]
            
            for i, segment_id in enumerate(np.unique(segments)):
                if segment_id == 0:  # 跳过背景
                    continue
                segment_mask = segments == segment_id
                segment_mask = segment_mask & background_mask
                
                if np.sum(segment_mask) > 100:  # 只处理足够大的区域
                    color_idx = (i - 1) % len(colors)
                    result[segment_mask] = colors[color_idx]
        
        return result
    
    def optimize_color_quantization(self, image: np.ndarray, n_colors: int = 20) -> Tuple[np.ndarray, List[Tuple[int, int, int]]]:
        """优化的色彩量化，确保使用指定的颜色数量"""
        # 检测熊猫特征
        features = self.detect_panda_features(image)
        
        # 保留熊猫特征
        result = self.preserve_panda_colors(image, features)
        
        # 增强背景色彩
        result = self.enhance_background_colors(result, features)
        
        # 转换为LAB色彩空间进行聚类
        lab_image = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
        Z = lab_image.reshape((-1, 3)).astype(np.float32)
        
        # 使用K-means聚类
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.1)
        K = min(n_colors, len(Z))
        _, labels, centers = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        
        # 转换回BGR
        centers = np.uint8(centers)
        quantized_lab = centers[labels.flatten()].reshape(lab_image.shape)
        quantized = cv2.cvtColor(quantized_lab, cv2.COLOR_LAB2BGR)
        
        # 确保熊猫特征仍然保留
        quantized = self.preserve_panda_colors(quantized, features)
        
        # 生成色表
        color_table = [tuple(map(int, c)) for c in centers]
        
        return quantized, color_table
    
    def smooth_boundaries(self, image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
        """平滑边界，减少锯齿"""
        # 使用双边滤波保持边缘
        smoothed = cv2.bilateralFilter(image, kernel_size, 75, 75)
        
        # 形态学操作进一步平滑
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        smoothed = cv2.morphologyEx(smoothed, cv2.MORPH_CLOSE, kernel)
        
        return smoothed
    
    def process_panda_image(self, 
                          image: np.ndarray, 
                          n_colors: int = 20, 
                          smooth_kernel: int = 5,
                          preserve_features: bool = True) -> Tuple[np.ndarray, List[Tuple[int, int, int]]]:
        """完整的熊猫图像处理流程"""
        
        # 1. 优化的色彩量化
        quantized, color_table = self.optimize_color_quantization(image, n_colors)
        
        # 2. 边界平滑
        if smooth_kernel > 0:
            quantized = self.smooth_boundaries(quantized, smooth_kernel)
        
        # 3. 最终特征保留
        if preserve_features:
            features = self.detect_panda_features(image)
            quantized = self.preserve_panda_colors(quantized, features)
        
        return quantized, color_table
    
    def export_color_table(self, color_table: List[Tuple[int, int, int]], file_path: str):
        """导出色表"""
        import csv
        with open(file_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Index', 'R', 'G', 'B'])
            for idx, (r, g, b) in enumerate(color_table):
                writer.writerow([idx, r, g, b])

def create_panda_optimized_tune_script():
    """创建熊猫优化的微调脚本"""
    script_content = '''# panda_optimized_tune.py
# 熊猫图像优化微调脚本

import cv2
import numpy as np
import os
import json
from datetime import datetime
from analysis_tools import AnalysisTools
from panda_optimized_processor import PandaOptimizedProcessor

def panda_optimized_tune(
    orig_path: str,
    pred_path: str,
    gt_path: str,
    out_dir: str = "panda_optimized_results",
    n_colors_list = [16, 18, 20],
    smooth_kernel_list = [3, 5, 7]
):
    os.makedirs(out_dir, exist_ok=True)
    
    # 加载图像
    orig = cv2.imread(orig_path)
    pred = cv2.imread(pred_path)
    gt = cv2.imread(gt_path)
    
    # 健壮性检查
    if orig is None:
        raise FileNotFoundError(f'原图未找到: {orig_path}')
    if pred is None:
        raise FileNotFoundError(f'程序生成识别图未找到: {pred_path}')
    if gt is None:
        raise FileNotFoundError(f'目标识别图未找到: {gt_path}')
    
    print(f"熊猫图像优化微调开始...")
    print(f"  原图尺寸: {orig.shape}")
    print(f"  程序生成图尺寸: {pred.shape}")
    print(f"  目标图尺寸: {gt.shape}")
    
    processor = PandaOptimizedProcessor()
    results = []
    
    for n_colors in n_colors_list:
        for kernel in smooth_kernel_list:
            print(f"\\n处理参数: 颜色={n_colors}, 核={kernel}")
            
            # 使用熊猫优化处理器
            quantized, color_table = processor.process_panda_image(
                orig, 
                n_colors=n_colors, 
                smooth_kernel=kernel,
                preserve_features=True
            )
            
            # 误差分析
            report, diff_vis = AnalysisTools.compare(gt, quantized)
            
            # 检查实际颜色数量
            unique_colors = np.unique(quantized.reshape(-1, 3), axis=0)
            actual_colors = len(unique_colors)
            
            if actual_colors < n_colors:
                print(f"  警告: 实际颜色数量({actual_colors})少于目标数量({n_colors})")
            
            # 保存结果
            tag = f"n{n_colors}_k{kernel}_panda_optimized"
            out_img = os.path.join(out_dir, f"result_{tag}.png")
            out_diff = os.path.join(out_dir, f"diff_{tag}.png")
            out_csv = os.path.join(out_dir, f"color_table_{tag}.csv")
            
            cv2.imwrite(out_img, quantized)
            cv2.imwrite(out_diff, diff_vis)
            processor.export_color_table(color_table, out_csv)
            
            # 记录结果
            result = {
                'params': {
                    'n_colors': n_colors,
                    'kernel': kernel,
                    'method': 'panda_optimized'
                },
                'report': report,
                'result_img': out_img,
                'diff_img': out_diff,
                'color_table': out_csv,
                'actual_colors': actual_colors
            }
            results.append(result)
            
            print(f"  完成: 色彩数={actual_colors}, 差异率={report['像素差异率']:.3f}")
    
    # 找到最佳结果
    best_result = min(results, key=lambda x: x['report']['像素差异率'])
    
    # 保存最佳结果
    with open(os.path.join(out_dir, "best_panda_result.json"), 'w') as f:
        json.dump(best_result, f, ensure_ascii=False, indent=2)
    
    # 保存完整结果
    with open(os.path.join(out_dir, "panda_optimized_summary.json"), 'w') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\\n最佳熊猫优化结果:")
    print(f"  参数: {best_result['params']}")
    print(f"  差异率: {best_result['report']['像素差异率']:.3f}")
    print(f"  实际色彩数: {best_result['actual_colors']}")
    
    print(f"\\n熊猫图像优化微调完成，结果保存在 {out_dir}")
    return results

if __name__ == '__main__':
    try:
        results = panda_optimized_tune(
            orig_path='orig.png',
            pred_path='pred.png',
            gt_path='gt.png'
        )
    except FileNotFoundError as e:
        print(f"错误: {e}. 请确保图片文件位于脚本运行目录。")
    except Exception as e:
        print(f"发生未知错误: {e}")
'''
    
    with open('panda_optimized_tune.py', 'w', encoding='utf-8') as f:
        f.write(script_content)
    
    print("熊猫优化微调脚本已创建: panda_optimized_tune.py")

if __name__ == '__main__':
    create_panda_optimized_tune_script() 