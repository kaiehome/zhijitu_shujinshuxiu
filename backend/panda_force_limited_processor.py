# panda_force_limited_processor.py
# 强制限制色彩数量的熊猫图像处理器

import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional
import json

class PandaForceLimitedProcessor:
    def __init__(self):
        # 预定义的熊猫色彩方案（确保不超过20色）
        self.panda_color_scheme = {
            'black': [0, 0, 0],           # 熊猫黑色特征
            'white': [255, 255, 255],     # 熊猫白色特征
            'gray': [128, 128, 128],      # 灰色过渡
            'light_gray': [192, 192, 192], # 浅灰色
            'red': [255, 0, 0],           # 红色（印章等）
            'dark_red': [128, 0, 0],      # 深红色
            'green': [0, 255, 0],         # 绿色（背景）
            'dark_green': [0, 128, 0],    # 深绿色
            'yellow': [255, 255, 0],      # 黄色（花朵）
            'orange': [255, 128, 0],      # 橙色
            'blue': [0, 0, 255],          # 蓝色
            'light_blue': [128, 128, 255], # 浅蓝色
            'purple': [128, 0, 128],      # 紫色
            'pink': [255, 192, 203],      # 粉色
            'brown': [128, 64, 0],        # 棕色
            'light_brown': [192, 128, 64], # 浅棕色
            'cream': [255, 255, 240],     # 奶油色
            'beige': [245, 245, 220],     # 米色
            'navy': [0, 0, 128],          # 深蓝色
            'olive': [128, 128, 0]        # 橄榄色
        }
        
    def detect_panda_features(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """检测熊猫特征区域"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 检测黑色区域（熊猫特征）
        _, black_mask = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
        
        # 检测白色区域
        _, white_mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        
        return {
            'black_mask': black_mask,
            'white_mask': white_mask,
            'gray': gray
        }
    
    def force_limited_colors(self, image: np.ndarray, n_colors: int = 20) -> Tuple[np.ndarray, List[Tuple[int, int, int]]]:
        """强制限制色彩数量"""
        # 获取预定义颜色列表
        color_list = list(self.panda_color_scheme.values())
        if n_colors < len(color_list):
            color_list = color_list[:n_colors]
        
        # 检测熊猫特征
        features = self.detect_panda_features(image)
        
        # 创建结果图像
        result = np.zeros_like(image)
        
        # 1. 保留熊猫特征
        black_areas = features['black_mask'] > 0
        white_areas = features['white_mask'] > 0
        
        result[black_areas] = self.panda_color_scheme['black']
        result[white_areas] = self.panda_color_scheme['white']
        
        # 2. 处理背景区域
        background_mask = ~(black_areas | white_areas)
        
        if np.sum(background_mask) > 1000:
            # 将背景区域转换为灰度
            background_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            background_gray = background_gray * background_mask.astype(np.uint8)
            
            # 使用K-means将背景区域聚类为剩余颜色数量
            remaining_colors = n_colors - 2  # 减去黑色和白色
            if remaining_colors > 0:
                background_colors = [c for c in color_list if c not in [self.panda_color_scheme['black'], self.panda_color_scheme['white']]]
                background_colors = background_colors[:remaining_colors]
                
                # 对背景区域进行聚类
                background_pixels = background_gray[background_mask].reshape(-1, 1).astype(np.float32)
                
                if len(background_pixels) > remaining_colors:
                    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.1)
                    _, labels, centers = cv2.kmeans(background_pixels, remaining_colors, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
                    
                    # 将聚类结果映射到预定义颜色
                    for i, color in enumerate(background_colors):
                        # 修复形状不匹配问题
                        cluster_mask_flat = (labels.flatten() == i)
                        cluster_mask = np.zeros_like(background_mask, dtype=bool)
                        cluster_mask[background_mask] = cluster_mask_flat
                        result[cluster_mask] = color
                else:
                    # 如果像素太少，直接分配颜色
                    result[background_mask] = background_colors[0] if background_colors else [128, 128, 128]
        
        # 3. 强制确保颜色数量不超过限制
        unique_colors = np.unique(result.reshape(-1, 3), axis=0)
        if len(unique_colors) > n_colors:
            # 如果仍然超过限制，进行最终的K-means聚类
            Z = result.reshape((-1, 3)).astype(np.float32)
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.1)
            _, labels, centers = cv2.kmeans(Z, n_colors, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
            
            centers = np.uint8(centers)
            result = centers[labels.flatten()].reshape(result.shape)
            
            # 重新应用熊猫特征
            result[black_areas] = self.panda_color_scheme['black']
            result[white_areas] = self.panda_color_scheme['white']
        
        # 生成最终色表
        final_colors = np.unique(result.reshape(-1, 3), axis=0)
        color_table = [tuple(map(int, c)) for c in final_colors]
        
        return result, color_table
    
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
                          smooth_kernel: int = 5) -> Tuple[np.ndarray, List[Tuple[int, int, int]]]:
        """完整的熊猫图像处理流程"""
        
        # 1. 强制限制色彩数量
        result, color_table = self.force_limited_colors(image, n_colors)
        
        # 2. 边界平滑
        if smooth_kernel > 0:
            result = self.smooth_boundaries(result, smooth_kernel)
            
            # 平滑后重新应用熊猫特征
            features = self.detect_panda_features(image)
            result[features['black_mask'] > 0] = self.panda_color_scheme['black']
            result[features['white_mask'] > 0] = self.panda_color_scheme['white']
        
        # 3. 最终颜色验证和强制限制
        final_colors = np.unique(result.reshape(-1, 3), axis=0)
        if len(final_colors) > n_colors:
            # 最后一次强制限制
            Z = result.reshape((-1, 3)).astype(np.float32)
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.1)
            _, labels, centers = cv2.kmeans(Z, n_colors, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
            
            centers = np.uint8(centers)
            result = centers[labels.flatten()].reshape(result.shape)
            
            # 重新生成色表
            final_colors = np.unique(result.reshape(-1, 3), axis=0)
            color_table = [tuple(map(int, c)) for c in final_colors]
        
        return result, color_table
    
    def export_color_table(self, color_table: List[Tuple[int, int, int]], file_path: str):
        """导出色表"""
        import csv
        with open(file_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Index', 'R', 'G', 'B'])
            for idx, (r, g, b) in enumerate(color_table):
                writer.writerow([idx, r, g, b])

def create_panda_force_limited_tune_script():
    """创建熊猫强制限制微调脚本"""
    script_content = '''# panda_force_limited_tune.py
# 熊猫图像强制限制微调脚本

import cv2
import numpy as np
import os
import json
from datetime import datetime
from analysis_tools import AnalysisTools
from panda_force_limited_processor import PandaForceLimitedProcessor

def panda_force_limited_tune(
    orig_path: str,
    pred_path: str,
    gt_path: str,
    out_dir: str = "panda_force_limited_results",
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
    
    print(f"熊猫图像强制限制微调开始...")
    print(f"  原图尺寸: {orig.shape}")
    print(f"  程序生成图尺寸: {pred.shape}")
    print(f"  目标图尺寸: {gt.shape}")
    
    processor = PandaForceLimitedProcessor()
    results = []
    
    for n_colors in n_colors_list:
        for kernel in smooth_kernel_list:
            print(f"\\n处理参数: 颜色={n_colors}, 核={kernel}")
            
            # 使用熊猫强制限制处理器
            quantized, color_table = processor.process_panda_image(
                orig, 
                n_colors=n_colors, 
                smooth_kernel=kernel
            )
            
            # 验证颜色数量
            unique_colors = np.unique(quantized.reshape(-1, 3), axis=0)
            actual_colors = len(unique_colors)
            
            if actual_colors > n_colors:
                print(f"  警告: 实际颜色数量({actual_colors})超过目标数量({n_colors})")
            else:
                print(f"  成功: 实际颜色数量({actual_colors})符合目标数量({n_colors})")
            
            # 误差分析
            report, diff_vis = AnalysisTools.compare(gt, quantized)
            
            # 保存结果
            tag = f"n{n_colors}_k{kernel}_panda_force_limited"
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
                    'method': 'panda_force_limited'
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
    with open(os.path.join(out_dir, "best_panda_force_limited_result.json"), 'w') as f:
        json.dump(best_result, f, ensure_ascii=False, indent=2)
    
    # 保存完整结果
    with open(os.path.join(out_dir, "panda_force_limited_summary.json"), 'w') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\\n最佳熊猫强制限制结果:")
    print(f"  参数: {best_result['params']}")
    print(f"  差异率: {best_result['report']['像素差异率']:.3f}")
    print(f"  实际色彩数: {best_result['actual_colors']}")
    
    print(f"\\n熊猫图像强制限制微调完成，结果保存在 {out_dir}")
    return results

if __name__ == '__main__':
    try:
        results = panda_force_limited_tune(
            orig_path='orig.png',
            pred_path='pred.png',
            gt_path='gt.png'
        )
    except FileNotFoundError as e:
        print(f"错误: {e}. 请确保图片文件位于脚本运行目录。")
    except Exception as e:
        print(f"发生未知错误: {e}")
'''
    
    with open('panda_force_limited_tune.py', 'w', encoding='utf-8') as f:
        f.write(script_content)
    
    print("熊猫强制限制微调脚本已创建: panda_force_limited_tune.py")

if __name__ == '__main__':
    create_panda_force_limited_tune_script() 