# analysis_tools.py
# 识别图误差分析与可视化

import numpy as np
import cv2
from typing import Tuple, Dict

class AnalysisTools:
    @staticmethod
    def compare(gt_image: np.ndarray, pred_image: np.ndarray) -> Tuple[Dict, np.ndarray]:
        """
        对比两张图像，输出误差报告和可视化diff图
        - gt_image: 目标识别图（ground truth）
        - pred_image: 程序生成的识别图
        返回：误差统计dict, diff可视化图
        """
        # 1. 尺寸对齐
        if gt_image.shape != pred_image.shape:
            pred_image = cv2.resize(pred_image, (gt_image.shape[1], gt_image.shape[0]), interpolation=cv2.INTER_NEAREST)
        # 2. 颜色聚类（简化色彩对比）
        gt_flat = gt_image.reshape(-1, 3)
        pred_flat = pred_image.reshape(-1, 3)
        # 3. 计算像素级差异
        diff_mask = np.any(gt_flat != pred_flat, axis=1).reshape(gt_image.shape[:2])
        diff_count = np.sum(diff_mask)
        total_pixels = diff_mask.size
        diff_ratio = diff_count / total_pixels
        # 4. 计算边界差异（Canny边缘）
        gt_edges = cv2.Canny(cv2.cvtColor(gt_image, cv2.COLOR_BGR2GRAY), 50, 150)
        pred_edges = cv2.Canny(cv2.cvtColor(pred_image, cv2.COLOR_BGR2GRAY), 50, 150)
        edge_diff = cv2.absdiff(gt_edges, pred_edges)
        edge_diff_count = np.sum(edge_diff > 0)
        edge_total = np.sum(gt_edges > 0) + np.sum(pred_edges > 0)
        edge_diff_ratio = edge_diff_count / (edge_total + 1e-6)
        # 5. 统计色彩数量
        gt_colors = np.unique(gt_flat, axis=0)
        pred_colors = np.unique(pred_flat, axis=0)
        # 6. 可视化diff
        diff_vis = pred_image.copy()
        diff_vis[diff_mask] = [0,0,255]  # 差异像素标红
        # 7. 汇总报告
        report = {
            '像素差异率': round(diff_ratio, 4),
            '边界差异率': round(edge_diff_ratio, 4),
            '目标色彩数': len(gt_colors),
            '生成色彩数': len(pred_colors),
            '像素总数': total_pixels,
            '像素差异数': int(diff_count),
            '边界差异像素数': int(edge_diff_count)
        }
        return report, diff_vis

# 示例用法
if __name__ == "__main__":
    gt = cv2.imread("gt.png")
    pred = cv2.imread("pred.png")
    report, diff = AnalysisTools.compare(gt, pred)
    print(report)
    cv2.imwrite("diff_vis.png", diff) 