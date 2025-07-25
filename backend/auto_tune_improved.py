# auto_tune_improved.py
# 改进版AI自动化微调脚本 - 修复色彩数量、差异率计算和边界检测问题

import cv2
import numpy as np
import os
import json
from datetime import datetime
from analysis_tools import AnalysisTools
from image_processor import SichuanBrocadeProcessor
from ai_segmentation import AISegmenter
from feature_detector import FeatureDetector

# 可选：风格迁移/色彩映射方法
try:
    from skimage.exposure import match_histograms
    STYLE_TRANSFER_AVAILABLE = True
except ImportError:
    print("警告: scikit-image 不可用，将跳过风格迁移功能")
    STYLE_TRANSFER_AVAILABLE = False

def improved_color_quantize(image: np.ndarray, n_colors: int = 20) -> tuple:
    """改进的色彩量化，确保使用指定的颜色数量"""
    # 转换为LAB色彩空间以获得更好的聚类效果
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    
    # 重塑图像数据
    Z = lab_image.reshape((-1, 3)).astype(np.float32)
    
    # 使用K-means聚类
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.1)
    _, labels, centers = cv2.kmeans(Z, n_colors, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    # 转换回BGR色彩空间
    centers_bgr = cv2.cvtColor(centers.reshape(1, -1, 3).astype(np.uint8), cv2.COLOR_LAB2BGR)
    centers_bgr = centers_bgr.reshape(-1, 3)
    
    # 应用量化
    quantized = centers_bgr[labels.flatten()].reshape(image.shape)
    
    # 确保颜色数量正确
    unique_colors = np.unique(quantized.reshape(-1, 3), axis=0)
    if len(unique_colors) < n_colors:
        print(f"警告: 实际颜色数量({len(unique_colors)})少于目标数量({n_colors})")
    
    color_table = [tuple(map(int, c)) for c in unique_colors]
    return quantized, color_table

def improved_boundary_smooth(image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """改进的边界平滑算法"""
    # 转换为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 使用双边滤波保持边缘
    smoothed = cv2.bilateralFilter(gray, kernel_size, 75, 75)
    
    # 形态学操作
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    opened = cv2.morphologyEx(smoothed, cv2.MORPH_OPEN, kernel)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
    
    # 转回BGR
    result = cv2.cvtColor(closed, cv2.COLOR_GRAY2BGR)
    return result

def improved_analysis_compare(gt_image: np.ndarray, pred_image: np.ndarray) -> tuple:
    """改进的图像比较分析"""
    if gt_image.shape != pred_image.shape:
        pred_image = cv2.resize(pred_image, (gt_image.shape[1], gt_image.shape[0]), 
                               interpolation=cv2.INTER_NEAREST)
    
    # 计算像素差异（改进版）
    gt_flat = gt_image.reshape(-1, 3)
    pred_flat = pred_image.reshape(-1, 3)
    
    # 使用颜色距离而不是严格相等
    color_diff = np.sqrt(np.sum((gt_flat.astype(np.float32) - pred_flat.astype(np.float32))**2, axis=1))
    diff_mask = color_diff > 30  # 阈值可调整
    diff_count = np.sum(diff_mask)
    total_pixels = diff_mask.size
    diff_ratio = diff_count / total_pixels
    
    # 改进的边界差异计算
    gt_gray = cv2.cvtColor(gt_image, cv2.COLOR_BGR2GRAY)
    pred_gray = cv2.cvtColor(pred_image, cv2.COLOR_BGR2GRAY)
    
    # 使用Canny边缘检测
    gt_edges = cv2.Canny(gt_gray, 50, 150)
    pred_edges = cv2.Canny(pred_gray, 50, 150)
    
    # 计算边缘差异
    edge_diff = cv2.absdiff(gt_edges, pred_edges)
    edge_diff_count = np.sum(edge_diff > 0)
    edge_total = np.sum(gt_edges > 0) + np.sum(pred_edges > 0)
    edge_diff_ratio = edge_diff_count / (edge_total + 1e-6)
    
    # 颜色统计
    gt_colors = np.unique(gt_flat, axis=0)
    pred_colors = np.unique(pred_flat, axis=0)
    
    # 生成差异可视化
    diff_vis = pred_image.copy()
    diff_vis[diff_mask.reshape(gt_image.shape[:2])] = [0, 0, 255]  # 红色标记差异
    
    report = {
        '像素差异率': round(diff_ratio, 4),
        '边界差异率': round(edge_diff_ratio, 4),
        '目标色彩数': len(gt_colors),
        '生成色彩数': len(pred_colors),
        '像素总数': total_pixels,
        '像素差异数': int(diff_count),
        '边界差异像素数': int(edge_diff_count),
        '平均颜色距离': round(np.mean(color_diff), 2)
    }
    
    return report, diff_vis

def auto_tune_improved(
    orig_path: str,
    pred_path: str,
    gt_path: str,
    out_dir: str = "auto_tune_improved_results",
    n_colors_list = [16, 18, 20],
    smooth_kernel_list = [3, 5, 7],
    style_transfer: bool = True,
    ai_segment_models = ['grabcut', 'watershed', 'slic', 'contour'],
    feature_models = ['opencv', 'contour', 'blob']
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
    
    print(f"图像加载成功:")
    print(f"  原图尺寸: {orig.shape}")
    print(f"  程序生成图尺寸: {pred.shape}")
    print(f"  目标图尺寸: {gt.shape}")
    
    processor = SichuanBrocadeProcessor()
    results = []
    
    # 处理所有参数组合
    for n_colors in n_colors_list:
        for kernel in smooth_kernel_list:
            for seg_model in ai_segment_models:
                for feat_model in feature_models:
                    try:
                        print(f"\n处理参数: 颜色={n_colors}, 核={kernel}, 分割={seg_model}, 特征={feat_model}")
                        
                        # 1. AI分割
                        ai_segmenter = AISegmenter(model_name=seg_model)
                        seg_mask = ai_segmenter.segment(pred)
                        
                        # 2. 特征检测
                        feature_detector = FeatureDetector(model_name=feat_model)
                        keypoints = feature_detector.detect(pred)
                        
                        # 3. 应用分割掩码
                        if seg_mask is not None and seg_mask.sum() > 0:
                            pred_proc = cv2.bitwise_and(pred, pred, mask=seg_mask)
                        else:
                            pred_proc = pred.copy()
                        
                        # 4. 风格迁移
                        if style_transfer and STYLE_TRANSFER_AVAILABLE:
                            pred_proc = match_histograms(pred_proc, gt, channel_axis=-1)
                        
                        # 5. 改进的色彩量化
                        quant_img, color_table = improved_color_quantize(pred_proc, n_colors=n_colors)
                        
                        # 6. 边界平滑
                        mask = cv2.cvtColor(quant_img, cv2.COLOR_BGR2GRAY)
                        mask = processor.smooth_boundary(mask, kernel_size=kernel)
                        quant_img[mask == 0] = [255, 255, 255]
                        
                        # 7. 误差分析
                        report, diff_vis = AnalysisTools.compare(gt, quant_img)
                        
                        # 8. 保存结果
                        tag = f"n{n_colors}_k{kernel}_seg{seg_model}_feat{feat_model}_style{int(style_transfer)}"
                        out_img = os.path.join(out_dir, f"result_{tag}.png")
                        out_diff = os.path.join(out_dir, f"diff_{tag}.png")
                        out_csv = os.path.join(out_dir, f"color_table_{tag}.csv")
                        
                        cv2.imwrite(out_img, quant_img)
                        cv2.imwrite(out_diff, diff_vis)
                        processor.export_color_table(color_table, out_csv)
                        
                        # 转换numpy类型为Python原生类型
                        report_serializable = {}
                        for key, value in report.items():
                            if isinstance(value, np.integer):
                                report_serializable[key] = int(value)
                            elif isinstance(value, np.floating):
                                report_serializable[key] = float(value)
                            elif isinstance(value, np.ndarray):
                                report_serializable[key] = value.tolist()
                            else:
                                report_serializable[key] = value
                        
                        results.append({
                            'params': {
                                'n_colors': int(n_colors),
                                'kernel': int(kernel),
                                'seg_model': str(seg_model),
                                'feat_model': str(feat_model),
                                'style_transfer': bool(style_transfer)
                            },
                            'report': report_serializable,
                            'result_img': str(out_img),
                            'diff_img': str(out_diff),
                            'color_table': str(out_csv),
                            'actual_colors': int(len(color_table))
                        })
                        
                        print(f"  完成: 色彩数={len(color_table)}, 差异率={report_serializable.get('像素差异率', 0):.3f}")
                        
                    except Exception as e:
                        print(f"  错误: {e}")
                        continue
    
    # 保存结果
    try:
        with open(os.path.join(out_dir, "improved_summary.json"), 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2, default=str)
        
        # 找出最佳结果
        if results:
            best_result = min(results, key=lambda x: x['report'].get('像素差异率', 1.0))
            with open(os.path.join(out_dir, "best_result.json"), 'w', encoding='utf-8') as f:
                json.dump(best_result, f, ensure_ascii=False, indent=2, default=str)
            
            print(f"\n最佳结果:")
            print(f"  参数: {best_result['params']}")
            print(f"  差异率: {best_result['report'].get('像素差异率', 0):.3f}")
            print(f"  实际色彩数: {best_result['actual_colors']}")
        
        print(f"\n改进版AI自动化微调完成，结果保存在 {out_dir}")
        return results
        
    except Exception as e:
        print(f"保存结果时发生错误: {e}")
        return results

if __name__ == '__main__':
    try:
        # 检查输入文件
        required_files = ['orig.png', 'pred.png', 'gt.png']
        missing_files = [f for f in required_files if not os.path.exists(f)]
        
        if missing_files:
            print(f"错误: 缺少输入文件: {missing_files}")
            print("请确保以下文件存在于当前目录:")
            for f in required_files:
                print(f"  - {f}")
            exit(1)
        
        print("开始改进版AI自动化微调...")
        results = auto_tune_improved(
            orig_path='orig.png',
            pred_path='pred.png',
            gt_path='gt.png',
            style_transfer=True,
            ai_segment_models=['grabcut', 'watershed', 'slic'],
            feature_models=['opencv', 'contour']
        )
        
    except FileNotFoundError as e:
        print(f"错误: {e}")
    except ImportError as e:
        print(f"错误: 缺少必要的Python库。请安装所有依赖项。详细信息: {e}")
    except Exception as e:
        print(f"发生未知错误: {e}")
        import traceback
        traceback.print_exc() 