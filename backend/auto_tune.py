# auto_tune.py
# 自动化微调与误差分析脚本

import cv2
import numpy as np
import os
from analysis_tools import AnalysisTools
from image_processor import SichuanBrocadeProcessor

# AI模块导入（带错误处理）
try:
    from ai_segmentation import AISegmenter
    AI_SEGMENTATION_AVAILABLE = True
except ImportError:
    print("警告: ai_segmentation 模块不可用，将跳过AI分割功能")
    AI_SEGMENTATION_AVAILABLE = False
    AISegmenter = None

try:
    from feature_detector import FeatureDetector
    FEATURE_DETECTION_AVAILABLE = True
except ImportError:
    print("警告: feature_detector 模块不可用，将跳过特征点检测功能")
    FEATURE_DETECTION_AVAILABLE = False
    FeatureDetector = None

# 可选：风格迁移/色彩映射方法
try:
    from skimage.exposure import match_histograms
    STYLE_TRANSFER_AVAILABLE = True
except ImportError:
    print("警告: scikit-image 不可用，将跳过风格迁移功能")
    STYLE_TRANSFER_AVAILABLE = False

def auto_tune(
    orig_path: str,
    pred_path: str,
    gt_path: str,
    out_dir: str = "auto_tune_results",
    n_colors_list = [16, 18, 20],
    smooth_kernel_list = [3, 5, 7],
    style_transfer: bool = True,
    use_ai_segment: bool = False,
    ai_segment_model: str = 'sam',
    use_feature_detect: bool = False,
    feature_model: str = 'mediapipe'
):
    os.makedirs(out_dir, exist_ok=True)
    
    # 检查文件是否存在
    if not os.path.exists(orig_path):
        raise FileNotFoundError(f'原图文件不存在: {orig_path}')
    if not os.path.exists(pred_path):
        raise FileNotFoundError(f'程序生成识别图文件不存在: {pred_path}')
    if not os.path.exists(gt_path):
        raise FileNotFoundError(f'目标识别图文件不存在: {gt_path}')
    
    orig = cv2.imread(orig_path)
    pred = cv2.imread(pred_path)
    gt = cv2.imread(gt_path)
    
    if orig is None:
        raise FileNotFoundError(f'原图未找到或无法读取: {orig_path}')
    if pred is None:
        raise FileNotFoundError(f'程序生成识别图未找到或无法读取: {pred_path}')
    if gt is None:
        raise FileNotFoundError(f'目标识别图未找到或无法读取: {gt_path}')
    
    processor = SichuanBrocadeProcessor()
    
    # AI分割器（带可用性检查）
    ai_segmenter = None
    if use_ai_segment and AI_SEGMENTATION_AVAILABLE:
        try:
            ai_segmenter = AISegmenter(model_name=ai_segment_model)
        except Exception as e:
            print(f"警告: AI分割器初始化失败: {e}")
            use_ai_segment = False
    
    # 特征点检测器（带可用性检查）
    feature_detector = None
    if use_feature_detect and FEATURE_DETECTION_AVAILABLE:
        try:
            feature_detector = FeatureDetector(model_name=feature_model)
        except Exception as e:
            print(f"警告: 特征点检测器初始化失败: {e}")
            use_feature_detect = False
    
    # 检查风格迁移可用性
    if style_transfer and not STYLE_TRANSFER_AVAILABLE:
        print("警告: 风格迁移不可用，将跳过此功能")
        style_transfer = False
    
    results = []
    for n_colors in n_colors_list:
        for kernel in smooth_kernel_list:
            # 1. AI分割（可选）
            if use_ai_segment and ai_segmenter is not None:
                try:
                    seg_mask = ai_segmenter.segment(orig)
                    pred_proc = cv2.bitwise_and(pred, pred, mask=seg_mask)
                except Exception as e:
                    print(f"警告: AI分割失败，使用原始图像: {e}")
                    pred_proc = pred.copy()
            else:
                pred_proc = pred.copy()
            
            # 2. 特征点检测（可选）
            if use_feature_detect and feature_detector is not None:
                try:
                    keypoints = feature_detector.detect(orig)
                    # TODO: 利用关键点优化分割/色块（如眼鼻耳区域特殊处理）
                    # 这里只做占位，后续可扩展
                except Exception as e:
                    print(f"警告: 特征点检测失败: {e}")
            
            # 3. 风格迁移/色彩映射
            if style_transfer:
                try:
                    pred_proc = match_histograms(pred_proc, gt, channel_axis=-1)
                except Exception as e:
                    print(f"警告: 风格迁移失败: {e}")
            
            # 4. 色彩聚类
            quant_img, color_table = processor.color_quantize(pred_proc, n_colors=n_colors)
            
            # 5. 边界平滑
            mask = cv2.cvtColor(quant_img, cv2.COLOR_BGR2GRAY)
            mask = processor.smooth_boundary(mask, kernel_size=kernel)
            quant_img[mask == 0] = 255
            
            # 6. 误差分析
            report, diff_vis = AnalysisTools.compare(gt, quant_img)
            
            # 7. 输出结果
            tag = f"n{n_colors}_k{kernel}_style{int(style_transfer)}_ai{int(use_ai_segment)}_fd{int(use_feature_detect)}"
            out_img = os.path.join(out_dir, f"result_{tag}.png")
            out_diff = os.path.join(out_dir, f"diff_{tag}.png")
            out_csv = os.path.join(out_dir, f"color_table_{tag}.csv")
            
            cv2.imwrite(out_img, quant_img)
            cv2.imwrite(out_diff, diff_vis)
            processor.export_color_table(color_table, out_csv)
            
            results.append({
                'params': {
                    'n_colors': n_colors, 
                    'kernel': kernel, 
                    'style_transfer': style_transfer, 
                    'ai_segment': use_ai_segment, 
                    'feature_detect': use_feature_detect
                },
                'report': report,
                'result_img': out_img,
                'diff_img': out_diff,
                'color_table': out_csv
            })
    
    # 汇总报告
    import json
    with open(os.path.join(out_dir, "summary.json"), 'w') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"自动化微调完成，结果保存在 {out_dir}")
    print(f"共处理了 {len(results)} 组参数组合")

# 示例用法
if __name__ == "__main__":
    auto_tune(
        orig_path="orig.png",
        pred_path="pred.png",
        gt_path="gt.png",
        out_dir="auto_tune_results",
        n_colors_list=[16, 18, 20],
        smooth_kernel_list=[3, 5, 7],
        style_transfer=True
    ) 