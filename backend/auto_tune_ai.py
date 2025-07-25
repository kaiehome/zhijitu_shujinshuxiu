# auto_tune_ai.py
# AI增强版自动化微调与误差分析脚本

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

def auto_tune_ai(
    orig_path: str,
    pred_path: str,
    gt_path: str,
    out_dir: str = "auto_tune_ai_results",
    n_colors_list = [16, 18, 20],
    smooth_kernel_list = [3, 5, 7],
    style_transfer: bool = True,
    ai_segment_models = ['grabcut', 'watershed', 'slic', 'contour'],
    feature_models = ['opencv', 'contour', 'blob'],
    use_feature_guided_segmentation: bool = True
):
    """
    AI增强版自动化微调
    
    Args:
        orig_path: 原图路径
        pred_path: 程序生成识别图路径
        gt_path: 目标识别图路径
        out_dir: 输出目录
        n_colors_list: 色彩数量列表
        smooth_kernel_list: 平滑核大小列表
        style_transfer: 是否启用风格迁移
        ai_segment_models: AI分割模型列表
        feature_models: 特征检测模型列表
        use_feature_guided_segmentation: 是否使用特征引导分割
    """
    
    # 创建输出目录
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
    
    print(f"✅ 图像加载成功:")
    print(f"   原图: {orig.shape}")
    print(f"   程序生成图: {pred.shape}")
    print(f"   目标图: {gt.shape}")
    
    # 初始化处理器
    processor = SichuanBrocadeProcessor()
    
    # 初始化AI分割器
    segmenters = {}
    for model_name in ai_segment_models:
        try:
            segmenters[model_name] = AISegmenter(model_name=model_name)
            print(f"✅ AI分割器 {model_name} 初始化成功")
        except Exception as e:
            print(f"⚠️ AI分割器 {model_name} 初始化失败: {e}")
    
    # 初始化特征检测器
    detectors = {}
    for model_name in feature_models:
        try:
            detectors[model_name] = FeatureDetector(model_name=model_name)
            print(f"✅ 特征检测器 {model_name} 初始化成功")
        except Exception as e:
            print(f"⚠️ 特征检测器 {model_name} 初始化失败: {e}")
    
    results = []
    total_combinations = len(n_colors_list) * len(smooth_kernel_list) * len(segmenters) * len(detectors)
    current_combination = 0
    
    print(f"\n🚀 开始AI增强微调，共 {total_combinations} 种参数组合...")
    
    for n_colors in n_colors_list:
        for kernel in smooth_kernel_list:
            for seg_model_name, segmenter in segmenters.items():
                for feat_model_name, detector in detectors.items():
                    current_combination += 1
                    print(f"\n📊 进度: {current_combination}/{total_combinations}")
                    print(f"   参数: 色彩={n_colors}, 核={kernel}, 分割={seg_model_name}, 特征={feat_model_name}")
                    
                    try:
                        # 1. AI分割
                        seg_params = {
                            'n_segments': 100 if seg_model_name == 'slic' else None,
                            'compactness': 10 if seg_model_name == 'slic' else None
                        }
                        seg_mask = segmenter.segment(orig, seg_params)
                        
                        # 2. 特征点检测
                        feat_params = {
                            'face_scale_factor': 1.1,
                            'face_min_neighbors': 5,
                            'eye_scale_factor': 1.1,
                            'eye_min_neighbors': 3
                        }
                        keypoints = detector.detect(orig, feat_params)
                        
                        # 3. 特征引导分割（可选）
                        if use_feature_guided_segmentation and keypoints:
                            seg_mask = _apply_feature_guidance(seg_mask, keypoints, orig.shape)
                        
                        # 4. 应用分割掩码
                        pred_proc = pred.copy()
                        if seg_mask is not None and seg_mask.sum() > 0:
                            # 将掩码应用到处理图像
                            pred_proc = cv2.bitwise_and(pred_proc, pred_proc, mask=seg_mask)
                            # 将背景设为白色
                            pred_proc[seg_mask == 0] = [255, 255, 255]
                        
                        # 5. 风格迁移/色彩映射
                        if style_transfer and STYLE_TRANSFER_AVAILABLE:
                            pred_proc = match_histograms(pred_proc, gt, channel_axis=-1)
                        
                        # 6. 色彩聚类
                        quant_img, color_table = processor.color_quantize(pred_proc, n_colors=n_colors)
                        
                        # 7. 边界平滑
                        mask = cv2.cvtColor(quant_img, cv2.COLOR_BGR2GRAY)
                        mask = processor.smooth_boundary(mask, kernel_size=kernel)
                        quant_img[mask == 0] = 255
                        
                        # 8. 误差分析
                        report, diff_vis = AnalysisTools.compare(gt, quant_img)
                        
                        # 9. 保存结果
                        tag = f"n{n_colors}_k{kernel}_seg{seg_model_name}_feat{feat_model_name}_style{int(style_transfer)}"
                        out_img = os.path.join(out_dir, f"result_{tag}.png")
                        out_diff = os.path.join(out_dir, f"diff_{tag}.png")
                        out_csv = os.path.join(out_dir, f"color_table_{tag}.csv")
                        out_mask = os.path.join(out_dir, f"mask_{tag}.png")
                        
                        cv2.imwrite(out_img, quant_img)
                        cv2.imwrite(out_diff, diff_vis)
                        cv2.imwrite(out_mask, seg_mask)
                        processor.export_color_table(color_table, out_csv)
                        
                        # 10. 记录结果
                        result = {
                            'params': {
                                'n_colors': n_colors,
                                'kernel': kernel,
                                'segmentation_model': seg_model_name,
                                'feature_model': feat_model_name,
                                'style_transfer': style_transfer,
                                'feature_guided': use_feature_guided_segmentation
                            },
                            'report': report,
                            'keypoints_count': len(keypoints),
                            'result_img': out_img,
                            'diff_img': out_diff,
                            'color_table': out_csv,
                            'mask_img': out_mask,
                            'timestamp': datetime.now().isoformat()
                        }
                        
                        results.append(result)
                        print(f"   ✅ 完成，像素差异率: {report['像素差异率']:.4f}")
                        
                    except Exception as e:
                        print(f"   ❌ 失败: {e}")
                        continue
    
    # 保存汇总报告
    summary_path = os.path.join(out_dir, "ai_summary.json")
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    # 生成最佳结果分析
    _generate_best_results_analysis(results, out_dir)
    
    print(f"\n🎉 AI增强微调完成！")
    print(f"   结果保存在: {out_dir}")
    print(f"   成功组合数: {len(results)}/{total_combinations}")
    print(f"   汇总报告: {summary_path}")

def _apply_feature_guidance(mask: np.ndarray, keypoints: dict, image_shape: tuple) -> np.ndarray:
    """应用特征点引导优化分割掩码"""
    enhanced_mask = mask.copy()
    
    # 根据检测到的特征点优化掩码
    for feature_name, feature_data in keypoints.items():
        if 'center' in feature_data:
            center = feature_data['center']
            if isinstance(center, tuple) and len(center) == 2:
                x, y = center
                # 在特征点周围创建圆形区域
                cv2.circle(enhanced_mask, (int(x), int(y)), 20, 255, -1)
        
        if 'bbox' in feature_data:
            bbox = feature_data['bbox']
            if isinstance(bbox, tuple) and len(bbox) == 4:
                x, y, w, h = bbox
                # 在边界框区域填充
                cv2.rectangle(enhanced_mask, (int(x), int(y)), (int(x+w), int(y+h)), 255, -1)
    
    return enhanced_mask

def _generate_best_results_analysis(results: list, out_dir: str):
    """生成最佳结果分析"""
    if not results:
        return
    
    # 按像素差异率排序
    sorted_results = sorted(results, key=lambda x: x['report']['像素差异率'])
    
    # 生成最佳结果报告
    best_results = {
        'best_overall': sorted_results[0],
        'best_by_segmentation': {},
        'best_by_feature': {},
        'statistics': {
            'total_combinations': len(results),
            'avg_pixel_diff': np.mean([r['report']['像素差异率'] for r in results]),
            'min_pixel_diff': min([r['report']['像素差异率'] for r in results]),
            'max_pixel_diff': max([r['report']['像素差异率'] for r in results])
        }
    }
    
    # 按分割模型分组最佳结果
    seg_models = set(r['params']['segmentation_model'] for r in results)
    for seg_model in seg_models:
        seg_results = [r for r in results if r['params']['segmentation_model'] == seg_model]
        best_seg = min(seg_results, key=lambda x: x['report']['像素差异率'])
        best_results['best_by_segmentation'][seg_model] = best_seg
    
    # 按特征模型分组最佳结果
    feat_models = set(r['params']['feature_model'] for r in results)
    for feat_model in feat_models:
        feat_results = [r for r in results if r['params']['feature_model'] == feat_model]
        best_feat = min(feat_results, key=lambda x: x['report']['像素差异率'])
        best_results['best_by_feature'][feat_model] = best_feat
    
    # 保存最佳结果分析
    best_analysis_path = os.path.join(out_dir, "best_results_analysis.json")
    with open(best_analysis_path, 'w', encoding='utf-8') as f:
        json.dump(best_results, f, ensure_ascii=False, indent=2)
    
    print(f"\n🏆 最佳结果分析:")
    print(f"   最佳整体: {best_results['best_overall']['params']}")
    print(f"   像素差异率: {best_results['best_overall']['report']['像素差异率']:.4f}")
    print(f"   分析报告: {best_analysis_path}")

if __name__ == "__main__":
    # 运行AI增强微调
    auto_tune_ai(
        orig_path="orig.png",
        pred_path="pred.png", 
        gt_path="gt.png",
        out_dir="auto_tune_ai_results",
        n_colors_list=[16, 18, 20],
        smooth_kernel_list=[3, 5, 7],
        style_transfer=True,
        ai_segment_models=['grabcut', 'watershed', 'slic', 'contour'],
        feature_models=['opencv', 'contour', 'blob'],
        use_feature_guided_segmentation=True
    ) 