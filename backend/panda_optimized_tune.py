# panda_optimized_tune.py
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
            print(f"\n处理参数: 颜色={n_colors}, 核={kernel}")
            
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
    
    print(f"\n最佳熊猫优化结果:")
    print(f"  参数: {best_result['params']}")
    print(f"  差异率: {best_result['report']['像素差异率']:.3f}")
    print(f"  实际色彩数: {best_result['actual_colors']}")
    
    print(f"\n熊猫图像优化微调完成，结果保存在 {out_dir}")
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
