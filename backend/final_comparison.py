#!/usr/bin/env python3
"""
最终对比测试
比较极简版本、优化版本和复杂版本的效果
"""

import os
import cv2
import numpy as np
import logging
import time
import subprocess
from simple_professional_generator import SimpleProfessionalGenerator
from optimized_professional_generator import OptimizedProfessionalGenerator
from professional_weaving_generator import ProfessionalWeavingGenerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def analyze_image_quality(original_path: str, processed_path: str) -> dict:
    """分析图像质量指标"""
    try:
        # 安装必要的依赖
        try:
            from skimage.metrics import structural_similarity as ssim
        except ImportError:
            subprocess.run(["pip", "install", "scikit-image"], check=False, capture_output=True)
            from skimage.metrics import structural_similarity as ssim
        
        # 加载图像
        original = cv2.imread(original_path)
        processed = cv2.imread(processed_path)
        
        if original is None or processed is None:
            return {"error": "无法加载图像"}
        
        # 转换为RGB
        original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
        processed_rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
        
        # 分析指标
        metrics = {}
        
        # 1. 颜色数量
        orig_colors = len(np.unique(original_rgb.reshape(-1, 3), axis=0))
        proc_colors = len(np.unique(processed_rgb.reshape(-1, 3), axis=0))
        metrics["color_reduction"] = (1 - proc_colors/orig_colors) * 100
        
        # 2. 结构相似性（最重要指标）
        if original.shape != processed.shape:
            processed_resized = cv2.resize(processed, (original.shape[1], original.shape[0]))
        else:
            processed_resized = processed
        
        orig_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        proc_gray = cv2.cvtColor(processed_resized, cv2.COLOR_BGR2GRAY)
        
        metrics["structural_similarity"] = ssim(orig_gray, proc_gray)
        
        # 3. 饱和度提升
        orig_hsv = cv2.cvtColor(original, cv2.COLOR_BGR2HSV)
        proc_hsv = cv2.cvtColor(processed_resized, cv2.COLOR_BGR2HSV)
        
        orig_saturation = np.mean(orig_hsv[:, :, 1])
        proc_saturation = np.mean(proc_hsv[:, :, 1])
        
        metrics["saturation_boost"] = proc_saturation / orig_saturation if orig_saturation > 0 else 1.0
        
        # 4. 对比度提升
        orig_contrast = np.std(orig_gray)
        proc_contrast = np.std(proc_gray)
        metrics["contrast_boost"] = proc_contrast / orig_contrast if orig_contrast > 0 else 1.0
        
        # 5. 文件大小
        metrics["file_size_mb"] = os.path.getsize(processed_path) / (1024 * 1024)
        
        return metrics
        
    except Exception as e:
        logger.error(f"图像质量分析失败: {str(e)}")
        return {"error": str(e)}

def final_comparison():
    """最终对比测试"""
    print("🏆 最终对比测试 - 三版本PK")
    print("=" * 80)
    
    test_image = "uploads/250625_205648.jpg"
    if not os.path.exists(test_image):
        print(f"❌ 测试图像不存在: {test_image}")
        return
    
    timestamp = int(time.time())
    results = {}
    
    # 1. 测试极简版本
    print("🔹 测试极简版本...")
    try:
        simple_generator = SimpleProfessionalGenerator()
        start_time = time.time()
        simple_prof, simple_comp, simple_time = simple_generator.generate_professional_image(
            test_image, f"final_simple_{timestamp}"
        )
        results["simple"] = {
            "success": True,
            "time": simple_time,
            "path": simple_prof,
            "metrics": analyze_image_quality(test_image, simple_prof)
        }
        print(f"✅ 极简版本成功，耗时: {simple_time:.2f}秒")
    except Exception as e:
        results["simple"] = {"success": False, "error": str(e)}
        print(f"❌ 极简版本失败: {str(e)}")
    
    # 2. 测试优化版本
    print("🔹 测试优化版本...")
    try:
        optimized_generator = OptimizedProfessionalGenerator()
        start_time = time.time()
        opt_prof, opt_comp, opt_time = optimized_generator.generate_professional_image(
            test_image, f"final_optimized_{timestamp}"
        )
        results["optimized"] = {
            "success": True,
            "time": opt_time,
            "path": opt_prof,
            "metrics": analyze_image_quality(test_image, opt_prof)
        }
        print(f"✅ 优化版本成功，耗时: {opt_time:.2f}秒")
    except Exception as e:
        results["optimized"] = {"success": False, "error": str(e)}
        print(f"❌ 优化版本失败: {str(e)}")
    
    # 3. 测试复杂版本（AI增强）
    print("🔹 测试复杂版本（AI增强）...")
    try:
        # 设置API密钥
        os.environ['QWEN_API_KEY'] = 'sk-ade7e6a1728741fcb009dcf1419000de'
        os.environ['AI_ENHANCED_MODE'] = 'true'
        
        complex_generator = ProfessionalWeavingGenerator()
        start_time = time.time()
        complex_prof, complex_comp, complex_time = complex_generator.generate_professional_image(
            test_image, f"final_complex_{timestamp}", color_count=8
        )
        results["complex"] = {
            "success": True,
            "time": complex_time,
            "path": complex_prof,
            "metrics": analyze_image_quality(test_image, complex_prof)
        }
        print(f"✅ 复杂版本成功，耗时: {complex_time:.2f}秒")
    except Exception as e:
        results["complex"] = {"success": False, "error": str(e)}
        print(f"❌ 复杂版本失败: {str(e)}")
    
    # 4. 分析对比结果
    print("\n📊 详细对比分析")
    print("=" * 80)
    
    # 表格头
    print(f"{'版本':<12} {'时间(秒)':<10} {'结构相似性':<12} {'颜色简化%':<12} {'饱和度':<10} {'对比度':<10} {'文件大小MB':<12}")
    print("-" * 80)
    
    # 数据行
    versions = [
        ("极简版本", "simple"),
        ("优化版本", "optimized"), 
        ("复杂版本", "complex")
    ]
    
    best_structure = 0
    best_version = ""
    
    for name, key in versions:
        if key in results and results[key]["success"]:
            r = results[key]
            metrics = r["metrics"]
            
            if "error" not in metrics:
                structure_sim = metrics["structural_similarity"]
                if structure_sim > best_structure:
                    best_structure = structure_sim
                    best_version = name
                
                print(f"{name:<12} {r['time']:<10.2f} {structure_sim:<12.3f} {metrics['color_reduction']:<12.1f} {metrics['saturation_boost']:<10.2f} {metrics['contrast_boost']:<10.2f} {metrics['file_size_mb']:<12.2f}")
            else:
                print(f"{name:<12} {r['time']:<10.2f} {'ERROR':<12} {'ERROR':<12} {'ERROR':<10} {'ERROR':<10} {'ERROR':<12}")
        else:
            print(f"{name:<12} {'FAILED':<10} {'FAILED':<12} {'FAILED':<12} {'FAILED':<10} {'FAILED':<10} {'FAILED':<12}")
    
    # 5. 最终评估和推荐
    print(f"\n🏆 最终评估")
    print("=" * 40)
    
    if best_version:
        print(f"🥇 结构保持最佳: {best_version} (SSIM: {best_structure:.3f})")
        
        # 速度对比
        successful_versions = [(name, key) for name, key in versions if key in results and results[key]["success"]]
        if len(successful_versions) > 1:
            fastest_time = min(results[key]["time"] for _, key in successful_versions)
            fastest_version = next(name for name, key in successful_versions if results[key]["time"] == fastest_time)
            print(f"🚀 速度最快: {fastest_version} ({fastest_time:.2f}秒)")
    
    # 6. 专业建议
    print(f"\n💡 专业建议:")
    
    if best_structure > 0.8:
        print(f"✅ {best_version}达到了优秀的结构保持性能（>0.8），推荐使用")
    elif best_structure > 0.6:
        print(f"⚠️  {best_version}结构保持性能良好（>0.6），可以使用但仍有改进空间")
    else:
        print("❌ 所有版本的结构保持性能都需要进一步改进")
    
    # 7. 使用场景推荐
    print(f"\n🎯 使用场景推荐:")
    for name, key in versions:
        if key in results and results[key]["success"]:
            r = results[key]
            metrics = r["metrics"]
            
            if "error" not in metrics:
                structure_sim = metrics["structural_similarity"]
                processing_time = r["time"]
                
                if structure_sim > 0.7 and processing_time < 20:
                    print(f"✅ {name}: 推荐用于生产环境（结构保持好且速度快）")
                elif structure_sim > 0.7:
                    print(f"⚡ {name}: 推荐用于质量要求高的场景（结构保持好但速度慢）")
                elif processing_time < 20:
                    print(f"🚀 {name}: 推荐用于快速处理场景（速度快但质量一般）")
                else:
                    print(f"⚠️  {name}: 需要进一步优化")
    
    # 8. 生成的文件路径
    print(f"\n📁 生成的文件:")
    for name, key in versions:
        if key in results and results[key]["success"]:
            print(f"   {name}: {results[key]['path']}")

if __name__ == "__main__":
    try:
        final_comparison()
    except Exception as e:
        print(f"❌ 最终对比测试失败: {str(e)}")
        logger.error(f"最终对比测试失败: {str(e)}", exc_info=True) 