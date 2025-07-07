#!/usr/bin/env python3
"""
算法对比分析工具
比较极简版本和复杂版本的效果
"""

import os
import cv2
import numpy as np
from PIL import Image
import logging
import time
from simple_professional_generator import SimpleProfessionalGenerator
from professional_weaving_generator import ProfessionalWeavingGenerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def analyze_image_quality(original_path: str, processed_path: str) -> dict:
    """分析图像质量指标"""
    try:
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
        
        # 2. 边缘密度
        orig_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        proc_gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
        
        orig_edges = cv2.Canny(orig_gray, 50, 150)
        proc_edges = cv2.Canny(proc_gray, 50, 150)
        
        orig_edge_density = np.sum(orig_edges > 0) / orig_edges.size
        proc_edge_density = np.sum(proc_edges > 0) / proc_edges.size
        
        metrics["edge_enhancement"] = proc_edge_density / orig_edge_density if orig_edge_density > 0 else 1.0
        
        # 3. 饱和度
        orig_hsv = cv2.cvtColor(original, cv2.COLOR_BGR2HSV)
        proc_hsv = cv2.cvtColor(processed, cv2.COLOR_BGR2HSV)
        
        orig_saturation = np.mean(orig_hsv[:, :, 1])
        proc_saturation = np.mean(proc_hsv[:, :, 1])
        
        metrics["saturation_boost"] = proc_saturation / orig_saturation if orig_saturation > 0 else 1.0
        
        # 4. 对比度
        metrics["original_contrast"] = np.std(orig_gray)
        metrics["processed_contrast"] = np.std(proc_gray)
        metrics["contrast_boost"] = metrics["processed_contrast"] / metrics["original_contrast"] if metrics["original_contrast"] > 0 else 1.0
        
        # 5. 结构相似性（SSIM）- 衡量主体是否保持可识别
        from skimage.metrics import structural_similarity as ssim
        
        # 调整尺寸以匹配
        if original.shape != processed.shape:
            processed_resized = cv2.resize(processed, (original.shape[1], original.shape[0]))
        else:
            processed_resized = processed
        
        orig_gray_resized = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        proc_gray_resized = cv2.cvtColor(processed_resized, cv2.COLOR_BGR2GRAY)
        
        metrics["structural_similarity"] = ssim(orig_gray_resized, proc_gray_resized)
        
        # 6. 文件大小
        metrics["file_size_mb"] = os.path.getsize(processed_path) / (1024 * 1024)
        
        return metrics
        
    except Exception as e:
        logger.error(f"图像质量分析失败: {str(e)}")
        return {"error": str(e)}

def compare_algorithms():
    """比较两种算法"""
    print("🔬 算法对比分析")
    print("=" * 60)
    
    # 测试图像
    test_image = "uploads/250625_205648.jpg"
    if not os.path.exists(test_image):
        print(f"❌ 测试图像不存在: {test_image}")
        return
    
    timestamp = int(time.time())
    
    # 1. 测试极简版本
    print("📊 测试极简版本...")
    simple_generator = SimpleProfessionalGenerator()
    
    simple_start = time.time()
    try:
        simple_prof, simple_comp, simple_time = simple_generator.generate_professional_image(
            test_image, f"simple_compare_{timestamp}"
        )
        simple_success = True
        print(f"✅ 极简版本成功，耗时: {simple_time:.2f}秒")
    except Exception as e:
        simple_success = False
        simple_time = time.time() - simple_start
        print(f"❌ 极简版本失败: {str(e)}")
    
    # 2. 测试复杂版本
    print("📊 测试复杂版本...")
    
    # 设置API密钥
    os.environ['QWEN_API_KEY'] = 'sk-ade7e6a1728741fcb009dcf1419000de'
    os.environ['AI_ENHANCED_MODE'] = 'true'
    
    complex_generator = ProfessionalWeavingGenerator()
    
    complex_start = time.time()
    try:
        complex_prof, complex_comp, complex_time = complex_generator.generate_professional_image(
            test_image, f"complex_compare_{timestamp}", color_count=8
        )
        complex_success = True
        print(f"✅ 复杂版本成功，耗时: {complex_time:.2f}秒")
    except Exception as e:
        complex_success = False
        complex_time = time.time() - complex_start
        print(f"❌ 复杂版本失败: {str(e)}")
    
    # 3. 分析对比
    print("\n📈 质量对比分析")
    print("-" * 40)
    
    if simple_success:
        print("🔹 极简版本分析:")
        simple_metrics = analyze_image_quality(test_image, simple_prof)
        if "error" not in simple_metrics:
            print(f"   颜色简化率: {simple_metrics['color_reduction']:.1f}%")
            print(f"   边缘增强倍数: {simple_metrics['edge_enhancement']:.2f}x")
            print(f"   饱和度提升: {simple_metrics['saturation_boost']:.2f}x")
            print(f"   对比度提升: {simple_metrics['contrast_boost']:.2f}x")
            print(f"   结构相似性: {simple_metrics['structural_similarity']:.3f} (越接近1越好)")
            print(f"   文件大小: {simple_metrics['file_size_mb']:.2f} MB")
    
    if complex_success:
        print("\n🔹 复杂版本分析:")
        complex_metrics = analyze_image_quality(test_image, complex_prof)
        if "error" not in complex_metrics:
            print(f"   颜色简化率: {complex_metrics['color_reduction']:.1f}%")
            print(f"   边缘增强倍数: {complex_metrics['edge_enhancement']:.2f}x")
            print(f"   饱和度提升: {complex_metrics['saturation_boost']:.2f}x")
            print(f"   对比度提升: {complex_metrics['contrast_boost']:.2f}x")
            print(f"   结构相似性: {complex_metrics['structural_similarity']:.3f} (越接近1越好)")
            print(f"   文件大小: {complex_metrics['file_size_mb']:.2f} MB")
    
    # 4. 综合评估
    print("\n🏆 综合评估")
    print("-" * 40)
    
    if simple_success and complex_success:
        print("⏱️  处理速度:")
        print(f"   极简版本: {simple_time:.2f}秒")
        print(f"   复杂版本: {complex_time:.2f}秒")
        print(f"   速度提升: {complex_time/simple_time:.1f}倍")
        
        if "error" not in simple_metrics and "error" not in complex_metrics:
            print("\n🎯 关键指标对比:")
            
            # 结构保持性（最重要）
            print(f"   结构保持性 (SSIM):")
            print(f"   - 极简版本: {simple_metrics['structural_similarity']:.3f}")
            print(f"   - 复杂版本: {complex_metrics['structural_similarity']:.3f}")
            
            if simple_metrics['structural_similarity'] > complex_metrics['structural_similarity']:
                print("   ✅ 极简版本更好地保持了原图结构")
            else:
                print("   ✅ 复杂版本更好地保持了原图结构")
            
            # 效果增强
            print(f"\n   效果增强:")
            print(f"   - 颜色简化: 极简{simple_metrics['color_reduction']:.1f}% vs 复杂{complex_metrics['color_reduction']:.1f}%")
            print(f"   - 饱和度提升: 极简{simple_metrics['saturation_boost']:.2f}x vs 复杂{complex_metrics['saturation_boost']:.2f}x")
            print(f"   - 边缘增强: 极简{simple_metrics['edge_enhancement']:.2f}x vs 复杂{complex_metrics['edge_enhancement']:.2f}x")
    
    print("\n💡 建议:")
    if simple_success and complex_success and "error" not in simple_metrics and "error" not in complex_metrics:
        if simple_metrics['structural_similarity'] > 0.8:
            print("✅ 极简版本保持了良好的结构相似性，推荐使用")
        elif complex_metrics['structural_similarity'] > 0.8:
            print("✅ 复杂版本保持了良好的结构相似性，但处理时间较长")
        else:
            print("⚠️  两个版本的结构相似性都需要改进")
    
    print(f"\n📁 生成的文件:")
    if simple_success:
        print(f"   极简版本: {simple_prof}")
    if complex_success:
        print(f"   复杂版本: {complex_prof}")

if __name__ == "__main__":
    try:
        # 安装必要的依赖
        import subprocess
        subprocess.run(["pip", "install", "scikit-image"], check=False, capture_output=True)
        
        compare_algorithms()
    except Exception as e:
        print(f"❌ 对比分析失败: {str(e)}")
        logger.error(f"对比分析失败: {str(e)}", exc_info=True) 