#!/usr/bin/env python3
"""
专业织机识别软件效果对比测试
测试我们的AI增强算法是否能达到专业软件的效果
"""

import os
import sys
import time
import logging
from datetime import datetime
from PIL import Image
import numpy as np
import cv2

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from professional_weaving_generator import ProfessionalWeavingGenerator

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_professional_vs_ai():
    """测试专业软件 vs AI增强算法效果对比"""
    print("🎯 专业织机识别软件效果对比测试")
    print("=" * 60)
    
    # 设置API密钥
    os.environ['QWEN_API_KEY'] = 'sk-ade7e6a1728741fcb009dcf1419000de'
    os.environ['AI_ENHANCED_MODE'] = 'true'
    
    # 测试图像路径
    test_image = "uploads/250625_205648.jpg"  # 原始熊猫图像
    
    if not os.path.exists(test_image):
        print(f"❌ 测试图像不存在: {test_image}")
        return
    
    print(f"📸 使用测试图像: {test_image}")
    
    # 初始化生成器
    generator = ProfessionalWeavingGenerator()
    
    # 生成任务ID
    timestamp = int(time.time())
    job_id = f"vs_professional_{timestamp}"
    
    print(f"🚀 开始生成，任务ID: {job_id}")
    print("🎨 目标：完全模拟专业织机识别软件效果")
    print("🤖 使用：AI增强算法 + 通义千问大模型")
    print("-" * 60)
    
    start_time = time.time()
    
    try:
        # 生成专业织机识别图像
        professional_path, comparison_path, processing_time = generator.generate_professional_image(
            input_path=test_image,
            job_id=job_id,
            color_count=16  # 使用16色，符合专业软件标准
        )
        
        total_time = time.time() - start_time
        
        print("=" * 60)
        print("🎉 生成完成！")
        print(f"📁 专业图像: {professional_path}")
        print(f"📊 对比图像: {comparison_path}")
        print(f"⏱️  处理时间: {processing_time:.2f}秒")
        print(f"🕐 总耗时: {total_time:.2f}秒")
        
        # 分析生成的图像
        analyze_result(professional_path, test_image)
        
    except Exception as e:
        print(f"❌ 生成失败: {str(e)}")
        logger.error(f"生成失败: {str(e)}", exc_info=True)

def analyze_result(professional_path: str, original_path: str):
    """分析生成结果"""
    print("\n📊 图像分析报告")
    print("-" * 40)
    
    try:
        # 加载图像
        with Image.open(professional_path) as prof_img:
            prof_array = np.array(prof_img)
        
        with Image.open(original_path) as orig_img:
            orig_array = np.array(orig_img)
        
        # 分析颜色数量
        orig_colors = len(np.unique(orig_array.reshape(-1, 3), axis=0))
        prof_colors = len(np.unique(prof_array.reshape(-1, 3), axis=0))
        
        print(f"🎨 原图颜色数量: {orig_colors}")
        print(f"🎨 处理后颜色数量: {prof_colors}")
        print(f"📉 颜色简化率: {(1 - prof_colors/orig_colors)*100:.1f}%")
        
        # 分析边缘锐度
        orig_edges = cv2.Canny(cv2.cvtColor(orig_array, cv2.COLOR_RGB2GRAY), 50, 150)
        prof_edges = cv2.Canny(cv2.cvtColor(prof_array, cv2.COLOR_RGB2GRAY), 50, 150)
        
        orig_edge_density = np.sum(orig_edges > 0) / orig_edges.size
        prof_edge_density = np.sum(prof_edges > 0) / prof_edges.size
        
        print(f"🔍 原图边缘密度: {orig_edge_density:.4f}")
        print(f"🔍 处理后边缘密度: {prof_edge_density:.4f}")
        print(f"📈 边缘锐化倍数: {prof_edge_density/orig_edge_density:.2f}x")
        
        # 分析饱和度
        orig_hsv = cv2.cvtColor(orig_array, cv2.COLOR_RGB2HSV)
        prof_hsv = cv2.cvtColor(prof_array, cv2.COLOR_RGB2HSV)
        
        orig_saturation = np.mean(orig_hsv[:, :, 1])
        prof_saturation = np.mean(prof_hsv[:, :, 1])
        
        print(f"🌈 原图平均饱和度: {orig_saturation:.1f}")
        print(f"🌈 处理后平均饱和度: {prof_saturation:.1f}")
        print(f"📊 饱和度提升: {(prof_saturation/orig_saturation):.2f}x")
        
        # 文件大小
        orig_size = os.path.getsize(original_path) / (1024 * 1024)
        prof_size = os.path.getsize(professional_path) / (1024 * 1024)
        
        print(f"📦 原图文件大小: {orig_size:.2f} MB")
        print(f"📦 处理后文件大小: {prof_size:.2f} MB")
        
        # 专业效果评分
        score = calculate_professional_score(prof_array)
        print(f"\n🏆 专业效果评分: {score:.1f}/10")
        
        if score >= 8.0:
            print("✅ 优秀！已达到专业软件水平")
        elif score >= 6.0:
            print("🟡 良好！接近专业软件效果")
        else:
            print("🔴 需要改进！与专业软件差距较大")
            
    except Exception as e:
        print(f"❌ 分析失败: {str(e)}")

def calculate_professional_score(image: np.ndarray) -> float:
    """计算专业效果评分"""
    try:
        score = 0.0
        
        # 1. 颜色简化度 (0-2分)
        unique_colors = len(np.unique(image.reshape(-1, 3), axis=0))
        if unique_colors <= 50:
            score += 2.0
        elif unique_colors <= 100:
            score += 1.5
        elif unique_colors <= 200:
            score += 1.0
        else:
            score += 0.5
        
        # 2. 边缘锐度 (0-2分)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        if edge_density >= 0.05:
            score += 2.0
        elif edge_density >= 0.03:
            score += 1.5
        elif edge_density >= 0.02:
            score += 1.0
        else:
            score += 0.5
        
        # 3. 饱和度 (0-2分)
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        avg_saturation = np.mean(hsv[:, :, 1])
        
        if avg_saturation >= 150:
            score += 2.0
        elif avg_saturation >= 120:
            score += 1.5
        elif avg_saturation >= 100:
            score += 1.0
        else:
            score += 0.5
        
        # 4. 对比度 (0-2分)
        contrast = np.std(gray)
        
        if contrast >= 80:
            score += 2.0
        elif contrast >= 60:
            score += 1.5
        elif contrast >= 40:
            score += 1.0
        else:
            score += 0.5
        
        # 5. 平面化程度 (0-2分)
        # 计算颜色变化的平滑度
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        smoothness = 1.0 / (1.0 + np.mean(gradient_magnitude))
        
        if smoothness >= 0.8:
            score += 2.0
        elif smoothness >= 0.6:
            score += 1.5
        elif smoothness >= 0.4:
            score += 1.0
        else:
            score += 0.5
        
        return min(10.0, score)
        
    except Exception as e:
        logger.warning(f"评分计算失败: {str(e)}")
        return 5.0

if __name__ == "__main__":
    test_professional_vs_ai() 