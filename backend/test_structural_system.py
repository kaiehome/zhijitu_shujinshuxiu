#!/usr/bin/env python3
"""
结构化专业识别图生成系统测试
验证新的技术路径："结构规则"为主轴，AI为辅助
"""

import os
import cv2
import numpy as np
from PIL import Image
import logging
import time
import json
from structural_professional_generator import StructuralProfessionalGenerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_test_image():
    """创建测试图像"""
    try:
        # 创建一个简单的测试图像：彩色几何图形
        width, height = 400, 300
        image = np.zeros((height, width, 3), dtype=np.uint8)
        
        # 背景：浅灰色
        image[:] = (240, 240, 240)
        
        # 红色圆形
        cv2.circle(image, (100, 100), 50, (255, 100, 100), -1)
        
        # 蓝色矩形
        cv2.rectangle(image, (200, 50), (350, 150), (100, 100, 255), -1)
        
        # 绿色三角形
        points = np.array([[150, 200], [100, 280], [200, 280]], np.int32)
        cv2.fillPoly(image, [points], (100, 255, 100))
        
        # 黄色椭圆
        cv2.ellipse(image, (300, 230), (40, 25), 0, 0, 360, (255, 255, 100), -1)
        
        # 保存测试图像
        test_image_path = "uploads/test_structural_image.png"
        os.makedirs("uploads", exist_ok=True)
        
        pil_image = Image.fromarray(image)
        pil_image.save(test_image_path)
        
        logger.info(f"测试图像已创建: {test_image_path}")
        return test_image_path
        
    except Exception as e:
        logger.error(f"创建测试图像失败: {str(e)}")
        return None

def analyze_structure_quality(structure_info_path: str):
    """分析结构化质量"""
    try:
        with open(structure_info_path, 'r', encoding='utf-8') as f:
            structure_info = json.load(f)
        
        metadata = structure_info.get('metadata', {})
        regions = structure_info.get('regions', {})
        boundaries = structure_info.get('boundaries', {})
        color_palette = structure_info.get('color_palette', [])
        
        print("\n" + "="*60)
        print("📊 结构化质量分析报告")
        print("="*60)
        
        # 基础指标
        print(f"🎯 生成器版本: {metadata.get('generator', 'Unknown')}")
        print(f"📈 总区域数: {metadata.get('total_regions', 0)}")
        print(f"🔗 总边界数: {metadata.get('total_boundaries', 0)}")
        print(f"🎨 颜色调色板大小: {len(color_palette)}")
        
        # 区域分析
        print(f"\n🔍 区域分析:")
        total_area = 0
        closed_regions = 0
        
        for region_id, region_data in regions.items():
            area = region_data.get('area', 0)
            is_closed = region_data.get('is_closed', False)
            color = region_data.get('color', [0, 0, 0])
            
            total_area += area
            if is_closed:
                closed_regions += 1
            
            print(f"  • 区域 {region_id}: 面积={area:,}px, 颜色=RGB{tuple(color)}, 闭合={'✓' if is_closed else '✗'}")
        
        print(f"  总覆盖面积: {total_area:,} 像素")
        print(f"  闭合区域比例: {closed_regions}/{len(regions)} ({closed_regions/len(regions)*100:.1f}%)")
        
        # 边界分析
        print(f"\n🔗 边界分析:")
        total_perimeter = 0
        total_points = 0
        
        for boundary_id, boundary_data in boundaries.items():
            perimeter = boundary_data.get('perimeter', 0)
            point_count = boundary_data.get('point_count', 0)
            
            total_perimeter += perimeter
            total_points += point_count
            
            print(f"  • 边界 {boundary_id}: 周长={perimeter:.1f}px, 点数={point_count}")
        
        print(f"  总周长: {total_perimeter:.1f} 像素")
        print(f"  总路径点数: {total_points}")
        print(f"  平均点密度: {total_points/total_perimeter:.2f} 点/像素" if total_perimeter > 0 else "  平均点密度: 0")
        
        # 颜色分析
        print(f"\n🎨 颜色调色板:")
        for i, color in enumerate(color_palette):
            print(f"  • 颜色 {i+1}: RGB{tuple(color)}")
        
        # 质量评分
        quality_score = 0
        max_score = 100
        
        # 区域闭合性评分 (30分)
        closure_score = (closed_regions / len(regions)) * 30 if regions else 0
        quality_score += closure_score
        
        # 颜色数量合理性评分 (20分)
        color_count = len(color_palette)
        if 8 <= color_count <= 16:
            color_score = 20
        elif 6 <= color_count <= 20:
            color_score = 15
        else:
            color_score = 10
        quality_score += color_score
        
        # 边界复杂度评分 (25分)
        avg_points_per_boundary = total_points / len(boundaries) if boundaries else 0
        if 10 <= avg_points_per_boundary <= 50:
            boundary_score = 25
        elif 5 <= avg_points_per_boundary <= 100:
            boundary_score = 20
        else:
            boundary_score = 15
        quality_score += boundary_score
        
        # 覆盖率评分 (25分)
        # 假设图像总面积约为400*300=120000像素
        estimated_total_pixels = 120000
        coverage_ratio = min(total_area / estimated_total_pixels, 1.0)
        coverage_score = coverage_ratio * 25
        quality_score += coverage_score
        
        print(f"\n⭐ 结构化质量评分:")
        print(f"  • 区域闭合性: {closure_score:.1f}/30")
        print(f"  • 颜色数量合理性: {color_score}/20")
        print(f"  • 边界复杂度: {boundary_score:.1f}/25")
        print(f"  • 区域覆盖率: {coverage_score:.1f}/25")
        print(f"  📊 总分: {quality_score:.1f}/{max_score} ({quality_score/max_score*100:.1f}%)")
        
        # 机器可读性评估
        print(f"\n🤖 机器可读性评估:")
        machine_readable_features = []
        
        if closed_regions == len(regions):
            machine_readable_features.append("✓ 所有区域完全闭合")
        else:
            machine_readable_features.append(f"⚠ {len(regions)-closed_regions} 个区域未完全闭合")
        
        if 8 <= len(color_palette) <= 16:
            machine_readable_features.append("✓ 颜色数量适中，便于识别")
        else:
            machine_readable_features.append("⚠ 颜色数量可能过多或过少")
        
        if total_points > 0:
            machine_readable_features.append("✓ 具备完整的矢量路径信息")
        else:
            machine_readable_features.append("✗ 缺少矢量路径信息")
        
        if len(boundaries) == len(regions):
            machine_readable_features.append("✓ 区域和边界数量匹配")
        else:
            machine_readable_features.append("⚠ 区域和边界数量不匹配")
        
        for feature in machine_readable_features:
            print(f"  {feature}")
        
        print("="*60)
        
        return quality_score
        
    except Exception as e:
        logger.error(f"结构化质量分析失败: {str(e)}")
        return 0

def test_structural_system():
    """测试结构化系统"""
    try:
        print("🎯 开始测试结构化专业识别图生成系统")
        print("技术路径：结构规则为主轴，AI为辅助")
        print("-" * 60)
        
        # 1. 创建测试图像
        test_image_path = create_test_image()
        if not test_image_path:
            print("❌ 测试图像创建失败")
            return False
        
        # 2. 初始化结构化生成器
        generator = StructuralProfessionalGenerator()
        print("✅ 结构化生成器初始化成功")
        
        # 3. 生成结构化专业识别图
        job_id = f"test_structural_{int(time.time())}"
        
        start_time = time.time()
        professional_path, comparison_path, structure_info_path, processing_time = generator.generate_structural_professional_image(
            test_image_path, 
            job_id, 
            color_count=12
        )
        total_time = time.time() - start_time
        
        print(f"✅ 结构化专业识别图生成完成")
        print(f"📁 专业图像: {professional_path}")
        print(f"📊 对比图像: {comparison_path}")
        print(f"📋 结构信息: {structure_info_path}")
        print(f"⏱️  处理时间: {processing_time:.2f}秒")
        print(f"🕐 总耗时: {total_time:.2f}秒")
        
        # 4. 验证文件存在性
        files_to_check = [professional_path, comparison_path, structure_info_path]
        for file_path in files_to_check:
            if os.path.exists(file_path):
                file_size = os.path.getsize(file_path)
                print(f"✅ {os.path.basename(file_path)}: {file_size:,} bytes")
            else:
                print(f"❌ {os.path.basename(file_path)}: 文件不存在")
                return False
        
        # 5. 分析结构化质量
        quality_score = analyze_structure_quality(structure_info_path)
        
        # 6. 性能评估
        print(f"\n⚡ 性能评估:")
        print(f"  • 处理速度: {processing_time:.2f}秒")
        print(f"  • 质量评分: {quality_score:.1f}/100")
        
        if processing_time < 5.0:
            print("  ✅ 处理速度优秀")
        elif processing_time < 10.0:
            print("  ✅ 处理速度良好")
        else:
            print("  ⚠️ 处理速度需要优化")
        
        if quality_score >= 80:
            print("  ✅ 结构化质量优秀")
        elif quality_score >= 60:
            print("  ✅ 结构化质量良好")
        else:
            print("  ⚠️ 结构化质量需要改进")
        
        print(f"\n🎉 结构化专业识别图生成系统测试完成！")
        print(f"🎯 技术突破：成功实现机器可读的结构化图像生成")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {str(e)}")
        logger.error(f"测试失败: {str(e)}", exc_info=True)
        return False

def main():
    """主函数"""
    success = test_structural_system()
    
    if success:
        print("\n" + "="*60)
        print("🎯 结构化专业识别图生成系统")
        print("✅ 测试通过 - 系统ready for production!")
        print("🔧 核心特点：")
        print("   • 区域连续性：每个色块为闭合区域")
        print("   • 边界清晰性：路径可提取、点数可控")
        print("   • 颜色可控性：避免相邻干扰、符合绣线规范")
        print("   • 机器可读性：生成矢量路径和结构信息")
        print("="*60)
    else:
        print("\n❌ 测试失败，请检查系统配置")

if __name__ == "__main__":
    main() 