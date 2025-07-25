"""
模型API使用演示
展示如何使用可用的模型API组件
"""

import asyncio
import cv2
import numpy as np
import time
from pathlib import Path

# 导入可用的模型API组件
from simple_model_api import SimpleModelAPIManager, SimpleModelAPIConfig, SimpleGenerationRequest
from ai_enhanced_processor import AIEnhancedProcessor
from ai_image_generator import AIImageGenerator
from ai_segmentation import AISegmenter


def create_demo_image() -> np.ndarray:
    """创建演示图像"""
    # 创建一个简单的熊猫图案
    image = np.ones((512, 512, 3), dtype=np.uint8) * 240
    
    # 熊猫头部
    cv2.ellipse(image, (256, 256), (120, 100), 0, 0, 360, (240, 240, 240), -1)
    
    # 眼睛
    cv2.circle(image, (220, 230), 15, (50, 50, 50), -1)
    cv2.circle(image, (292, 230), 15, (50, 50, 50), -1)
    
    # 鼻子
    cv2.circle(image, (256, 270), 8, (50, 50, 50), -1)
    
    # 添加一些细节
    cv2.putText(image, "Demo", (220, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 100, 100), 2)
    
    return image


async def demo_simple_model_api():
    """演示简化模型API"""
    print("🎨 演示简化模型API...")
    
    # 创建输出目录
    output_dir = Path("demo_outputs")
    output_dir.mkdir(exist_ok=True)
    
    # 创建演示图像
    demo_image = create_demo_image()
    input_path = output_dir / "demo_input.png"
    cv2.imwrite(str(input_path), demo_image)
    
    # 初始化API管理器
    config = SimpleModelAPIConfig(
        enable_optimization=True,
        enable_quality_analysis=True,
        default_style="embroidery"
    )
    api_manager = SimpleModelAPIManager(config)
    
    # 演示不同风格
    styles = ["basic", "embroidery", "traditional", "modern"]
    
    for style in styles:
        print(f"\n🔄 处理 {style} 风格...")
        
        request = SimpleGenerationRequest(
            image_path=str(input_path),
            style=style,
            color_count=16,
            edge_enhancement=True,
            noise_reduction=True,
            optimization_level="balanced"
        )
        
        # 提交任务
        job_id = await api_manager.generate_embroidery(request)
        print(f"任务ID: {job_id}")
        
        # 等待完成
        max_wait = 30
        while max_wait > 0:
            status = api_manager.get_job_status(job_id)
            if status and status['status'] in ['completed', 'failed']:
                break
            await asyncio.sleep(1)
            max_wait -= 1
        
        if status and status['status'] == 'completed':
            print(f"✅ {style} 风格处理完成")
            print(f"   处理时间: {status['processing_time']:.2f}秒")
            print(f"   输出文件: {status['output_files']}")
        else:
            print(f"❌ {style} 风格处理失败")
    
    # 显示系统统计
    stats = api_manager.get_system_stats()
    print(f"\n📊 系统统计: {stats}")


def demo_ai_enhanced_processor():
    """演示AI增强处理器"""
    print("\n🤖 演示AI增强处理器...")
    
    # 创建演示图像
    demo_image = create_demo_image()
    
    # 初始化处理器
    processor = AIEnhancedProcessor()
    
    # 分析图像内容
    print("🔍 分析图像内容...")
    analysis = processor.analyze_image_content(demo_image)
    print(f"分析结果: {analysis}")
    
    # 回退分析
    print("🔄 执行回退分析...")
    fallback_analysis = processor._fallback_analysis(demo_image)
    print(f"回退分析结果: {fallback_analysis}")


def demo_ai_image_generator():
    """演示AI图像生成器"""
    print("\n🎨 演示AI图像生成器...")
    
    # 初始化生成器
    generator = AIImageGenerator()
    
    # 生成专业背景
    print("🖼️ 生成专业背景...")
    background = generator.generate_professional_background(512, 512, "熊猫刺绣")
    
    # 保存背景
    output_dir = Path("demo_outputs")
    output_dir.mkdir(exist_ok=True)
    cv2.imwrite(str(output_dir / "generated_background.png"), background)
    print(f"✅ 背景已保存: {output_dir}/generated_background.png")
    
    # 生成基础背景
    print("🖼️ 生成基础背景...")
    basic_bg = generator._generate_basic_background(256, 256)
    cv2.imwrite(str(output_dir / "basic_background.png"), basic_bg)
    print(f"✅ 基础背景已保存: {output_dir}/basic_background.png")


def demo_ai_segmentation():
    """演示AI分割器"""
    print("\n✂️ 演示AI分割器...")
    
    # 创建演示图像
    demo_image = create_demo_image()
    
    # 测试不同分割方法
    methods = ['grabcut', 'watershed', 'slic', 'contour']
    
    output_dir = Path("demo_outputs")
    output_dir.mkdir(exist_ok=True)
    
    for method in methods:
        print(f"🔍 使用 {method} 方法分割...")
        
        try:
            # 初始化分割器
            segmenter = AISegmenter(model_name=method)
            
            # 执行分割
            result = segmenter.segment(demo_image)
            
            # 保存结果
            output_path = output_dir / f"segmentation_{method}.png"
            cv2.imwrite(str(output_path), result)
            print(f"✅ {method} 分割完成，已保存: {output_path}")
            
        except Exception as e:
            print(f"❌ {method} 分割失败: {e}")


def demo_integration():
    """演示组件集成"""
    print("\n🔗 演示组件集成...")
    
    # 创建演示图像
    demo_image = create_demo_image()
    
    # 1. 使用AI增强处理器分析
    processor = AIEnhancedProcessor()
    analysis = processor.analyze_image_content(demo_image)
    print(f"📊 图像分析: {analysis}")
    
    # 2. 使用AI分割器分割
    segmenter = AISegmenter(model_name='grabcut')
    segmentation = segmenter.segment(demo_image)
    print(f"✂️ 图像分割完成，形状: {segmentation.shape}")
    
    # 3. 使用AI图像生成器生成背景
    generator = AIImageGenerator()
    background = generator.generate_professional_background(512, 512, "熊猫")
    print(f"🎨 背景生成完成，形状: {background.shape}")
    
    # 4. 保存集成结果
    output_dir = Path("demo_outputs")
    output_dir.mkdir(exist_ok=True)
    
    cv2.imwrite(str(output_dir / "integration_segmentation.png"), segmentation)
    cv2.imwrite(str(output_dir / "integration_background.png"), background)
    
    print("✅ 集成演示完成，结果已保存")


async def main():
    """主演示函数"""
    print("🚀 模型API使用演示")
    print("=" * 50)
    
    # 创建演示输出目录
    output_dir = Path("demo_outputs")
    output_dir.mkdir(exist_ok=True)
    
    # 运行各个演示
    await demo_simple_model_api()
    demo_ai_enhanced_processor()
    demo_ai_image_generator()
    demo_ai_segmentation()
    demo_integration()
    
    print("\n🎉 演示完成！")
    print(f"📁 所有输出文件保存在: {output_dir.absolute()}")
    print("\n💡 使用建议:")
    print("1. 简化模型API: 用于图像处理和风格转换")
    print("2. AI增强处理器: 用于图像内容分析")
    print("3. AI图像生成器: 用于背景生成")
    print("4. AI分割器: 用于图像分割")
    print("5. 组件集成: 组合使用以获得最佳效果")


if __name__ == "__main__":
    asyncio.run(main()) 