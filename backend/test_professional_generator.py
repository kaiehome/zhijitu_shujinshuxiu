#!/usr/bin/env python3
"""
专业织机识别图像生成器测试脚本
"""

import sys
import os
import logging
from pathlib import Path

# 添加backend目录到路径
backend_dir = Path(__file__).parent
sys.path.append(str(backend_dir))

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_professional_generator():
    """测试专业织机生成器"""
    try:
        # 导入专业织机生成器
        from professional_weaving_generator import ProfessionalWeavingGenerator
        
        logger.info("🧪 开始测试专业织机识别图像生成器")
        
        # 初始化生成器
        generator = ProfessionalWeavingGenerator()
        logger.info("✅ 专业织机生成器初始化成功")
        
        # 检查输出目录
        outputs_dir = Path("outputs")
        if not outputs_dir.exists():
            outputs_dir.mkdir(parents=True)
            logger.info(f"✅ 创建输出目录: {outputs_dir}")
        
        # 检查是否有测试图像
        test_images = []
        uploads_dir = Path("uploads")
        
        if uploads_dir.exists():
            for ext in ['.jpg', '.jpeg', '.png']:
                test_images.extend(uploads_dir.glob(f'*{ext}'))
        
        if test_images:
            test_image = test_images[0]
            logger.info(f"📸 找到测试图像: {test_image}")
            
            # 生成测试任务ID
            import time
            job_id = f"test_{int(time.time())}"
            
            logger.info(f"🚀 开始生成专业织机图像，任务ID: {job_id}")
            
            # 生成专业织机图像
            professional_path, comparison_path, processing_time = generator.generate_professional_image(
                input_path=str(test_image),
                job_id=job_id,
                color_count=16
            )
            
            logger.info(f"🎯 专业织机图像生成完成！")
            logger.info(f"   📁 专业图像: {professional_path}")
            logger.info(f"   📊 对比图像: {comparison_path}")
            logger.info(f"   ⏱️  处理耗时: {processing_time:.2f}秒")
            
            # 验证输出文件
            if Path(professional_path).exists():
                file_size = Path(professional_path).stat().st_size
                logger.info(f"✅ 专业图像文件验证通过 ({file_size/1024/1024:.2f} MB)")
            else:
                logger.error("❌ 专业图像文件未生成")
                
            if Path(comparison_path).exists():
                file_size = Path(comparison_path).stat().st_size
                logger.info(f"✅ 对比图像文件验证通过 ({file_size/1024/1024:.2f} MB)")
            else:
                logger.error("❌ 对比图像文件未生成")
            
        else:
            logger.warning("⚠️  未找到测试图像，跳过图像生成测试")
            logger.info("📝 请将测试图像放到 uploads/ 目录下")
        
        logger.info("🎉 专业织机生成器测试完成！")
        
    except ImportError as e:
        logger.error(f"❌ 导入错误: {e}")
        logger.info("💡 请确保所有依赖包已正确安装")
        
    except Exception as e:
        logger.error(f"❌ 测试失败: {e}")
        import traceback
        logger.error(traceback.format_exc())


def test_image_processor_integration():
    """测试图像处理器集成"""
    try:
        from image_processor import SichuanBrocadeProcessor
        
        logger.info("🔧 测试图像处理器集成")
        
        # 初始化处理器
        processor = SichuanBrocadeProcessor()
        logger.info("✅ 图像处理器初始化成功")
        
        # 检查专业生成器是否已集成
        if hasattr(processor, 'professional_generator'):
            logger.info("✅ 专业织机生成器已成功集成到图像处理器")
            
            # 检查专业处理方法是否存在
            if hasattr(processor, 'process_image_professional'):
                logger.info("✅ 专业织机处理方法已添加")
            else:
                logger.error("❌ 专业织机处理方法未找到")
        else:
            logger.error("❌ 专业织机生成器未集成到图像处理器")
            
    except Exception as e:
        logger.error(f"❌ 集成测试失败: {e}")


if __name__ == "__main__":
    logger.info("🚀 开始专业织机识别图像生成器全面测试")
    
    print("\n" + "="*60)
    print("🏭 专业织机识别图像生成器测试")
    print("="*60)
    
    # 测试1: 专业生成器基本功能
    print("\n📋 测试1: 专业生成器基本功能")
    test_professional_generator()
    
    # 测试2: 图像处理器集成
    print("\n📋 测试2: 图像处理器集成")
    test_image_processor_integration()
    
    print("\n" + "="*60)
    print("🎯 测试完成！")
    print("="*60)
    
    logger.info("✨ 全面测试完成") 