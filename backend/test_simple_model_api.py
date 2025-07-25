"""
简化模型API测试脚本
验证简化模型API的功能和集成
"""

import asyncio
import cv2
import numpy as np
import time
import json
from pathlib import Path
from simple_model_api import SimpleModelAPIManager, SimpleModelAPIConfig, SimpleGenerationRequest


def create_test_image() -> np.ndarray:
    """创建测试图像"""
    # 创建一个简单的测试图像
    image = np.ones((512, 512, 3), dtype=np.uint8) * 240
    
    # 添加一些图案
    cv2.rectangle(image, (100, 100), (412, 412), (100, 100, 100), 2)
    cv2.circle(image, (256, 256), 80, (150, 150, 150), -1)
    cv2.putText(image, "Test", (220, 260), cv2.FONT_HERSHEY_SIMPLEX, 2, (50, 50, 50), 3)
    
    return image


def save_test_image(image: np.ndarray, filename: str):
    """保存测试图像"""
    test_dir = Path("test_images")
    test_dir.mkdir(exist_ok=True)
    
    filepath = test_dir / filename
    cv2.imwrite(str(filepath), image)
    return str(filepath)


async def test_simple_model_api():
    """测试简化模型API管理器"""
    print("🚀 开始测试简化模型API管理器...")
    
    # 1. 初始化API管理器
    print("\n1. 初始化API管理器...")
    config = SimpleModelAPIConfig(
        enable_optimization=True,
        enable_quality_analysis=True,
        default_style="basic"
    )
    
    api_manager = SimpleModelAPIManager(config)
    
    # 2. 检查可用模型
    print("\n2. 检查可用模型...")
    available_models = api_manager.get_available_models()
    print(f"可用模型: {json.dumps(available_models, indent=2, ensure_ascii=False)}")
    
    # 3. 创建测试图像
    print("\n3. 创建测试图像...")
    test_image = create_test_image()
    test_image_path = save_test_image(test_image, "test_input.png")
    print(f"测试图像已保存: {test_image_path}")
    
    # 4. 测试基础图像处理
    print("\n4. 测试基础图像处理...")
    request = SimpleGenerationRequest(
        image_path=test_image_path,
        style="basic",
        color_count=12,
        edge_enhancement=True,
        noise_reduction=True,
        optimization_level="balanced"
    )
    
    # 5. 提交生成任务
    print("\n5. 提交生成任务...")
    job_id = await api_manager.generate_embroidery(request)
    print(f"任务ID: {job_id}")
    
    # 6. 监控任务状态
    print("\n6. 监控任务状态...")
    max_wait_time = 30  # 最大等待30秒
    start_time = time.time()
    
    while time.time() - start_time < max_wait_time:
        status = api_manager.get_job_status(job_id)
        if status:
            print(f"任务状态: {status['status']}")
            if status['status'] == 'completed':
                print(f"✅ 任务完成！处理时间: {status['processing_time']:.2f}秒")
                print(f"输出文件: {status['output_files']}")
                if 'quality_metrics' in status and status['quality_metrics']:
                    print(f"质量指标: {json.dumps(status['quality_metrics'], indent=2, ensure_ascii=False)}")
                if 'optimization_params' in status and status['optimization_params']:
                    print(f"优化参数: {json.dumps(status['optimization_params'], indent=2, ensure_ascii=False)}")
                break
            elif status['status'] == 'failed':
                print(f"❌ 任务失败: {status['error_message']}")
                break
        else:
            print("等待任务开始...")
        
        await asyncio.sleep(1)
    else:
        print("⏰ 任务超时")
    
    # 7. 测试不同风格
    print("\n7. 测试不同风格...")
    styles = ["basic", "embroidery", "traditional", "modern"]
    
    for style in styles:
        print(f"\n测试 {style} 风格...")
        request.style = style
        
        job_id = await api_manager.generate_embroidery(request)
        print(f"任务ID: {job_id}")
        
        # 等待完成
        start_time = time.time()
        while time.time() - start_time < 20:
            status = api_manager.get_job_status(job_id)
            if status and status['status'] in ['completed', 'failed']:
                if status['status'] == 'completed':
                    print(f"✅ {style} 风格处理完成")
                else:
                    print(f"❌ {style} 风格处理失败: {status['error_message']}")
                break
            await asyncio.sleep(1)
    
    # 8. 测试不同优化级别
    print("\n8. 测试不同优化级别...")
    optimization_levels = ["conservative", "balanced", "aggressive"]
    
    for level in optimization_levels:
        print(f"\n测试 {level} 优化级别...")
        request.optimization_level = level
        request.style = "embroidery"  # 使用刺绣风格
        
        job_id = await api_manager.generate_embroidery(request)
        print(f"任务ID: {job_id}")
        
        # 等待完成
        start_time = time.time()
        while time.time() - start_time < 20:
            status = api_manager.get_job_status(job_id)
            if status and status['status'] in ['completed', 'failed']:
                if status['status'] == 'completed':
                    print(f"✅ {level} 优化完成")
                else:
                    print(f"❌ {level} 优化失败: {status['error_message']}")
                break
            await asyncio.sleep(1)
    
    # 9. 测试系统统计
    print("\n9. 测试系统统计...")
    stats = api_manager.get_system_stats()
    print(f"系统统计: {json.dumps(stats, indent=2, ensure_ascii=False)}")
    
    # 10. 测试质量分析（如果有参考图像）
    print("\n10. 测试质量分析...")
    # 创建一个参考图像
    reference_image = create_test_image()
    reference_image_path = save_test_image(reference_image, "test_reference.png")
    
    request.reference_image = reference_image_path
    request.style = "embroidery"
    request.optimization_level = "balanced"
    job_id = await api_manager.generate_embroidery(request)
    
    # 等待完成
    start_time = time.time()
    while time.time() - start_time < 20:
        status = api_manager.get_job_status(job_id)
        if status and status['status'] in ['completed', 'failed']:
            if status['status'] == 'completed' and 'quality_metrics' in status:
                print(f"✅ 质量分析完成: {json.dumps(status['quality_metrics'], indent=2, ensure_ascii=False)}")
            break
        await asyncio.sleep(1)
    
    print("\n🎉 简化模型API测试完成！")


async def test_api_functions():
    """测试API函数"""
    print("\n🔧 测试API函数...")
    
    # 测试获取可用模型
    from simple_model_api import simple_get_available_models_api, simple_get_system_stats_api
    
    models = simple_get_available_models_api()
    print(f"API - 可用模型: {json.dumps(models, indent=2, ensure_ascii=False)}")
    
    stats = simple_get_system_stats_api()
    print(f"API - 系统统计: {json.dumps(stats, indent=2, ensure_ascii=False)}")


def test_error_handling():
    """测试错误处理"""
    print("\n🛡️ 测试错误处理...")
    
    api_manager = SimpleModelAPIManager()
    
    # 测试无效图像路径
    request = SimpleGenerationRequest(
        image_path="nonexistent_image.png",
        style="basic"
    )
    
    try:
        # 这应该会失败
        loop = asyncio.get_event_loop()
        job_id = loop.run_until_complete(api_manager.generate_embroidery(request))
        
        # 等待失败
        time.sleep(2)
        status = api_manager.get_job_status(job_id)
        if status and status['status'] == 'failed':
            print(f"✅ 错误处理正常: {status['error_message']}")
        else:
            print("❌ 错误处理异常")
            
    except Exception as e:
        print(f"✅ 异常处理正常: {e}")


def test_image_processing_methods():
    """测试图像处理方法"""
    print("\n🖼️ 测试图像处理方法...")
    
    api_manager = SimpleModelAPIManager()
    
    # 创建测试图像
    test_image = create_test_image()
    
    # 测试颜色量化
    quantized = api_manager._quantize_colors(test_image, 8)
    print(f"✅ 颜色量化完成: {quantized.shape}")
    
    # 测试边缘增强
    enhanced = api_manager._enhance_edges(test_image)
    print(f"✅ 边缘增强完成: {enhanced.shape}")
    
    # 测试噪声减少
    denoised = api_manager._reduce_noise(test_image)
    print(f"✅ 噪声减少完成: {denoised.shape}")
    
    # 测试不同风格
    styles = ["embroidery", "traditional", "modern"]
    for style in styles:
        if style == "embroidery":
            styled = api_manager._apply_embroidery_style(test_image)
        elif style == "traditional":
            styled = api_manager._apply_traditional_style(test_image)
        elif style == "modern":
            styled = api_manager._apply_modern_style(test_image)
        
        print(f"✅ {style} 风格应用完成: {styled.shape}")
        
        # 保存风格化图像
        output_path = f"test_images/test_{style}_style.png"
        cv2.imwrite(output_path, styled)
        print(f"   保存到: {output_path}")


if __name__ == "__main__":
    print("🧪 简化模型API管理器测试套件")
    print("=" * 50)
    
    # 运行测试
    try:
        # 图像处理方法测试
        test_image_processing_methods()
        
        # 主测试
        asyncio.run(test_simple_model_api())
        
        # API函数测试
        asyncio.run(test_api_functions())
        
        # 错误处理测试
        test_error_handling()
        
        print("\n🎊 所有测试完成！")
        
    except Exception as e:
        print(f"\n❌ 测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc() 