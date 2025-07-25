#!/usr/bin/env python3
"""
AI模型API测试脚本
测试训练好的AI大模型生成织机识别图
"""

import requests
import json
import time
import os
from pathlib import Path

def test_ai_model_status():
    """测试AI模型状态"""
    print("=" * 60)
    print("测试AI模型状态")
    print("=" * 60)
    
    try:
        response = requests.get("http://localhost:8000/api/ai-model-status")
        
        if response.status_code == 200:
            model_info = response.json()
            print("✓ AI模型状态获取成功")
            print(f"模型加载状态: {model_info.get('is_loaded', False)}")
            print(f"模型目录: {model_info.get('model_dir', 'N/A')}")
            print(f"模型版本: {model_info.get('model_epoch', 'N/A')}")
            print(f"使用设备: {model_info.get('device_used', 'N/A')}")
            
            if model_info.get('is_loaded'):
                print(f"生成器参数数量: {model_info.get('generator_params', 'N/A')}")
                print(f"判别器参数数量: {model_info.get('discriminator_params', 'N/A')}")
                return True
            else:
                print("✗ AI模型未加载")
                return False
        else:
            print(f"✗ 状态获取失败: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"✗ 连接失败: {e}")
        return False

def test_ai_model_generation(input_image_path):
    """测试AI模型生成"""
    print("\n" + "=" * 60)
    print("测试AI模型生成")
    print("=" * 60)
    
    if not os.path.exists(input_image_path):
        print(f"✗ 输入图片不存在: {input_image_path}")
        return False
    
    try:
        # 上传图片并生成
        with open(input_image_path, 'rb') as f:
            files = {'file': (os.path.basename(input_image_path), f, 'image/jpeg')}
            
            print(f"上传图片: {input_image_path}")
            response = requests.post("http://localhost:8000/api/generate-ai-model", files=files)
        
        if response.status_code == 200:
            # 保存生成的图片
            timestamp = int(time.time())
            output_filename = f"ai_model_generated_{timestamp}.jpg"
            
            with open(output_filename, 'wb') as f:
                f.write(response.content)
            
            print(f"✓ AI模型生成成功")
            print(f"输出文件: {output_filename}")
            print(f"文件大小: {len(response.content) / 1024:.1f} KB")
            return True
            
        elif response.status_code == 503:
            print("✗ AI模型未加载，请先训练模型")
            return False
        else:
            print(f"✗ 生成失败: {response.status_code}")
            print(f"错误信息: {response.text}")
            return False
            
    except Exception as e:
        print(f"✗ 生成过程出错: {e}")
        return False

def test_all_generation_methods(input_image_path):
    """测试所有生成方法"""
    print("\n" + "=" * 60)
    print("测试所有生成方法对比")
    print("=" * 60)
    
    if not os.path.exists(input_image_path):
        print(f"✗ 输入图片不存在: {input_image_path}")
        return
    
    endpoints = [
        {
            "name": "传统专业生成",
            "url": "http://localhost:8000/api/upload"
        },
        {
            "name": "专业识别图生成",
            "url": "http://localhost:8000/api/generate-professional-recognition"
        },
        {
            "name": "高级专业识别图生成", 
            "url": "http://localhost:8000/api/generate-advanced-professional"
        },
        {
            "name": "终极专业识别图生成",
            "url": "http://localhost:8000/api/generate-ultimate-professional"
        },
        {
            "name": "本地AI增强专业识别图生成",
            "url": "http://localhost:8000/api/generate-local-ai-enhanced"
        },
        {
            "name": "AI大模型生成",
            "url": "http://localhost:8000/api/generate-ai-model"
        }
    ]
    
    results = []
    
    for endpoint in endpoints:
        print(f"\n测试: {endpoint['name']}")
        
        try:
            with open(input_image_path, 'rb') as f:
                files = {'file': (os.path.basename(input_image_path), f, 'image/jpeg')}
                
                response = requests.post(endpoint['url'], files=files)
            
            if response.status_code == 200:
                # 保存生成的图片
                timestamp = int(time.time())
                safe_name = endpoint['name'].replace(' ', '_').replace('生成', '')
                output_filename = f"{safe_name}_{timestamp}.jpg"
                
                with open(output_filename, 'wb') as f:
                    f.write(response.content)
                
                file_size = len(response.content) / 1024
                print(f"  ✓ 成功 - 文件: {output_filename} ({file_size:.1f} KB)")
                
                results.append({
                    "method": endpoint['name'],
                    "status": "success",
                    "file": output_filename,
                    "size_kb": file_size
                })
                
            elif response.status_code == 503:
                print(f"  ⚠ 服务不可用 - AI模型未加载")
                results.append({
                    "method": endpoint['name'],
                    "status": "unavailable",
                    "error": "AI模型未加载"
                })
            else:
                print(f"  ✗ 失败 - {response.status_code}")
                results.append({
                    "method": endpoint['name'],
                    "status": "failed",
                    "error": f"HTTP {response.status_code}"
                })
                
        except Exception as e:
            print(f"  ✗ 错误 - {e}")
            results.append({
                "method": endpoint['name'],
                "status": "error",
                "error": str(e)
            })
    
    # 输出总结
    print("\n" + "=" * 60)
    print("测试结果总结")
    print("=" * 60)
    
    success_count = 0
    for result in results:
        status_icon = "✓" if result['status'] == 'success' else "✗"
        print(f"{status_icon} {result['method']}: {result['status']}")
        if result['status'] == 'success':
            success_count += 1
            print(f"   文件: {result['file']} ({result['size_kb']:.1f} KB)")
        elif result['status'] == 'unavailable':
            print(f"   原因: {result['error']}")
        elif result['status'] in ['failed', 'error']:
            print(f"   错误: {result['error']}")
    
    print(f"\n总计: {success_count}/{len(results)} 个方法成功")

def main():
    """主函数"""
    print("AI大模型API测试工具")
    print("=" * 60)
    
    # 检查服务器是否运行
    try:
        response = requests.get("http://localhost:8000/api/health")
        if response.status_code != 200:
            print("✗ 服务器未运行，请先启动服务器")
            return
        print("✓ 服务器运行正常")
    except:
        print("✗ 无法连接到服务器，请先启动服务器")
        return
    
    # 查找测试图片
    test_images = [
        "uploads/1750831737383.jpg",
        "uploads/1753251189_1750831737383.jpg",
        "backend/test_images/test_input.png"
    ]
    
    input_image = None
    for img_path in test_images:
        if os.path.exists(img_path):
            input_image = img_path
            break
    
    if input_image is None:
        print("✗ 未找到测试图片，请上传一些图片到uploads目录")
        return
    
    print(f"使用测试图片: {input_image}")
    
    # 测试AI模型状态
    model_available = test_ai_model_status()
    
    if model_available:
        # 测试AI模型生成
        test_ai_model_generation(input_image)
    
    # 测试所有方法对比
    test_all_generation_methods(input_image)
    
    print("\n测试完成！")

if __name__ == "__main__":
    main() 