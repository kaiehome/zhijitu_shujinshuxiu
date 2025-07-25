#!/usr/bin/env python3
"""
测试所有专业识别图生成方式（包括终极版本）
"""

import requests
import time
import os

def test_all_ultimate_styles():
    """测试所有专业识别图生成方式"""
    
    # 测试图像路径
    test_image_path = "backend/uploads/250625_162043.jpg"
    
    if not os.path.exists(test_image_path):
        print(f"测试图像不存在: {test_image_path}")
        return
    
    # 定义所有API端点
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
        }
    ]
    
    print("开始测试所有专业识别图生成方式...")
    print(f"使用测试图像: {test_image_path}")
    print("=" * 70)
    
    results = []
    
    for i, endpoint in enumerate(endpoints, 1):
        print(f"\n{i}. 测试 {endpoint['name']}...")
        
        try:
            # 准备文件上传
            with open(test_image_path, 'rb') as f:
                files = {'file': (os.path.basename(test_image_path), f, 'image/jpeg')}
                
                # 发送请求
                start_time = time.time()
                response = requests.post(endpoint['url'], files=files)
                end_time = time.time()
                
                processing_time = end_time - start_time
                print(f"   请求耗时: {processing_time:.2f}秒")
                print(f"   响应状态码: {response.status_code}")
                
                if response.status_code == 200:
                    result = response.json()
                    print(f"   成功: {result.get('success')}")
                    print(f"   消息: {result.get('message')}")
                    
                    # 检查输出文件
                    output_path = result.get('processed_image') or result.get('output_path')
                    if output_path and os.path.exists(output_path):
                        file_size = os.path.getsize(output_path)
                        print(f"   输出文件: {output_path}")
                        print(f"   文件大小: {file_size} 字节")
                        print(f"   ✅ {endpoint['name']} 成功")
                        
                        results.append({
                            "name": endpoint['name'],
                            "success": True,
                            "processing_time": processing_time,
                            "output_path": output_path,
                            "file_size": file_size
                        })
                    else:
                        print(f"   ❌ 输出文件不存在")
                        results.append({
                            "name": endpoint['name'],
                            "success": False,
                            "error": "输出文件不存在"
                        })
                else:
                    print(f"   ❌ API请求失败: {response.text}")
                    results.append({
                        "name": endpoint['name'],
                        "success": False,
                        "error": f"HTTP {response.status_code}"
                    })
                    
        except Exception as e:
            print(f"   ❌ 测试过程中出错: {str(e)}")
            results.append({
                "name": endpoint['name'],
                "success": False,
                "error": str(e)
            })
    
    # 输出总结
    print("\n" + "=" * 70)
    print("测试总结:")
    print("=" * 70)
    
    successful_tests = [r for r in results if r['success']]
    failed_tests = [r for r in results if not r['success']]
    
    print(f"成功: {len(successful_tests)}/{len(results)}")
    print(f"失败: {len(failed_tests)}/{len(results)}")
    
    if successful_tests:
        print("\n成功的测试:")
        for result in successful_tests:
            print(f"  ✅ {result['name']}")
            print(f"     处理时间: {result['processing_time']:.2f}秒")
            print(f"     输出文件: {result['output_path']}")
            print(f"     文件大小: {result['file_size']} 字节")
    
    if failed_tests:
        print("\n失败的测试:")
        for result in failed_tests:
            print(f"  ❌ {result['name']}: {result['error']}")
    
    print("\n所有测试完成！")
    print("\n推荐使用顺序（按效果排序）:")
    print("1. 终极专业识别图生成 - 最新优化版本")
    print("2. 高级专业识别图生成 - 高饱和度强对比度")
    print("3. 专业识别图生成 - 基础专业版本")
    print("4. 传统专业生成 - 原始版本")

if __name__ == "__main__":
    test_all_ultimate_styles() 