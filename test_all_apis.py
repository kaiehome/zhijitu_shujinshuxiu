#!/usr/bin/env python3
"""
测试所有API端点的脚本
包括传统图像处理和AI模型生成
"""

import requests
import time
import os
from pathlib import Path

def test_api_endpoint(endpoint_name, url, file_path):
    """测试单个API端点"""
    print(f"\n=== 测试 {endpoint_name} ===")
    
    try:
        with open(file_path, 'rb') as f:
            files = {'file': f}
            response = requests.post(url, files=files, timeout=30)
        
        if response.status_code == 200:
            # 保存生成的图片
            output_filename = f"test_output_{endpoint_name.replace(' ', '_').lower()}.jpg"
            with open(output_filename, 'wb') as f:
                f.write(response.content)
            
            print(f"✓ {endpoint_name} 成功")
            print(f"  输出文件: {output_filename}")
            print(f"  文件大小: {len(response.content)} 字节")
        else:
            print(f"✗ {endpoint_name} 失败")
            print(f"  状态码: {response.status_code}")
            print(f"  错误信息: {response.text}")
            
    except Exception as e:
        print(f"✗ {endpoint_name} 异常")
        print(f"  错误: {str(e)}")

def test_status_endpoint(endpoint_name, url):
    """测试状态端点"""
    print(f"\n=== 测试 {endpoint_name} 状态 ===")
    
    try:
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            print(f"✓ {endpoint_name} 状态正常")
            print(f"  响应: {response.json()}")
        else:
            print(f"✗ {endpoint_name} 状态异常")
            print(f"  状态码: {response.status_code}")
            print(f"  错误信息: {response.text}")
            
    except Exception as e:
        print(f"✗ {endpoint_name} 状态检查异常")
        print(f"  错误: {str(e)}")

def main():
    """主函数"""
    base_url = "http://localhost:8000"
    
    # 查找测试图片
    test_image = None
    for path in ["uploads", "target_images"]:
        if os.path.exists(path):
            for file in os.listdir(path):
                if file.endswith(('.jpg', '.png')):
                    test_image = os.path.join(path, file)
                    break
        if test_image:
            break
    
    if not test_image:
        print("错误: 找不到测试图片")
        return
    
    print(f"使用测试图片: {test_image}")
    
    # 测试状态端点
    test_status_endpoint("传统专业生成", f"{base_url}/api/professional-status")
    test_status_endpoint("高级专业生成", f"{base_url}/api/advanced-status")
    test_status_endpoint("终极专业生成", f"{base_url}/api/ultimate-status")
    test_status_endpoint("本地AI增强", f"{base_url}/api/local-ai-status")
    test_status_endpoint("AI大模型", f"{base_url}/api/ai-model-status")
    test_status_endpoint("简化AI模型", f"{base_url}/api/simple-ai-status")
    
    # 测试图片生成端点
    endpoints = [
        ("传统专业生成", f"{base_url}/api/generate-professional"),
        ("高级专业生成", f"{base_url}/api/generate-advanced"),
        ("终极专业生成", f"{base_url}/api/generate-ultimate"),
        ("本地AI增强", f"{base_url}/api/generate-local-ai-enhanced"),
        ("AI大模型", f"{base_url}/api/generate-ai-model"),
        ("简化AI模型", f"{base_url}/api/generate-simple-ai")
    ]
    
    for endpoint_name, url in endpoints:
        test_api_endpoint(endpoint_name, url, test_image)
        time.sleep(1)  # 避免请求过于频繁
    
    print(f"\n=== 测试完成 ===")
    print("所有生成的图片已保存到当前目录")

if __name__ == "__main__":
    main() 