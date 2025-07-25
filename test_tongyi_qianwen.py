#!/usr/bin/env python3
"""
测试通义千问API的脚本
"""

import requests
import time
import os
from pathlib import Path

def test_tongyi_qianwen_api():
    """测试通义千问API功能"""
    base_url = "http://localhost:8000"
    
    print("=== 测试通义千问API ===")
    
    # 1. 测试状态端点
    print("\n1. 测试通义千问状态...")
    try:
        response = requests.get(f"{base_url}/api/tongyi-qianwen-status", timeout=10)
        if response.status_code == 200:
            status = response.json()
            print(f"✓ 通义千问状态正常")
            print(f"  API可用: {status.get('api_available', False)}")
            print(f"  模型: {status.get('model', 'unknown')}")
            print(f"  API密钥已设置: {status.get('api_key_set', False)}")
        else:
            print(f"✗ 通义千问状态异常: {response.status_code}")
    except Exception as e:
        print(f"✗ 通义千问状态检查失败: {e}")
    
    # 2. 查找测试图片
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
    
    print(f"\n使用测试图片: {test_image}")
    
    # 3. 测试图片生成
    print("\n2. 测试通义千问图片生成...")
    try:
        with open(test_image, 'rb') as f:
            files = {'file': f}
            response = requests.post(
                f"{base_url}/api/generate-tongyi-qianwen", 
                files=files, 
                timeout=120  # 通义千问可能需要更长时间
            )
        
        if response.status_code == 200:
            # 保存生成的图片
            output_filename = "test_tongyi_qianwen_output.jpg"
            with open(output_filename, 'wb') as f:
                f.write(response.content)
            
            print(f"✓ 通义千问图片生成成功")
            print(f"  输出文件: {output_filename}")
            print(f"  文件大小: {len(response.content)} 字节")
        elif response.status_code == 503:
            print("⚠ 通义千问API未配置")
            print("  请设置TONGYI_API_KEY环境变量")
        else:
            print(f"✗ 通义千问图片生成失败")
            print(f"  状态码: {response.status_code}")
            print(f"  错误信息: {response.text}")
            
    except Exception as e:
        print(f"✗ 通义千问图片生成异常: {e}")
    
    # 4. 测试图片增强
    print("\n3. 测试通义千问图片增强...")
    try:
        with open(test_image, 'rb') as f:
            files = {'file': f}
            response = requests.post(
                f"{base_url}/api/enhance-tongyi-qianwen", 
                files=files, 
                timeout=120
            )
        
        if response.status_code == 200:
            # 保存增强的图片
            output_filename = "test_tongyi_qianwen_enhanced.jpg"
            with open(output_filename, 'wb') as f:
                f.write(response.content)
            
            print(f"✓ 通义千问图片增强成功")
            print(f"  输出文件: {output_filename}")
            print(f"  文件大小: {len(response.content)} 字节")
        elif response.status_code == 503:
            print("⚠ 通义千问API未配置")
            print("  请设置TONGYI_API_KEY环境变量")
        else:
            print(f"✗ 通义千问图片增强失败")
            print(f"  状态码: {response.status_code}")
            print(f"  错误信息: {response.text}")
            
    except Exception as e:
        print(f"✗ 通义千问图片增强异常: {e}")
    
    print(f"\n=== 测试完成 ===")

if __name__ == "__main__":
    test_tongyi_qianwen_api() 