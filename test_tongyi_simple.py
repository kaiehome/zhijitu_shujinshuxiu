#!/usr/bin/env python3
"""
简单的通义千问API测试脚本
"""

import os
import requests
import base64
from pathlib import Path

def test_tongyi_api():
    """测试通义千问API"""
    
    # API配置
    api_key = "sk-ade7e6a1728741fcb009dcf1419000de"
    base_url = "https://dashscope.aliyuncs.com"
    endpoint = "/api/v1/services/aigc/multimodal-generation/generation"
    model = "qwen-vl-plus"
    
    # 检查API密钥
    if not api_key:
        print("❌ API密钥未设置")
        return
    
    print(f"✅ API密钥已设置: {api_key[:10]}...")
    
    # 测试图片路径
    image_path = "uploads/orig.png"
    if not os.path.exists(image_path):
        print(f"❌ 测试图片不存在: {image_path}")
        return
    
    print(f"✅ 测试图片存在: {image_path}")
    
    # 编码图片
    try:
        with open(image_path, "rb") as f:
            image_data = f.read()
        base64_image = base64.b64encode(image_data).decode('utf-8')
        print(f"✅ 图片编码成功，大小: {len(base64_image)} 字符")
    except Exception as e:
        print(f"❌ 图片编码失败: {e}")
        return
    
    # 构建请求
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    style_prompt = "请将这张图片转换为织机识别图，风格要求：像素化，色彩丰富，边缘清晰，强对比度"
    data = {
        "model": model,
        "input": {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"image": base64_image},
                        {"text": style_prompt}
                    ]
                }
            ]
        }
    }
    try:
        response = requests.post(
            base_url + endpoint,
            headers=headers,
            json=data,
            timeout=60
        )
        print(f"状态码: {response.status_code}")
        print(f"响应: {response.text[:500]}...")
        if response.status_code == 200:
            print("✅ 成功!")
        else:
            print("❌ 失败")
    except Exception as e:
        print(f"❌ 请求失败: {e}")

if __name__ == "__main__":
    test_tongyi_api() 