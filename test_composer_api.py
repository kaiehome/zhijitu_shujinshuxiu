# test_composer_api.py
# 测试通义千问Composer图生图API

import requests
import os
import base64
import json
from pathlib import Path

def test_composer_api():
    """测试通义千问Composer图生图API"""
    
    # API配置
    base_url = "http://localhost:8000"
    
    # 测试图片路径
    test_image_path = "uploads/orig.png"
    if not os.path.exists(test_image_path):
        print(f"❌ 测试图片不存在: {test_image_path}")
        return
    
    print(f"✅ 测试图片存在: {test_image_path}")
    
    # 1. 测试API状态
    print("\n🔍 测试Composer API状态...")
    try:
        response = requests.get(f"{base_url}/api/composer-status")
        if response.status_code == 200:
            status_info = response.json()
            print(f"✅ API状态: {status_info}")
        else:
            print(f"❌ 获取API状态失败: {response.status_code}")
            return
    except Exception as e:
        print(f"❌ 连接失败: {e}")
        return
    
    # 2. 测试图生图功能 - 无提示词
    print("\n🎨 测试图生图 - 无提示词...")
    try:
        with open(test_image_path, "rb") as f:
            files = {"file": f}
            data = {
                "prompt": "",
                "style_preset": "weaving_machine"
            }
            
            response = requests.post(
                f"{base_url}/api/generate-composer-image-to-image",
                files=files,
                data=data,
                timeout=120
            )
            
            if response.status_code == 200:
                output_path = "test_composer_weaving.jpg"
                with open(output_path, "wb") as f:
                    f.write(response.content)
                print(f"✅ 生成成功: {output_path}")
                print(f"文件大小: {len(response.content)} bytes")
            else:
                print(f"❌ 生成失败: {response.status_code}")
                print(f"错误信息: {response.text}")
    except Exception as e:
        print(f"❌ 请求失败: {e}")
    
    # 3. 测试图生图功能 - 带提示词
    print("\n🎨 测试图生图 - 带提示词...")
    try:
        with open(test_image_path, "rb") as f:
            files = {"file": f}
            data = {
                "prompt": "生成织机识别图，保持原有结构，增强边缘清晰度",
                "style_preset": "embroidery"
            }
            
            response = requests.post(
                f"{base_url}/api/generate-composer-image-to-image",
                files=files,
                data=data,
                timeout=120
            )
            
            if response.status_code == 200:
                output_path = "test_composer_embroidery.jpg"
                with open(output_path, "wb") as f:
                    f.write(response.content)
                print(f"✅ 生成成功: {output_path}")
                print(f"文件大小: {len(response.content)} bytes")
            else:
                print(f"❌ 生成失败: {response.status_code}")
                print(f"错误信息: {response.text}")
    except Exception as e:
        print(f"❌ 请求失败: {e}")
    
    # 4. 测试图生图功能 - 像素艺术风格
    print("\n🎨 测试图生图 - 像素艺术风格...")
    try:
        with open(test_image_path, "rb") as f:
            files = {"file": f}
            data = {
                "prompt": "转换为像素艺术风格的织机识别图",
                "style_preset": "pixel_art"
            }
            
            response = requests.post(
                f"{base_url}/api/generate-composer-image-to-image",
                files=files,
                data=data,
                timeout=120
            )
            
            if response.status_code == 200:
                output_path = "test_composer_pixel.jpg"
                with open(output_path, "wb") as f:
                    f.write(response.content)
                print(f"✅ 生成成功: {output_path}")
                print(f"文件大小: {len(response.content)} bytes")
            else:
                print(f"❌ 生成失败: {response.status_code}")
                print(f"错误信息: {response.text}")
    except Exception as e:
        print(f"❌ 请求失败: {e}")
    
    # 5. 检查生成的文件
    print("\n📁 检查生成的文件...")
    generated_files = [
        "test_composer_weaving.jpg",
        "test_composer_embroidery.jpg", 
        "test_composer_pixel.jpg"
    ]
    
    for file_path in generated_files:
        if os.path.exists(file_path):
            file_size = os.path.getsize(file_path)
            print(f"✅ {file_path}: {file_size} bytes")
        else:
            print(f"❌ {file_path}: 文件不存在")

if __name__ == "__main__":
    print("🚀 开始测试通义千问Composer图生图API...")
    test_composer_api()
    print("\n✨ 测试完成！") 