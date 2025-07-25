# test_image_to_image.py
# 测试图生图功能

import requests
import os
import json
from pathlib import Path

def test_image_to_image_api():
    """测试图生图API"""
    
    # API配置
    base_url = "http://localhost:8000"
    
    # 测试图片路径
    test_image_path = "uploads/orig.png"
    if not os.path.exists(test_image_path):
        print(f"❌ 测试图片不存在: {test_image_path}")
        return
    
    print(f"✅ 测试图片存在: {test_image_path}")
    
    # 1. 获取可用风格
    print("\n🔍 获取可用风格...")
    try:
        response = requests.get(f"{base_url}/api/image-to-image-styles")
        if response.status_code == 200:
            styles_info = response.json()
            print(f"✅ 可用风格: {styles_info['available_styles']}")
            print(f"当前风格: {styles_info['current_style']}")
        else:
            print(f"❌ 获取风格失败: {response.status_code}")
            return
    except Exception as e:
        print(f"❌ 获取风格失败: {e}")
        return
    
    # 2. 测试不同风格
    available_styles = styles_info['available_styles']
    
    for style in available_styles:
        print(f"\n🎨 测试风格: {style}")
        
        try:
            # 准备文件上传
            with open(test_image_path, 'rb') as f:
                files = {'file': (os.path.basename(test_image_path), f, 'image/png')}
                data = {
                    'style_preset': style,
                    'color_count': 16
                }
                
                # 发送请求
                response = requests.post(
                    f"{base_url}/api/generate-image-to-image",
                    files=files,
                    data=data,
                    timeout=60
                )
                
                if response.status_code == 200:
                    # 保存结果
                    output_filename = f"test_img2img_{style}.jpg"
                    with open(output_filename, 'wb') as f:
                        f.write(response.content)
                    print(f"✅ 风格 '{style}' 处理成功: {output_filename}")
                else:
                    print(f"❌ 风格 '{style}' 处理失败: {response.status_code}")
                    print(f"错误信息: {response.text}")
                    
        except Exception as e:
            print(f"❌ 风格 '{style}' 测试失败: {e}")
    
    print("\n🎉 图生图测试完成!")

def test_custom_style():
    """测试自定义风格"""
    
    base_url = "http://localhost:8000"
    
    # 自定义风格配置
    custom_style_config = {
        "color_count": 8,
        "edge_enhancement": True,
        "noise_reduction": True,
        "saturation_boost": 2.0,
        "contrast_boost": 1.8,
        "smooth_kernel": 0,
        "quantization_method": "force_limited"
    }
    
    print("\n🎨 测试自定义风格...")
    
    try:
        data = {
            'style_name': 'custom_test',
            'style_config': json.dumps(custom_style_config)
        }
        
        response = requests.post(
            f"{base_url}/api/add-custom-style",
            data=data,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ 自定义风格添加成功: {result['message']}")
            print(f"可用风格: {result['available_styles']}")
        else:
            print(f"❌ 自定义风格添加失败: {response.status_code}")
            print(f"错误信息: {response.text}")
            
    except Exception as e:
        print(f"❌ 自定义风格测试失败: {e}")

if __name__ == "__main__":
    print("🚀 开始测试图生图功能...")
    
    # 测试基本功能
    test_image_to_image_api()
    
    # 测试自定义风格
    test_custom_style()
    
    print("\n✨ 所有测试完成!") 