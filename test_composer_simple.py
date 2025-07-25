# test_composer_simple.py
# 简单测试通义千问Composer图生图API

import requests
import os
import time

def test_composer_api():
    """测试Composer API"""
    
    base_url = "http://localhost:8000"
    test_image_path = "uploads/test_input.png"
    
    if not os.path.exists(test_image_path):
        print(f"❌ 测试图片不存在: {test_image_path}")
        return
    
    print(f"✅ 测试图片存在: {test_image_path}")
    
    # 1. 检查API状态
    print("\n🔍 检查API状态...")
    try:
        response = requests.get(f"{base_url}/api/composer-status")
        if response.status_code == 200:
            status = response.json()
            print(f"✅ API状态: {status['api_status']}")
        else:
            print(f"❌ API状态检查失败: {response.status_code}")
            return
    except Exception as e:
        print(f"❌ API状态检查异常: {e}")
        return
    
    # 2. 测试图生图
    print("\n🎨 测试图生图功能...")
    try:
        with open(test_image_path, "rb") as f:
            files = {"file": ("test_input.png", f, "image/png")}
            data = {
                "prompt": "生成织机识别图，保持原有结构，增强边缘清晰度",
                "style_preset": "weaving_machine"
            }
            
            print("📤 发送请求...")
            response = requests.post(
                f"{base_url}/api/generate-composer-image-to-image",
                files=files,
                data=data,
                timeout=120
            )
            
            print(f"📊 响应状态码: {response.status_code}")
            print(f"📄 响应头: {dict(response.headers)}")
            
            if response.status_code == 200:
                # 检查响应内容类型
                content_type = response.headers.get('content-type', '')
                if 'image' in content_type:
                    # 保存图片
                    output_path = "test_composer_simple.jpg"
                    with open(output_path, "wb") as f:
                        f.write(response.content)
                    print(f"✅ 图片已保存: {output_path}")
                    print(f"📏 文件大小: {len(response.content)} bytes")
                else:
                    # 可能是JSON响应
                    try:
                        result = response.json()
                        print(f"📋 响应内容: {result}")
                    except:
                        print(f"📋 响应内容: {response.text[:500]}")
            else:
                print(f"❌ 请求失败: {response.status_code}")
                print(f"错误信息: {response.text}")
                
    except Exception as e:
        print(f"❌ 图生图测试异常: {e}")

if __name__ == "__main__":
    print("🚀 开始测试通义千问Composer API...")
    test_composer_api()
    print("✨ 测试完成！") 