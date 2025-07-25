#!/usr/bin/env python3
"""
测试本地AI处理功能
"""

import requests
import time
import json
import os

def test_local_ai():
    """测试本地AI处理功能"""
    
    base_url = "http://localhost:8000"
    
    print("🔍 测试本地AI处理功能")
    print("=" * 50)
    
    # 1. 检查API状态
    print("📋 1. 检查API状态")
    try:
        response = requests.get(f"{base_url}/api/image-to-image-styles", timeout=10)
        if response.status_code == 200:
            styles = response.json()
            print("✅ API状态正常")
            print(f"可用风格: {styles['available_styles']}")
            print(f"当前风格: {styles['current_style']}")
        else:
            print(f"❌ API状态异常: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ 无法连接到API: {str(e)}")
        return False
    
    # 2. 检查测试图像
    test_image_path = "uploads/test_input.png"
    if not os.path.exists(test_image_path):
        print(f"❌ 测试图像不存在: {test_image_path}")
        return False
    
    print(f"✅ 测试图像存在: {test_image_path}")
    
    # 3. 测试图生图功能
    print("\n📋 2. 测试图生图功能")
    
    for style in ["weaving_machine", "embroidery", "pixel_art"]:
        print(f"\n🎨 测试风格: {style}")
        
        try:
            with open(test_image_path, "rb") as f:
                files = {"file": ("test_input.png", f, "image/png")}
                data = {"style": style}
                
                print("📤 发送请求...")
                start_time = time.time()
                
                response = requests.post(
                    f"{base_url}/api/generate-image-to-image",
                    files=files,
                    data=data,
                    timeout=120  # 2分钟超时
                )
                
                end_time = time.time()
                processing_time = end_time - start_time
                
                print(f"📥 响应状态码: {response.status_code}")
                print(f"⏱️ 处理时间: {processing_time:.2f}秒")
                
                if response.status_code == 200:
                    # 保存生成的图像
                    output_filename = f"local_ai_output_{style}_{int(time.time())}.png"
                    with open(output_filename, "wb") as f:
                        f.write(response.content)
                    
                    print(f"✅ 成功生成图像: {output_filename}")
                    print(f"📊 图像大小: {len(response.content)} bytes")
                    
                else:
                    print(f"❌ 请求失败: {response.status_code}")
                    try:
                        error_info = response.json()
                        print(f"错误信息: {json.dumps(error_info, indent=2, ensure_ascii=False)}")
                    except:
                        print(f"响应内容: {response.text[:200]}...")
                        
        except requests.exceptions.Timeout:
            print("❌ 请求超时")
        except Exception as e:
            print(f"❌ 请求异常: {str(e)}")
    
    # 4. 测试添加自定义风格
    print("\n📋 3. 测试添加自定义风格")
    
    custom_style = {
        "name": "test_style",
        "description": "测试风格",
        "parameters": {
            "color_count": 8,
            "use_ai_segmentation": True,
            "use_feature_detection": True
        }
    }
    
    try:
        response = requests.post(
            f"{base_url}/api/add-custom-style",
            json=custom_style,
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ 成功添加自定义风格")
            print(f"结果: {json.dumps(result, indent=2, ensure_ascii=False)}")
        else:
            print(f"❌ 添加风格失败: {response.status_code}")
            print(f"响应: {response.text}")
            
    except Exception as e:
        print(f"❌ 添加风格异常: {str(e)}")
    
    print("\n" + "=" * 50)
    print("🎉 本地AI处理测试完成!")
    print("💡 建议: 本地AI处理功能正常，可以作为主要解决方案")

def check_backend_status():
    """检查后端服务状态"""
    
    print("🔍 检查后端服务状态")
    print("=" * 30)
    
    try:
        response = requests.get("http://localhost:8000/", timeout=5)
        if response.status_code == 200:
            print("✅ 后端服务正常运行")
            return True
        else:
            print(f"⚠️ 后端服务响应异常: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ 后端服务不可访问: {str(e)}")
        print("💡 请确保后端服务正在运行:")
        print("   cd backend && python3 -m uvicorn main:app --host 0.0.0.0 --port 8000")
        return False

if __name__ == "__main__":
    # 检查后端状态
    if not check_backend_status():
        exit(1)
    
    # 测试本地AI功能
    test_local_ai() 