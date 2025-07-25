#!/usr/bin/env python3
"""
测试专业识别图生成API
"""

import requests
import time
import os

def test_professional_recognition_api():
    """测试专业识别图生成API"""
    
    # API端点
    url = "http://localhost:8000/api/generate-professional-recognition"
    
    # 测试图像路径
    test_image_path = "backend/uploads/250625_162043.jpg"
    
    if not os.path.exists(test_image_path):
        print(f"测试图像不存在: {test_image_path}")
        return
    
    try:
        print("开始测试专业识别图生成API...")
        print(f"使用测试图像: {test_image_path}")
        
        # 准备文件上传
        with open(test_image_path, 'rb') as f:
            files = {'file': (os.path.basename(test_image_path), f, 'image/jpeg')}
            
            # 发送请求
            start_time = time.time()
            response = requests.post(url, files=files)
            end_time = time.time()
            
            print(f"请求耗时: {end_time - start_time:.2f}秒")
            print(f"响应状态码: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                print("API响应:")
                print(f"  成功: {result.get('success')}")
                print(f"  消息: {result.get('message')}")
                print(f"  原始图像: {result.get('original_image')}")
                print(f"  处理后图像: {result.get('processed_image')}")
                print(f"  时间戳: {result.get('timestamp')}")
                
                # 检查输出文件是否存在
                output_path = result.get('processed_image')
                if output_path and os.path.exists(output_path):
                    file_size = os.path.getsize(output_path)
                    print(f"  输出文件大小: {file_size} 字节")
                    print("✅ 专业识别图生成成功！")
                else:
                    print("❌ 输出文件不存在")
            else:
                print(f"❌ API请求失败: {response.text}")
                
    except Exception as e:
        print(f"❌ 测试过程中出错: {str(e)}")

if __name__ == "__main__":
    test_professional_recognition_api() 