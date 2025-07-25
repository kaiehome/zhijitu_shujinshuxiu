#!/usr/bin/env python3
"""
测试本地AI增强专业识别图生成API
"""

import requests
import time
import os

def test_local_ai_enhanced_api():
    """测试本地AI增强专业识别图生成API"""
    
    # API端点
    url = "http://localhost:8000/api/generate-local-ai-enhanced"
    
    # 测试图像路径
    test_image_path = "backend/uploads/250625_162043.jpg"
    
    if not os.path.exists(test_image_path):
        print(f"测试图像不存在: {test_image_path}")
        return
    
    try:
        print("开始测试本地AI增强专业识别图生成API...")
        print(f"使用测试图像: {test_image_path}")
        
        # 准备文件上传
        with open(test_image_path, 'rb') as f:
            files = {'file': (os.path.basename(test_image_path), f, 'image/jpeg')}
            
            # 发送请求
            start_time = time.time()
            response = requests.post(url, files=files)
            end_time = time.time()
            
            processing_time = end_time - start_time
            
            print(f"API响应状态码: {response.status_code}")
            print(f"处理时间: {processing_time:.2f}秒")
            
            if response.status_code == 200:
                print("✅ 本地AI增强专业识别图生成成功!")
                
                # 保存响应内容
                output_filename = f"local_ai_enhanced_response_{int(time.time())}.jpg"
                with open(output_filename, 'wb') as f:
                    f.write(response.content)
                print(f"响应内容已保存到: {output_filename}")
                
                # 检查文件大小
                file_size = len(response.content)
                print(f"生成图像文件大小: {file_size} 字节")
                
            else:
                print(f"❌ API调用失败: {response.text}")
                
    except Exception as e:
        print(f"❌ 测试失败: {str(e)}")

if __name__ == "__main__":
    test_local_ai_enhanced_api() 