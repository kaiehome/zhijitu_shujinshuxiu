#!/usr/bin/env python3
"""
测试数字化识别图风格的脚本
"""

import requests
import time
import json

def test_digital_recognition_style():
    """测试数字化识别图风格"""
    
    print("🎯 测试数字化识别图风格...")
    
    # 1. 测试健康检查
    print("🔍 检查服务器状态...")
    try:
        response = requests.get("http://127.0.0.1:8000/api/health")
        print(f"✅ 服务器状态: {response.json()}")
    except Exception as e:
        print(f"❌ 服务器检查失败: {e}")
        return
    
    # 2. 上传并处理图像
    print("\n📤 上传图像并应用数字化识别图风格...")
    try:
        with open("backend/uploads/1750831737383.jpg", "rb") as f:
            files = {"file": ("test_image.jpg", f, "image/jpeg")}
            response = requests.post("http://127.0.0.1:8000/api/upload", files=files)
        
        if response.status_code == 200:
            upload_result = response.json()
            print(f"✅ 上传成功: {upload_result}")
            
            # 3. 处理图像
            print("\n🖼️ 处理图像...")
            with open("backend/uploads/1750831737383.jpg", "rb") as f:
                files = {"file": ("test_image.jpg", f, "image/jpeg")}
                data = {
                    "color_count": 16,
                    "edge_enhancement": True,
                    "noise_reduction": True
                }
                response = requests.post("http://127.0.0.1:8000/api/process", files=files, data=data)
            
            if response.status_code == 200:
                process_result = response.json()
                print(f"✅ 处理成功: {process_result}")
                
                # 4. 检查生成的文件
                job_id = process_result.get("job_id")
                if job_id:
                    print(f"\n📁 检查生成的文件...")
                    import os
                    job_dir = f"backend/outputs/{job_id}"
                    if os.path.exists(job_dir):
                        files = os.listdir(job_dir)
                        print(f"✅ 生成的文件: {files}")
                        
                        # 显示文件大小
                        for file in files:
                            file_path = os.path.join(job_dir, file)
                            size = os.path.getsize(file_path)
                            print(f"   📄 {file}: {size/1024:.1f}KB")
                    else:
                        print(f"❌ 输出目录不存在: {job_dir}")
                else:
                    print("❌ 未获取到任务ID")
            else:
                print(f"❌ 处理失败: {response.status_code} - {response.text}")
        else:
            print(f"❌ 上传失败: {response.status_code} - {response.text}")
            
    except Exception as e:
        print(f"❌ 测试失败: {e}")

if __name__ == "__main__":
    test_digital_recognition_style() 