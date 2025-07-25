#!/usr/bin/env python3
"""
测试上传流程的脚本
"""

import requests
import time
import json

def test_upload_flow():
    """测试完整的上传流程"""
    
    # 1. 测试健康检查
    print("🔍 测试健康检查...")
    try:
        response = requests.get("http://127.0.0.1:8000/api/health")
        print(f"✅ 健康检查: {response.status_code}")
        print(f"   响应: {response.json()}")
    except Exception as e:
        print(f"❌ 健康检查失败: {e}")
        return
    
    # 2. 测试文件上传
    print("\n📤 测试文件上传...")
    try:
        with open("backend/uploads/1750831737383.jpg", "rb") as f:
            files = {"file": ("test_image.jpg", f, "image/jpeg")}
            response = requests.post("http://127.0.0.1:8000/api/upload", files=files)
        
        if response.status_code == 200:
            upload_result = response.json()
            print(f"✅ 上传成功: {upload_result}")
            
            # 3. 测试图像处理
            print("\n🖼️ 测试图像处理...")
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
                
                # 4. 检查任务状态
                if "job_id" in process_result:
                    job_id = process_result["job_id"]
                    print(f"\n📊 检查任务状态: {job_id}")
                    
                    for i in range(5):  # 最多检查5次
                        time.sleep(2)
                        try:
                            status_response = requests.get(f"http://127.0.0.1:8000/api/status/{job_id}")
                            if status_response.status_code == 200:
                                status = status_response.json()
                                print(f"   状态: {status.get('status', 'unknown')}")
                                if status.get('status') == 'completed':
                                    print(f"✅ 任务完成: {status}")
                                    break
                        except Exception as e:
                            print(f"   状态检查失败: {e}")
            else:
                print(f"❌ 处理失败: {response.status_code} - {response.text}")
        else:
            print(f"❌ 上传失败: {response.status_code} - {response.text}")
            
    except Exception as e:
        print(f"❌ 测试失败: {e}")

if __name__ == "__main__":
    test_upload_flow() 