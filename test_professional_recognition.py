#!/usr/bin/env python3
"""
测试专业识别图效果的脚本
"""

import requests
import time
import json
import os

def test_professional_recognition():
    """测试专业识别图效果"""
    
    print("🎯 测试专业识别图效果...")
    
    # 测试不同的专业参数
    test_configs = [
        {
            "name": "极端专业识别图 (4色)",
            "color_count": 4,
            "edge_enhancement": True,
            "noise_reduction": True
        },
        {
            "name": "专业识别图 (6色)",
            "color_count": 6,
            "edge_enhancement": True,
            "noise_reduction": True
        },
        {
            "name": "标准识别图 (8色)",
            "color_count": 8,
            "edge_enhancement": True,
            "noise_reduction": True
        }
    ]
    
    results = []
    
    for config in test_configs:
        print(f"\n🔄 测试 {config['name']}...")
        
        try:
            # 上传图像
            with open("backend/uploads/1750831737383.jpg", "rb") as f:
                files = {"file": ("test_image.jpg", f, "image/jpeg")}
                response = requests.post("http://127.0.0.1:8000/api/upload", files=files)
            
            if response.status_code == 200:
                print(f"✅ 上传成功")
                
                # 处理图像
                with open("backend/uploads/1750831737383.jpg", "rb") as f:
                    files = {"file": ("test_image.jpg", f, "image/jpeg")}
                    data = {
                        "color_count": config["color_count"],
                        "edge_enhancement": config["edge_enhancement"],
                        "noise_reduction": config["noise_reduction"]
                    }
                    response = requests.post("http://127.0.0.1:8000/api/process", files=files, data=data)
                
                if response.status_code == 200:
                    process_result = response.json()
                    job_id = process_result.get("job_id")
                    processing_time = process_result.get("processing_time", 0)
                    
                    print(f"✅ 处理成功 - 耗时: {processing_time:.2f}秒")
                    
                    # 检查文件大小
                    job_dir = f"backend/outputs/{job_id}"
                    if os.path.exists(job_dir):
                        files = os.listdir(job_dir)
                        for file in files:
                            if "professional" in file:
                                file_path = os.path.join(job_dir, file)
                                size = os.path.getsize(file_path)
                                print(f"   📄 {file}: {size/1024:.1f}KB")
                    
                    results.append({
                        "config": config,
                        "job_id": job_id,
                        "processing_time": processing_time,
                        "status": "success"
                    })
                else:
                    print(f"❌ 处理失败: {response.status_code}")
                    results.append({
                        "config": config,
                        "status": "failed",
                        "error": f"HTTP {response.status_code}"
                    })
            else:
                print(f"❌ 上传失败: {response.status_code}")
                results.append({
                    "config": config,
                    "status": "failed", 
                    "error": f"Upload HTTP {response.status_code}"
                })
                
        except Exception as e:
            print(f"❌ 测试失败: {e}")
            results.append({
                "config": config,
                "status": "failed",
                "error": str(e)
            })
        
        # 等待一下再测试下一个
        time.sleep(3)
    
    # 总结结果
    print(f"\n📊 专业识别图测试结果:")
    print("=" * 60)
    for result in results:
        config = result["config"]
        print(f"🎯 {config['name']}:")
        print(f"   - 颜色数量: {config['color_count']}")
        print(f"   - 边缘增强: {config['edge_enhancement']}")
        print(f"   - 降噪: {config['noise_reduction']}")
        print(f"   - 状态: {result['status']}")
        if result['status'] == 'success':
            print(f"   - 处理时间: {result['processing_time']:.2f}秒")
            print(f"   - 任务ID: {result['job_id']}")
        else:
            print(f"   - 错误: {result['error']}")
        print()

if __name__ == "__main__":
    test_professional_recognition() 