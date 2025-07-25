#!/usr/bin/env python3
"""
测试所有专业识别图生成方式（包括本地AI增强版本）
"""

import requests
import time
import os

def test_all_ai_enhanced_styles():
    """测试所有专业识别图生成方式"""
    
    # 测试图像路径
    test_image_path = "backend/uploads/250625_162043.jpg"
    
    if not os.path.exists(test_image_path):
        print(f"测试图像不存在: {test_image_path}")
        return
    
    # 定义所有API端点
    endpoints = [
        {
            "name": "传统专业生成",
            "url": "http://localhost:8000/api/upload"
        },
        {
            "name": "专业识别图生成",
            "url": "http://localhost:8000/api/generate-professional-recognition"
        },
        {
            "name": "高级专业识别图生成", 
            "url": "http://localhost:8000/api/generate-advanced-professional"
        },
        {
            "name": "终极专业识别图生成",
            "url": "http://localhost:8000/api/generate-ultimate-professional"
        },
        {
            "name": "本地AI增强专业识别图生成",
            "url": "http://localhost:8000/api/generate-local-ai-enhanced"
        }
    ]
    
    print("开始测试所有专业识别图生成方式...")
    print(f"使用测试图像: {test_image_path}")
    print("=" * 60)
    
    results = []
    
    for i, endpoint in enumerate(endpoints, 1):
        print(f"\n{i}. 测试 {endpoint['name']}...")
        
        try:
            # 准备文件上传
            with open(test_image_path, 'rb') as f:
                files = {'file': (os.path.basename(test_image_path), f, 'image/jpeg')}
                
                # 发送请求
                start_time = time.time()
                response = requests.post(endpoint['url'], files=files)
                end_time = time.time()
                
                processing_time = end_time - start_time
                
                if response.status_code == 200:
                    print(f"✅ {endpoint['name']} 成功!")
                    print(f"   处理时间: {processing_time:.2f}秒")
                    print(f"   文件大小: {len(response.content)} 字节")
                    
                    # 保存响应内容
                    output_filename = f"{endpoint['name'].replace(' ', '_')}_{int(time.time())}.jpg"
                    with open(output_filename, 'wb') as f:
                        f.write(response.content)
                    print(f"   保存到: {output_filename}")
                    
                    results.append({
                        "name": endpoint['name'],
                        "status": "success",
                        "time": processing_time,
                        "size": len(response.content),
                        "file": output_filename
                    })
                    
                else:
                    print(f"❌ {endpoint['name']} 失败: {response.status_code}")
                    print(f"   错误信息: {response.text}")
                    
                    results.append({
                        "name": endpoint['name'],
                        "status": "failed",
                        "error": response.text
                    })
                    
        except Exception as e:
            print(f"❌ {endpoint['name']} 异常: {str(e)}")
            results.append({
                "name": endpoint['name'],
                "status": "error",
                "error": str(e)
            })
    
    # 输出总结
    print("\n" + "=" * 60)
    print("测试总结:")
    print("=" * 60)
    
    successful = [r for r in results if r['status'] == 'success']
    failed = [r for r in results if r['status'] != 'success']
    
    print(f"✅ 成功: {len(successful)}/{len(endpoints)}")
    print(f"❌ 失败: {len(failed)}/{len(endpoints)}")
    
    if successful:
        print("\n成功生成的专业识别图:")
        for result in successful:
            print(f"  • {result['name']}: {result['time']:.2f}秒, {result['size']}字节")
    
    if failed:
        print("\n失败的生成方式:")
        for result in failed:
            print(f"  • {result['name']}: {result.get('error', '未知错误')}")
    
    print(f"\n所有生成的文件已保存到当前目录")
    print("请查看生成的结果，选择最符合您需求的专业识别图效果")

if __name__ == "__main__":
    test_all_ai_enhanced_styles() 