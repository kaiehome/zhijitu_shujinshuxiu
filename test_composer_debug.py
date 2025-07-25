# test_composer_debug.py
# 调试通义千问Composer API

import os
import base64
import requests
import json
from pathlib import Path
import time

def test_composer_direct():
    """直接测试Composer API调用"""
    
    # API配置
    api_key = "sk-ade7e6a1728741fcb009dcf1419000de"
    base_url = "https://dashscope.aliyuncs.com"
    endpoint = "/api/v1/services/aigc/image2image/image-synthesis"
    model = "wanx-v1"  # 尝试使用wanx-v1模型
    
    # 检查API密钥
    if not api_key:
        print("❌ API密钥未设置")
        return
    
    print(f"✅ API密钥已设置: {api_key[:10]}...")
    
    # 测试图片路径
    image_path = "uploads/test_embroidery_style.png"
    if not os.path.exists(image_path):
        print(f"❌ 测试图片不存在: {image_path}")
        return
    
    print(f"✅ 测试图片存在: {image_path}")
    
    # 编码图片
    try:
        with open(image_path, "rb") as f:
            image_data = f.read()
        base64_image = base64.b64encode(image_data).decode('utf-8')
        print(f"✅ 图片编码成功，大小: {len(base64_image)} 字符")
    except Exception as e:
        print(f"❌ 图片编码失败: {e}")
        return
    
    # 构造请求头
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "X-DashScope-Async": "enable"
    }
    
    # 构造请求体 - 通义万相图生图格式
    data = {
        "model": model,
        "input": {
            "image": base64_image
        },
        "parameters": {
            "prompt": "生成织机识别图，保持原有结构，增强边缘清晰度"
        }
    }
    
    print(f"🔍 发送请求到: {base_url}{endpoint}")
    print(f"📝 请求体预览: {json.dumps(data, indent=2)[:500]}...")
    
    try:
        response = requests.post(
            f"{base_url}{endpoint}",
            headers=headers,
            json=data,
            timeout=60
        )
        
        print(f"📊 响应状态码: {response.status_code}")
        print(f"📄 响应头: {dict(response.headers)}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ 请求成功: {json.dumps(result, indent=2)}")
            
            # 检查是否是异步任务
            if "output" in result and "task_id" in result["output"]:
                task_id = result["output"]["task_id"]
                task_status = result["output"].get("task_status", "PENDING")
                print(f"🔄 异步任务ID: {task_id}")
                print(f"📊 任务状态: {task_status}")
                
                # 轮询任务状态
                if task_status in ["PENDING", "RUNNING"]:
                    print("⏳ 开始轮询任务状态...")
                    max_attempts = 60  # 最多等待5分钟
                    attempt = 0
                    
                    while attempt < max_attempts:
                        attempt += 1
                        time.sleep(5)  # 等待5秒
                        
                        # 查询任务状态
                        status_response = requests.get(
                            f"{base_url}/api/v1/tasks/{task_id}",
                            headers=headers,
                            timeout=30
                        )
                        
                        if status_response.status_code == 200:
                            status_result = status_response.json()
                            print(f"📊 轮询 {attempt}: {status_result}")
                            
                            if "output" in status_result:
                                task_status = status_result["output"].get("task_status")
                                
                                if task_status == "SUCCEEDED":
                                    print("✅ 任务完成！")
                                    # 检查是否有图片数据
                                    if "images" in status_result["output"]:
                                        images = status_result["output"]["images"]
                                        if images:
                                            image_data = images[0]
                                            if "url" in image_data:
                                                print(f"🎨 生成图片URL: {image_data['url']}")
                                                
                                                # 下载图片
                                                img_response = requests.get(image_data['url'])
                                                if img_response.status_code == 200:
                                                    output_path = "test_composer_direct.jpg"
                                                    with open(output_path, "wb") as f:
                                                        f.write(img_response.content)
                                                    print(f"✅ 图片已保存: {output_path}")
                                                    print(f"📏 文件大小: {len(img_response.content)} bytes")
                                                    return
                                                else:
                                                    print(f"❌ 下载图片失败: {img_response.status_code}")
                                            else:
                                                print(f"❌ 图片数据格式错误: {image_data}")
                                        else:
                                            print("❌ 没有生成图片")
                                    break
                                elif task_status == "FAILED":
                                    print("❌ 任务失败")
                                    break
                                elif task_status in ["PENDING", "RUNNING"]:
                                    print(f"⏳ 任务状态: {task_status}, 继续等待...")
                                else:
                                    print(f"❓ 未知任务状态: {task_status}")
                                    break
                        else:
                            print(f"❌ 查询任务状态失败: {status_response.status_code}")
                            break
                    
                    if attempt >= max_attempts:
                        print("⏰ 轮询超时")
                else:
                    print(f"❌ 任务状态异常: {task_status}")
            else:
                print(f"❌ 响应格式错误: {result}")
        else:
            print(f"❌ 请求失败: {response.status_code}")
            print(f"错误信息: {response.text}")
            
    except Exception as e:
        print(f"❌ 请求异常: {e}")

def test_text_to_image():
    """测试通义万相文生图功能"""
    
    # API配置
    api_key = "sk-ade7e6a1728741fcb009dcf1419000de"
    base_url = "https://dashscope.aliyuncs.com"
    endpoint = "/api/v1/services/aigc/text2image/generation"
    model = "wanx2.1-imageedit"
    
    print(f"\n🎨 测试通义万相文生图...")
    
    # 构造请求头
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "X-DashScope-SSE": "disable"
    }
    
    # 构造请求体 - 文生图格式
    data = {
        "model": model,
        "input": {
            "text": "生成一张织机识别图，具有清晰的边缘和结构，适合织机识别"
        },
        "parameters": {
            "style": "realistic",
            "size": "1024*1024",
            "n": 1,
            "seed": 42
        }
    }
    
    print(f"🔍 发送请求到: {base_url}{endpoint}")
    
    try:
        response = requests.post(
            f"{base_url}{endpoint}",
            headers=headers,
            json=data,
            timeout=60
        )
        
        print(f"📊 响应状态码: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ 文生图请求成功")
            
            # 检查是否有图片数据
            if "output" in result and "images" in result["output"]:
                images = result["output"]["images"]
                if images:
                    image_data = images[0]
                    if "url" in image_data:
                        print(f"🎨 生成图片URL: {image_data['url']}")
                        
                        # 下载图片
                        img_response = requests.get(image_data['url'])
                        if img_response.status_code == 200:
                            output_path = "test_composer_text2img.jpg"
                            with open(output_path, "wb") as f:
                                f.write(img_response.content)
                            print(f"✅ 图片已保存: {output_path}")
                            print(f"📏 文件大小: {len(img_response.content)} bytes")
                        else:
                            print(f"❌ 下载图片失败: {img_response.status_code}")
                    else:
                        print(f"❌ 图片数据格式错误: {image_data}")
                else:
                    print("❌ 没有生成图片")
            else:
                print(f"❌ 响应格式错误: {result}")
        else:
            print(f"❌ 请求失败: {response.status_code}")
            print(f"错误信息: {response.text}")
            
    except Exception as e:
        print(f"❌ 请求异常: {e}")

if __name__ == "__main__":
    print("🚀 开始调试通义千问Composer API...")
    test_composer_direct()
    test_text_to_image()
    print("\n✨ 调试完成！") 