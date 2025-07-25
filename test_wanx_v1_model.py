#!/usr/bin/env python3
"""
测试wanx-v1模型
"""

import os
import requests
import base64
import json
import time

def test_wanx_v1_model():
    """测试wanx-v1模型"""
    
    api_key = os.getenv('TONGYI_API_KEY')
    if not api_key:
        print("❌ 未设置TONGYI_API_KEY环境变量")
        return False
    
    print("🔍 开始测试wanx-v1模型")
    print("=" * 50)
    
    # API配置
    base_url = "https://dashscope.aliyuncs.com"
    endpoint = "/api/v1/services/aigc/image2image/image-synthesis"
    model = "wanx-v1"  # 使用wanx-v1模型
    
    # 准备测试图像
    test_image_path = "uploads/test_input.png"
    if not os.path.exists(test_image_path):
        print(f"❌ 测试图像不存在: {test_image_path}")
        return False
    
    # 读取并编码图像
    with open(test_image_path, "rb") as f:
        image_data = f.read()
        image_base64 = base64.b64encode(image_data).decode('utf-8')
    
    print(f"✅ 测试图像已加载: {len(image_data)} bytes")
    print(f"📤 使用模型: {model}")
    
    # 准备请求
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "X-DashScope-Async": "enable"
    }
    
    data = {
        "model": model,
        "input": {
            "image": image_base64
        },
        "parameters": {
            "prompt": "生成织机识别图像"
        }
    }
    
    print(f"📤 发送请求到: {base_url}{endpoint}")
    
    try:
        # 发送请求
        response = requests.post(
            f"{base_url}{endpoint}",
            headers=headers,
            json=data,
            timeout=30
        )
        
        print(f"📥 响应状态码: {response.status_code}")
        print(f"📥 响应头: {dict(response.headers)}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ 请求成功: {json.dumps(result, indent=2, ensure_ascii=False)}")
            
            # 检查任务状态
            if "output" in result and "task_id" in result["output"]:
                task_id = result["output"]["task_id"]
                print(f"🔄 任务ID: {task_id}")
                
                # 轮询任务状态
                return poll_task_status(task_id, headers, base_url)
            else:
                print("❌ 响应中没有找到task_id")
                return False
                
        elif response.status_code == 403:
            print("❌ 403错误 - 可能是权限或配额问题")
            print(f"响应内容: {response.text}")
            return False
            
        elif response.status_code == 429:
            print("❌ 429错误 - API配额已用完")
            print(f"响应内容: {response.text}")
            return False
            
        else:
            print(f"❌ 请求失败: {response.status_code}")
            print(f"响应内容: {response.text}")
            return False
            
    except requests.exceptions.Timeout:
        print("❌ 请求超时")
        return False
    except requests.exceptions.RequestException as e:
        print(f"❌ 请求异常: {str(e)}")
        return False
    except Exception as e:
        print(f"❌ 未知错误: {str(e)}")
        return False

def poll_task_status(task_id, headers, base_url, max_attempts=10):
    """轮询任务状态"""
    
    poll_url = f"{base_url}/api/v1/tasks/{task_id}"
    
    for attempt in range(max_attempts):
        print(f"🔄 轮询任务状态 (尝试 {attempt + 1}/{max_attempts})")
        
        try:
            response = requests.get(poll_url, headers=headers, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                print(f"📥 任务状态: {json.dumps(result, indent=2, ensure_ascii=False)}")
                
                if "output" in result:
                    task_status = result["output"].get("task_status", "UNKNOWN")
                    
                    if task_status == "SUCCEEDED":
                        print("✅ 任务成功完成!")
                        
                        # 尝试获取生成的图像
                        if "images" in result["output"]:
                            images = result["output"]["images"]
                            print(f"🎨 生成了 {len(images)} 张图像")
                            
                            for i, image_info in enumerate(images):
                                if "url" in image_info:
                                    print(f"图像 {i+1} URL: {image_info['url']}")
                                elif "base64" in image_info:
                                    print(f"图像 {i+1}: base64数据 (长度: {len(image_info['base64'])})")
                                    
                                    # 保存图像
                                    try:
                                        image_data = base64.b64decode(image_info['base64'])
                                        filename = f"wanx_v1_generated_{int(time.time())}_{i+1}.png"
                                        with open(filename, "wb") as f:
                                            f.write(image_data)
                                        print(f"💾 图像已保存: {filename}")
                                    except Exception as e:
                                        print(f"❌ 保存图像失败: {str(e)}")
                        
                        return True
                    elif task_status == "FAILED":
                        print("❌ 任务失败")
                        error_code = result["output"].get("code", "UNKNOWN")
                        error_message = result["output"].get("message", "UNKNOWN")
                        print(f"错误代码: {error_code}")
                        print(f"错误消息: {error_message}")
                        return False
                    elif task_status in ["PENDING", "RUNNING"]:
                        print(f"⏳ 任务进行中: {task_status}")
                        time.sleep(5)  # 等待5秒后再次轮询
                        continue
                    else:
                        print(f"❓ 未知任务状态: {task_status}")
                        return False
                else:
                    print("❌ 响应中没有output字段")
                    return False
            else:
                print(f"❌ 轮询失败: {response.status_code}")
                print(f"响应内容: {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ 轮询异常: {str(e)}")
            return False
    
    print("❌ 轮询超时")
    return False

def test_text_to_image():
    """测试wanx-v1的文本生成图像功能"""
    
    api_key = os.getenv('TONGYI_API_KEY')
    if not api_key:
        print("❌ 未设置TONGYI_API_KEY环境变量")
        return False
    
    print("\n🔍 测试wanx-v1文本生成图像功能")
    print("=" * 50)
    
    # API配置
    base_url = "https://dashscope.aliyuncs.com"
    endpoint = "/api/v1/services/aigc/text2image/generation"
    model = "wanx-v1"
    
    # 准备请求
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "X-DashScope-Async": "enable"
    }
    
    data = {
        "model": model,
        "input": {
            "prompt": "织机识别图像，传统织锦风格"
        },
        "parameters": {
            "size": "1024*1024",
            "n": 1
        }
    }
    
    print(f"📤 发送文本生成图像请求")
    print(f"📤 提示词: {data['input']['prompt']}")
    
    try:
        response = requests.post(
            f"{base_url}{endpoint}",
            headers=headers,
            json=data,
            timeout=30
        )
        
        print(f"📥 响应状态码: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ 请求成功: {json.dumps(result, indent=2, ensure_ascii=False)}")
            
            if "output" in result and "task_id" in result["output"]:
                task_id = result["output"]["task_id"]
                print(f"🔄 任务ID: {task_id}")
                
                # 轮询任务状态
                return poll_task_status(task_id, headers, base_url)
            else:
                print("❌ 响应中没有找到task_id")
                return False
        else:
            print(f"❌ 请求失败: {response.status_code}")
            print(f"响应内容: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ 请求异常: {str(e)}")
        return False

if __name__ == "__main__":
    print("🔍 开始测试wanx-v1模型")
    print("=" * 60)
    
    # 测试图生图功能
    print("📋 测试1: 图生图功能")
    success1 = test_wanx_v1_model()
    
    # 测试文本生成图像功能
    print("\n📋 测试2: 文本生成图像功能")
    success2 = test_text_to_image()
    
    print("\n" + "=" * 60)
    print("📊 测试结果总结:")
    print(f"图生图功能: {'✅ 成功' if success1 else '❌ 失败'}")
    print(f"文本生成图像: {'✅ 成功' if success2 else '❌ 失败'}")
    
    if success1 or success2:
        print("\n🎉 wanx-v1模型测试成功！")
        print("💡 建议: 可以考虑使用wanx-v1作为备选模型")
    else:
        print("\n❌ wanx-v1模型测试失败")
        print("💡 建议: 继续使用本地AI处理或联系技术支持") 