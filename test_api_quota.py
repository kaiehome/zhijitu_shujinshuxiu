#!/usr/bin/env python3
"""
测试通义千问Composer API的配额和权限
"""

import os
import requests
import base64
import json
import time

def test_api_quota():
    """测试API配额和权限"""
    
    # 获取API Key
    api_key = os.getenv('TONGYI_API_KEY')
    if not api_key:
        print("❌ 未设置TONGYI_API_KEY环境变量")
        return False
    
    print(f"✅ API Key已设置: {api_key[:10]}...")
    
    # API配置
    base_url = "https://dashscope.aliyuncs.com"
    endpoint = "/api/v1/services/aigc/image2image/image-synthesis"
    model = "wanx2.1-imageedit"
    
    # 准备一个简单的测试图像
    test_image_path = "uploads/test_input.png"
    if not os.path.exists(test_image_path):
        print(f"❌ 测试图像不存在: {test_image_path}")
        return False
    
    # 读取并编码图像
    with open(test_image_path, "rb") as f:
        image_data = f.read()
        image_base64 = base64.b64encode(image_data).decode('utf-8')
    
    print(f"✅ 测试图像已加载: {len(image_data)} bytes")
    
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
    print(f"📤 使用模型: {model}")
    
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
                        return True
                    elif task_status == "FAILED":
                        print("❌ 任务失败")
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

if __name__ == "__main__":
    print("🔍 开始测试通义千问Composer API配额和权限")
    print("=" * 50)
    
    success = test_api_quota()
    
    print("=" * 50)
    if success:
        print("✅ 测试完成 - API工作正常")
    else:
        print("❌ 测试失败 - 请检查API配额和权限") 