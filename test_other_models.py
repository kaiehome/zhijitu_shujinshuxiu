#!/usr/bin/env python3
"""
测试其他可能的图生图模型
"""

import os
import requests
import base64
import json
import time

def test_model(model_name, endpoint, is_image_to_image=True):
    """测试指定模型"""
    
    api_key = os.getenv('TONGYI_API_KEY')
    if not api_key:
        print("❌ 未设置TONGYI_API_KEY环境变量")
        return False
    
    print(f"🔍 测试模型: {model_name}")
    print(f"📤 端点: {endpoint}")
    print("=" * 50)
    
    # API配置
    base_url = "https://dashscope.aliyuncs.com"
    
    # 准备请求头
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "X-DashScope-Async": "enable"
    }
    
    if is_image_to_image:
        # 图生图请求
        test_image_path = "uploads/test_input.png"
        if not os.path.exists(test_image_path):
            print(f"❌ 测试图像不存在: {test_image_path}")
            return False
        
        # 读取并编码图像
        with open(test_image_path, "rb") as f:
            image_data = f.read()
            image_base64 = base64.b64encode(image_data).decode('utf-8')
        
        print(f"✅ 测试图像已加载: {len(image_data)} bytes")
        
        data = {
            "model": model_name,
            "input": {
                "image": image_base64
            },
            "parameters": {
                "prompt": "生成织机识别图像"
            }
        }
    else:
        # 文本生成图像请求
        data = {
            "model": model_name,
            "input": {
                "prompt": "织机识别图像，传统织锦风格"
            },
            "parameters": {
                "size": "1024*1024",
                "n": 1
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
                
        elif response.status_code == 400:
            error_text = response.text
            print(f"❌ 400错误: {error_text}")
            
            if "url error" in error_text.lower():
                print("💡 提示: 此模型可能不支持此端点")
            elif "invalid parameter" in error_text.lower():
                print("💡 提示: 请求参数可能不正确")
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
            
    except Exception as e:
        print(f"❌ 请求异常: {str(e)}")
        return False

def poll_task_status(task_id, headers, base_url, max_attempts=5):
    """轮询任务状态"""
    
    poll_url = f"{base_url}/api/v1/tasks/{task_id}"
    
    for attempt in range(max_attempts):
        print(f"🔄 轮询任务状态 (尝试 {attempt + 1}/{max_attempts})")
        
        try:
            response = requests.get(poll_url, headers=headers, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                
                if "output" in result:
                    task_status = result["output"].get("task_status", "UNKNOWN")
                    
                    if task_status == "SUCCEEDED":
                        print("✅ 任务成功完成!")
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
                        time.sleep(3)
                        continue
                    else:
                        print(f"❓ 未知任务状态: {task_status}")
                        return False
                else:
                    print("❌ 响应中没有output字段")
                    return False
            else:
                print(f"❌ 轮询失败: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ 轮询异常: {str(e)}")
            return False
    
    print("❌ 轮询超时")
    return False

def get_available_models():
    """获取可用的模型列表"""
    
    api_key = os.getenv('TONGYI_API_KEY')
    if not api_key:
        print("❌ 未设置TONGYI_API_KEY环境变量")
        return []
    
    print("🔍 获取可用模型列表")
    print("=" * 40)
    
    headers = {"Authorization": f"Bearer {api_key}"}
    
    try:
        response = requests.get(
            "https://dashscope.aliyuncs.com/api/v1/models",
            headers=headers,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ 成功获取模型列表")
            
            models = []
            if "data" in result:
                for model in result["data"]:
                    model_id = model.get("id", "")
                    model_name = model.get("name", "")
                    model_type = model.get("type", "")
                    
                    print(f"📋 模型ID: {model_id}")
                    print(f"   名称: {model_name}")
                    print(f"   类型: {model_type}")
                    print("-" * 30)
                    
                    models.append({
                        "id": model_id,
                        "name": model_name,
                        "type": model_type
                    })
            
            return models
        else:
            print(f"❌ 获取模型列表失败: {response.status_code}")
            print(f"响应内容: {response.text}")
            return []
            
    except Exception as e:
        print(f"❌ 获取模型列表异常: {str(e)}")
        return []

def main():
    """主函数"""
    
    print("🔍 开始测试其他图生图模型")
    print("=" * 60)
    
    # 获取可用模型列表
    models = get_available_models()
    
    # 定义要测试的模型和端点
    test_cases = [
        # 图生图模型
        {
            "name": "wanx2.1-imageedit",
            "endpoint": "/api/v1/services/aigc/image2image/image-synthesis",
            "is_image_to_image": True,
            "description": "官方图生图模型"
        },
        {
            "name": "wanx-v1",
            "endpoint": "/api/v1/services/aigc/image2image/image-synthesis",
            "is_image_to_image": True,
            "description": "万相v1模型"
        },
        {
            "name": "wanx-v1",
            "endpoint": "/api/v1/services/aigc/text2image/generation",
            "is_image_to_image": False,
            "description": "万相v1文本生成图像"
        },
        # 可以添加更多模型测试
    ]
    
    results = {}
    
    for test_case in test_cases:
        print(f"\n📋 测试: {test_case['description']}")
        print(f"模型: {test_case['name']}")
        print(f"端点: {test_case['endpoint']}")
        
        success = test_model(
            test_case['name'],
            test_case['endpoint'],
            test_case['is_image_to_image']
        )
        
        results[f"{test_case['name']}_{test_case['endpoint']}"] = {
            "success": success,
            "description": test_case['description']
        }
        
        print("-" * 60)
        time.sleep(2)  # 避免请求过于频繁
    
    # 总结结果
    print("\n" + "=" * 60)
    print("📊 测试结果总结:")
    
    for key, result in results.items():
        status = "✅ 成功" if result["success"] else "❌ 失败"
        print(f"{result['description']}: {status}")
    
    # 建议
    successful_models = [k for k, v in results.items() if v["success"]]
    
    if successful_models:
        print(f"\n🎉 找到 {len(successful_models)} 个可用模型!")
        print("💡 建议: 使用成功的模型作为备选方案")
    else:
        print("\n❌ 所有模型测试都失败了")
        print("💡 建议:")
        print("   1. 联系Alibaba Cloud技术支持")
        print("   2. 使用本地AI处理")
        print("   3. 等待服务恢复")

if __name__ == "__main__":
    main() 