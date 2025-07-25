#!/usr/bin/env python3
"""
分析通义千问Composer API错误
"""

import os
import requests
import base64
import json
import time

def analyze_api_error():
    """分析API错误"""
    
    api_key = os.getenv('TONGYI_API_KEY')
    if not api_key:
        print("❌ 未设置TONGYI_API_KEY环境变量")
        return
    
    print("🔍 开始分析通义千问Composer API错误")
    print("=" * 60)
    
    # API配置
    base_url = "https://dashscope.aliyuncs.com"
    endpoint = "/api/v1/services/aigc/image2image/image-synthesis"
    model = "wanx2.1-imageedit"
    
    # 测试不同的图像和参数
    test_cases = [
        {
            "name": "测试1: 最小图像",
            "image_path": "uploads/test_input.png",
            "prompt": "生成织机识别图像",
            "description": "使用标准测试图像"
        },
        {
            "name": "测试2: 简单提示",
            "image_path": "uploads/test_input.png", 
            "prompt": "织机",
            "description": "使用最简单的提示词"
        },
        {
            "name": "测试3: 英文提示",
            "image_path": "uploads/test_input.png",
            "prompt": "loom recognition image",
            "description": "使用英文提示词"
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n📋 {test_case['name']}")
        print(f"描述: {test_case['description']}")
        print(f"提示词: {test_case['prompt']}")
        
        # 检查图像文件
        if not os.path.exists(test_case['image_path']):
            print(f"❌ 图像文件不存在: {test_case['image_path']}")
            continue
        
        # 读取图像
        with open(test_case['image_path'], "rb") as f:
            image_data = f.read()
            image_base64 = base64.b64encode(image_data).decode('utf-8')
        
        print(f"✅ 图像大小: {len(image_data)} bytes")
        
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
                "prompt": test_case['prompt']
            }
        }
        
        try:
            # 发送请求
            print("📤 发送请求...")
            response = requests.post(
                f"{base_url}{endpoint}",
                headers=headers,
                json=data,
                timeout=30
            )
            
            print(f"📥 响应状态码: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ 任务提交成功")
                print(f"请求ID: {result.get('request_id', 'N/A')}")
                
                if "output" in result and "task_id" in result["output"]:
                    task_id = result["output"]["task_id"]
                    print(f"任务ID: {task_id}")
                    
                    # 轮询任务状态
                    error_info = poll_task_for_error(task_id, headers, base_url)
                    if error_info:
                        print(f"❌ 错误代码: {error_info.get('code', 'N/A')}")
                        print(f"❌ 错误消息: {error_info.get('message', 'N/A')}")
                        print(f"❌ 提交时间: {error_info.get('submit_time', 'N/A')}")
                        print(f"❌ 结束时间: {error_info.get('end_time', 'N/A')}")
                        
                        # 分析错误类型
                        analyze_error_type(error_info)
                    else:
                        print("✅ 任务成功完成")
                else:
                    print("❌ 响应格式异常")
                    print(f"响应内容: {json.dumps(result, indent=2, ensure_ascii=False)}")
            else:
                print(f"❌ 请求失败: {response.status_code}")
                print(f"响应内容: {response.text}")
                
        except Exception as e:
            print(f"❌ 请求异常: {str(e)}")
        
        print("-" * 40)
        
        # 在测试之间稍作延迟
        if i < len(test_cases):
            time.sleep(2)

def poll_task_for_error(task_id, headers, base_url, max_attempts=5):
    """轮询任务状态并返回错误信息"""
    
    poll_url = f"{base_url}/api/v1/tasks/{task_id}"
    
    for attempt in range(max_attempts):
        try:
            response = requests.get(poll_url, headers=headers, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                
                if "output" in result:
                    task_status = result["output"].get("task_status", "UNKNOWN")
                    
                    if task_status == "SUCCEEDED":
                        return None  # 成功，无错误
                    elif task_status == "FAILED":
                        # 返回错误信息
                        return {
                            "code": result["output"].get("code"),
                            "message": result["output"].get("message"),
                            "submit_time": result["output"].get("submit_time"),
                            "end_time": result["output"].get("end_time")
                        }
                    elif task_status in ["PENDING", "RUNNING"]:
                        print(f"⏳ 任务进行中: {task_status}")
                        time.sleep(3)
                        continue
                    else:
                        print(f"❓ 未知状态: {task_status}")
                        return None
                else:
                    print("❌ 响应格式异常")
                    return None
            else:
                print(f"❌ 轮询失败: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"❌ 轮询异常: {str(e)}")
            return None
    
    print("❌ 轮询超时")
    return None

def analyze_error_type(error_info):
    """分析错误类型"""
    code = error_info.get('code', '')
    message = error_info.get('message', '')
    
    print("\n🔍 错误分析:")
    
    if 'InternalError' in code:
        print("📋 错误类型: 服务器内部错误")
        print("💡 可能原因:")
        print("   - 服务器端临时故障")
        print("   - 模型服务不可用")
        print("   - 资源不足")
        print("   - 系统维护")
        
    elif 'QuotaExceeded' in code:
        print("📋 错误类型: 配额超限")
        print("💡 解决方案:")
        print("   - 检查API配额使用情况")
        print("   - 升级服务套餐")
        print("   - 等待配额重置")
        
    elif 'PermissionDenied' in code:
        print("📋 错误类型: 权限不足")
        print("💡 解决方案:")
        print("   - 检查API Key权限")
        print("   - 确认模型访问权限")
        print("   - 联系技术支持")
        
    elif 'InvalidParameter' in code:
        print("📋 错误类型: 参数错误")
        print("💡 解决方案:")
        print("   - 检查请求参数格式")
        print("   - 验证图像格式")
        print("   - 确认提示词内容")
        
    else:
        print(f"📋 未知错误类型: {code}")
        print("💡 建议:")
        print("   - 查看官方文档")
        print("   - 联系技术支持")
        print("   - 检查服务状态")

def check_api_status():
    """检查API服务状态"""
    print("\n🌐 检查API服务状态")
    print("=" * 40)
    
    try:
        # 检查DashScope服务状态
        response = requests.get("https://dashscope.aliyuncs.com", timeout=10)
        print(f"✅ DashScope服务可访问: {response.status_code}")
    except Exception as e:
        print(f"❌ DashScope服务不可访问: {str(e)}")
    
    # 检查模型可用性
    api_key = os.getenv('TONGYI_API_KEY')
    if api_key:
        headers = {"Authorization": f"Bearer {api_key}"}
        try:
            # 尝试获取模型列表
            response = requests.get(
                "https://dashscope.aliyuncs.com/api/v1/models",
                headers=headers,
                timeout=10
            )
            if response.status_code == 200:
                print("✅ 模型列表API可访问")
            else:
                print(f"⚠️ 模型列表API响应异常: {response.status_code}")
        except Exception as e:
            print(f"❌ 模型列表API不可访问: {str(e)}")

if __name__ == "__main__":
    analyze_api_error()
    check_api_status()
    
    print("\n" + "=" * 60)
    print("📋 总结建议:")
    print("1. 当前问题是服务器端内部错误，不是客户端问题")
    print("2. 建议联系Alibaba Cloud技术支持")
    print("3. 可以尝试使用其他模型或等待服务恢复")
    print("4. 考虑使用本地AI处理作为备选方案") 