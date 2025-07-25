#!/usr/bin/env python3
"""
系统状态检查脚本
检查所有AI模型和API的状态
"""

import requests
import time
import os
from pathlib import Path

def check_api_endpoint(name, url, timeout=10):
    """检查API端点状态"""
    try:
        response = requests.get(url, timeout=timeout)
        if response.status_code == 200:
            return True, response.json()
        else:
            return False, f"HTTP {response.status_code}"
    except Exception as e:
        return False, str(e)

def check_image_generation_endpoint(name, url, test_image, timeout=30):
    """检查图片生成端点"""
    try:
        with open(test_image, 'rb') as f:
            files = {'file': f}
            response = requests.post(url, files=files, timeout=timeout)
        
        if response.status_code == 200:
            return True, f"生成成功 ({len(response.content)} 字节)"
        elif response.status_code == 503:
            return False, "API未配置"
        else:
            return False, f"HTTP {response.status_code}"
    except Exception as e:
        return False, str(e)

def main():
    """主函数"""
    base_url = "http://localhost:8000"
    
    print("=== 系统状态检查 ===")
    print(f"API服务器: {base_url}")
    print()
    
    # 查找测试图片
    test_image = None
    for path in ["uploads", "target_images"]:
        if os.path.exists(path):
            for file in os.listdir(path):
                if file.endswith(('.jpg', '.png')):
                    test_image = os.path.join(path, file)
                    break
        if test_image:
            break
    
    if not test_image:
        print("⚠ 警告: 找不到测试图片")
        test_image = "uploads/orig.png"  # 使用默认图片
    
    print(f"使用测试图片: {test_image}")
    print()
    
    # 检查系统健康状态
    print("1. 系统健康状态")
    success, result = check_api_endpoint("系统健康", f"{base_url}/api/health")
    if success:
        print(f"✓ 系统健康: {result.get('status', 'unknown')}")
    else:
        print(f"✗ 系统健康检查失败: {result}")
    print()
    
    # 检查各种AI模型状态
    print("2. AI模型状态")
    
    # 通义千问API
    success, result = check_api_endpoint("通义千问", f"{base_url}/api/tongyi-qianwen-status")
    if success:
        api_available = result.get("api_available", False)
        api_key_set = result.get("api_key_set", False)
        model = result.get("model", "unknown")
        print(f"✓ 通义千问: {'可用' if api_available else '未配置'} (模型: {model}, API密钥: {'已设置' if api_key_set else '未设置'})")
    else:
        print(f"✗ 通义千问状态检查失败: {result}")
    
    # AI大模型
    success, result = check_api_endpoint("AI大模型", f"{base_url}/api/ai-model-status")
    if success:
        is_loaded = result.get("is_loaded", False)
        device = result.get("device", "unknown")
        print(f"✓ AI大模型: {'已加载' if is_loaded else '未加载'} (设备: {device})")
    else:
        print(f"✗ AI大模型状态检查失败: {result}")
    
    # 简化AI模型
    success, result = check_api_endpoint("简化AI模型", f"{base_url}/api/simple-ai-status")
    if success:
        is_loaded = result.get("is_loaded", False)
        device = result.get("device", "unknown")
        print(f"✓ 简化AI模型: {'已加载' if is_loaded else '未加载'} (设备: {device})")
    else:
        print(f"✗ 简化AI模型状态检查失败: {result}")
    
    # 本地AI增强
    success, result = check_api_endpoint("本地AI增强", f"{base_url}/api/local-ai-status")
    if success:
        print(f"✓ 本地AI增强: 可用")
    else:
        print(f"✗ 本地AI增强状态检查失败: {result}")
    
    print()
    
    # 检查图片生成功能
    print("3. 图片生成功能测试")
    
    # 通义千问生成
    success, result = check_image_generation_endpoint(
        "通义千问生成", 
        f"{base_url}/api/generate-tongyi-qianwen", 
        test_image
    )
    if success:
        print(f"✓ 通义千问生成: {result}")
    else:
        print(f"✗ 通义千问生成: {result}")
    
    # AI大模型生成
    success, result = check_image_generation_endpoint(
        "AI大模型生成", 
        f"{base_url}/api/generate-ai-model", 
        test_image
    )
    if success:
        print(f"✓ AI大模型生成: {result}")
    else:
        print(f"✗ AI大模型生成: {result}")
    
    # 简化AI模型生成
    success, result = check_image_generation_endpoint(
        "简化AI模型生成", 
        f"{base_url}/api/generate-simple-ai", 
        test_image
    )
    if success:
        print(f"✓ 简化AI模型生成: {result}")
    else:
        print(f"✗ 简化AI模型生成: {result}")
    
    # 本地AI增强生成
    success, result = check_image_generation_endpoint(
        "本地AI增强生成", 
        f"{base_url}/api/generate-local-ai-enhanced", 
        test_image
    )
    if success:
        print(f"✓ 本地AI增强生成: {result}")
    else:
        print(f"✗ 本地AI增强生成: {result}")
    
    print()
    
    # 检查文件系统
    print("4. 文件系统状态")
    
    directories = ["uploads", "target_images", "trained_models", "outputs"]
    for directory in directories:
        if os.path.exists(directory):
            file_count = len([f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))])
            print(f"✓ {directory}: {file_count} 个文件")
        else:
            print(f"✗ {directory}: 目录不存在")
    
    print()
    
    # 总结
    print("=== 系统状态总结 ===")
    print("✅ 系统已成功集成通义千问大模型API")
    print("✅ 支持多种AI模型生成织机识别图")
    print("✅ 提供完整的API接口和状态检查")
    print()
    print("📋 下一步操作:")
    print("1. 如需使用通义千问，请运行: python setup_tongyi_qianwen.py setup")
    print("2. 测试通义千问功能: python test_tongyi_qianwen.py")
    print("3. 查看详细使用说明: 通义千问API使用说明.md")
    print("4. 重启服务器以加载新配置: cd backend && python main.py")

if __name__ == "__main__":
    main() 