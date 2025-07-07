#!/usr/bin/env python3
"""
验证颜色数量范围更新脚本
确认所有相关配置都已从6-12更新为10-20
"""

import json
import requests
import time
from pathlib import Path

def test_api_validation():
    """测试API颜色数量验证"""
    print("🧪 测试API颜色数量验证...")
    
    # 等待后端启动
    for i in range(5):
        try:
            response = requests.get("http://localhost:8000/api/health", timeout=2)
            if response.status_code == 200:
                break
        except:
            print(f"   等待后端启动... ({i+1}/5)")
            time.sleep(2)
    else:
        print("   ❌ 后端服务未响应")
        return False
    
    test_cases = [
        (5, "应该拒绝 - 低于最小值10"),
        (9, "应该拒绝 - 低于最小值10"), 
        (10, "应该接受 - 最小有效值"),
        (15, "应该接受 - 默认值"),
        (20, "应该接受 - 最大有效值"),
        (21, "应该拒绝 - 高于最大值20"),
        (25, "应该拒绝 - 高于最大值20")
    ]
    
    passed = 0
    total = len(test_cases)
    
    for color_count, expected in test_cases:
        try:
            response = requests.post(
                "http://localhost:8000/api/process",
                json={
                    "filename": "test.png",
                    "color_count": color_count,
                    "edge_enhancement": True,
                    "noise_reduction": True
                },
                timeout=5
            )
            
            if color_count < 10 or color_count > 20:
                # 应该被拒绝
                if response.status_code == 422:
                    print(f"   ✅ 颜色数量 {color_count}: {expected}")
                    passed += 1
                else:
                    print(f"   ❌ 颜色数量 {color_count}: 应该被拒绝但被接受了")
            else:
                # 应该被接受（可能因为文件不存在而有其他错误）
                if response.status_code in [422, 404, 400]:
                    if "greater than or equal to" not in str(response.content) and \
                       "less than or equal to" not in str(response.content):
                        print(f"   ✅ 颜色数量 {color_count}: {expected}")
                        passed += 1
                    else:
                        print(f"   ❌ 颜色数量 {color_count}: 范围验证错误")
                else:
                    print(f"   ✅ 颜色数量 {color_count}: {expected}")
                    passed += 1
                    
        except Exception as e:
            print(f"   ❌ 颜色数量 {color_count}: 请求失败 - {e}")
    
    print(f"\n🏆 API验证结果: {passed}/{total} 通过")
    return passed == total

def check_config_files():
    """检查配置文件中的颜色范围"""
    print("\n📁 检查配置文件...")
    
    checks = []
    
    # 检查 models.py
    models_file = Path("models.py")
    if models_file.exists():
        content = models_file.read_text()
        if "ge=10, le=20" in content:
            checks.append("✅ models.py: 颜色范围 10-20")
        else:
            checks.append("❌ models.py: 颜色范围未更新")
        
        if "DEFAULT_COLOR_COUNT = 15" in content:
            checks.append("✅ models.py: 默认值 15")
        else:
            checks.append("❌ models.py: 默认值未更新")
    else:
        checks.append("❌ models.py: 文件不存在")
    
    # 检查 main.py
    main_file = Path("main.py")
    if main_file.exists():
        content = main_file.read_text()
        if "10 <= request.color_count <= 20" in content:
            checks.append("✅ main.py: 验证逻辑 10-20")
        else:
            checks.append("❌ main.py: 验证逻辑未更新")
    else:
        checks.append("❌ main.py: 文件不存在")
    
    # 检查 image_processor.py
    processor_file = Path("image_processor.py")
    if processor_file.exists():
        content = processor_file.read_text()
        if "10 <= color_count <= 20" in content:
            checks.append("✅ image_processor.py: 验证逻辑 10-20")
        else:
            checks.append("❌ image_processor.py: 验证逻辑未更新")
            
        if "color_count: int = 15" in content:
            checks.append("✅ image_processor.py: 默认值 15")
        else:
            checks.append("❌ image_processor.py: 默认值未更新")
    else:
        checks.append("❌ image_processor.py: 文件不存在")
    
    # 检查前端文件
    frontend_index = Path("../frontend/src/pages/index.tsx")
    if frontend_index.exists():
        content = frontend_index.read_text()
        if "colorCount: 15" in content:
            checks.append("✅ frontend/index.tsx: 默认值 15")
        else:
            checks.append("❌ frontend/index.tsx: 默认值未更新")
    else:
        checks.append("❌ frontend/index.tsx: 文件不存在")
    
    frontend_process = Path("../frontend/src/components/ProcessSection.tsx")
    if frontend_process.exists():
        content = frontend_process.read_text()
        if "min={10}" in content and "max={20}" in content:
            checks.append("✅ frontend/ProcessSection.tsx: 滑块范围 10-20")
        else:
            checks.append("❌ frontend/ProcessSection.tsx: 滑块范围未更新")
    else:
        checks.append("❌ frontend/ProcessSection.tsx: 文件不存在")
    
    for check in checks:
        print(f"   {check}")
    
    passed = len([c for c in checks if c.startswith("✅")])
    total = len(checks)
    print(f"\n🏆 配置检查结果: {passed}/{total} 通过")
    return passed == total

def main():
    """主验证流程"""
    print("🔍 颜色数量范围更新验证")
    print("=" * 50)
    print("🎯 目标: 将颜色数量范围从 6-12 更新为 10-20")
    print("🎯 默认值: 从 8 更新为 15")
    print("=" * 50)
    
    # 检查配置文件
    config_ok = check_config_files()
    
    # 测试API验证
    api_ok = test_api_validation()
    
    print("\n" + "=" * 50)
    if config_ok and api_ok:
        print("🎉 所有验证通过！颜色数量范围已成功更新为 10-20")
        print("💡 现在用户可以选择 10-20 种颜色进行处理")
        print("💡 默认值为 15 种颜色，提供更丰富的色彩层次")
    else:
        print("⚠️ 部分验证失败，请检查上述错误")
        if not config_ok:
            print("   - 配置文件需要进一步更新")
        if not api_ok:
            print("   - API验证存在问题")
    
    print("=" * 50)

if __name__ == "__main__":
    main() 