#!/usr/bin/env python3
"""
验证离散颜色数量配置脚本
确认只允许 10, 12, 14, 16, 18, 20 这6个特定值
"""

import json
import requests
import time
from pathlib import Path

def test_discrete_color_validation():
    """测试离散颜色数量验证"""
    print("🧪 测试离散颜色数量验证...")
    
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
    
    # 测试用例: [颜色数量, 预期结果, 描述]
    test_cases = [
        (9, False, "应该拒绝 - 9不在允许列表中"),
        (10, True, "应该接受 - 允许值"),
        (11, False, "应该拒绝 - 11不在允许列表中"),
        (12, True, "应该接受 - 允许值"),
        (13, False, "应该拒绝 - 13不在允许列表中"),
        (14, True, "应该接受 - 允许值"),
        (15, False, "应该拒绝 - 15不在允许列表中"),
        (16, True, "应该接受 - 允许值(默认)"),
        (17, False, "应该拒绝 - 17不在允许列表中"),
        (18, True, "应该接受 - 允许值"),
        (19, False, "应该拒绝 - 19不在允许列表中"),
        (20, True, "应该接受 - 允许值"),
        (21, False, "应该拒绝 - 21不在允许列表中"),
    ]
    
    passed = 0
    total = len(test_cases)
    
    for color_count, should_accept, description in test_cases:
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
            
            if should_accept:
                # 应该被接受（可能因为文件不存在而有其他错误）
                if response.status_code in [422, 404, 400]:
                    response_text = str(response.content)
                    if "颜色数量必须是以下值之一" not in response_text:
                        print(f"   ✅ 颜色数量 {color_count}: {description}")
                        passed += 1
                    else:
                        print(f"   ❌ 颜色数量 {color_count}: 应该被接受但被颜色验证拒绝")
                else:
                    print(f"   ✅ 颜色数量 {color_count}: {description}")
                    passed += 1
            else:
                # 应该被拒绝
                if response.status_code == 422:
                    response_text = str(response.content)
                    if "颜色数量必须是以下值之一" in response_text:
                        print(f"   ✅ 颜色数量 {color_count}: {description}")
                        passed += 1
                    else:
                        print(f"   ❌ 颜色数量 {color_count}: 被拒绝但错误信息不正确")
                        print(f"       响应: {response_text[:100]}...")
                else:
                    print(f"   ❌ 颜色数量 {color_count}: 应该被拒绝但被接受了")
                    
        except Exception as e:
            print(f"   ❌ 颜色数量 {color_count}: 请求失败 - {e}")
    
    print(f"\n🏆 API验证结果: {passed}/{total} 通过")
    return passed == total

def main():
    """主验证流程"""
    print("🎯 离散颜色数量配置验证")
    print("=" * 60)
    print("🎨 允许的颜色数量: 10, 12, 14, 16, 18, 20")
    print("🎨 默认值: 16")
    print("=" * 60)
    
    # 测试API验证
    api_ok = test_discrete_color_validation()
    
    print("\n" + "=" * 60)
    if api_ok:
        print("🎉 离散颜色数量验证通过！")
        print("💡 用户现在只能选择 10, 12, 14, 16, 18, 20 这6个颜色数量")
    else:
        print("⚠️ API验证存在问题")
    
    print("=" * 60)

if __name__ == "__main__":
    main() 