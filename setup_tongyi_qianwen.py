#!/usr/bin/env python3
"""
通义千问API快速设置脚本
"""

import os
import sys
import getpass

def setup_tongyi_api():
    """设置通义千问API"""
    print("=== 通义千问API设置向导 ===")
    print()
    
    # 检查当前环境变量
    current_key = os.getenv("TONGYI_API_KEY")
    if current_key:
        print(f"当前已设置API密钥: {current_key[:8]}...")
        choice = input("是否要重新设置? (y/N): ").lower()
        if choice != 'y':
            print("保持当前设置")
            return
    
    print("请按照以下步骤获取通义千问API密钥：")
    print("1. 访问 https://dashscope.console.aliyun.com/")
    print("2. 注册/登录阿里云账号")
    print("3. 开通通义千问服务")
    print("4. 创建API密钥")
    print()
    
    # 获取API密钥
    api_key = getpass.getpass("请输入您的通义千问API密钥: ").strip()
    
    if not api_key:
        print("错误: API密钥不能为空")
        return
    
    # 验证API密钥格式
    if len(api_key) < 20:
        print("警告: API密钥长度似乎过短，请确认是否正确")
        choice = input("是否继续? (y/N): ").lower()
        if choice != 'y':
            return
    
    # 设置环境变量
    os.environ["TONGYI_API_KEY"] = api_key
    
    # 创建环境变量文件
    env_file = ".env"
    with open(env_file, "w") as f:
        f.write(f"TONGYI_API_KEY={api_key}\n")
    
    print(f"✓ API密钥已设置并保存到 {env_file}")
    print()
    
    # 测试API连接
    print("正在测试API连接...")
    try:
        import requests
        response = requests.get("http://localhost:8000/api/tongyi-qianwen-status", timeout=10)
        if response.status_code == 200:
            status = response.json()
            if status.get("api_available"):
                print("✓ 通义千问API连接成功！")
            else:
                print("⚠ API密钥可能无效，请检查")
        else:
            print("⚠ 无法连接到API服务器")
    except Exception as e:
        print(f"⚠ 测试失败: {e}")
    
    print()
    print("设置完成！")
    print("现在您可以：")
    print("1. 重启API服务器以加载新的API密钥")
    print("2. 运行 python test_tongyi_qianwen.py 测试功能")
    print("3. 使用通义千问生成织机识别图")

def check_api_status():
    """检查API状态"""
    print("=== 检查通义千问API状态 ===")
    
    try:
        import requests
        response = requests.get("http://localhost:8000/api/tongyi-qianwen-status", timeout=10)
        if response.status_code == 200:
            status = response.json()
            print(f"API可用: {status.get('api_available', False)}")
            print(f"模型: {status.get('model', 'unknown')}")
            print(f"API密钥已设置: {status.get('api_key_set', False)}")
            print(f"基础URL: {status.get('base_url', 'unknown')}")
        else:
            print(f"API服务器响应异常: {response.status_code}")
    except Exception as e:
        print(f"无法连接到API服务器: {e}")

def main():
    """主函数"""
    if len(sys.argv) > 1:
        command = sys.argv[1]
        if command == "status":
            check_api_status()
        elif command == "setup":
            setup_tongyi_api()
        else:
            print("用法: python setup_tongyi_qianwen.py [setup|status]")
    else:
        print("通义千问API设置工具")
        print()
        print("可用命令:")
        print("  setup   - 设置API密钥")
        print("  status  - 检查API状态")
        print()
        choice = input("请选择操作 (setup/status): ").lower()
        if choice == "setup":
            setup_tongyi_api()
        elif choice == "status":
            check_api_status()
        else:
            print("无效选择")

if __name__ == "__main__":
    main() 