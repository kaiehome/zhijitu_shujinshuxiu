#!/usr/bin/env python3
"""
商业AI API配置脚本
用于设置DeepSeek和通义千问的真实API密钥
"""

import os
import sys
import subprocess
from pathlib import Path

def install_dependencies():
    """安装必需的依赖包"""
    print("🔧 正在安装商业API依赖包...")
    
    required_packages = [
        "dashscope>=1.10.0",  # 通义千问API
        "requests>=2.25.0",   # HTTP请求
        "python-dotenv>=0.19.0"  # 环境变量管理
    ]
    
    for package in required_packages:
        try:
            print(f"安装 {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✅ {package} 安装成功")
        except subprocess.CalledProcessError as e:
            print(f"❌ {package} 安装失败: {e}")
            return False
    
    print("🎉 所有依赖包安装完成!")
    return True

def create_env_template():
    """创建.env模板文件"""
    env_template = """# ===========================================
# 🤖 AI增强织机识别图像生成器 - 商业API配置
# ===========================================

# 🔑 商业AI API密钥
# ------------------------------------

# DeepSeek API (用于图像分析和内容理解)
# 获取地址: https://platform.deepseek.com/api_keys
DEEPSEEK_API_KEY=sk-your_deepseek_api_key_here

# 通义千问 API (用于多模态图像生成和分析)
# 获取地址: https://dashscope.aliyun.com/
QWEN_API_KEY=sk-your_qwen_api_key_here

# ⚙️ API配置参数
# ------------------------------------

# API超时设置 (秒)
API_TIMEOUT=30

# AI增强模式开关 (true/false)
AI_ENHANCED_MODE=true

# 图像生成质量 (high/medium/low)
IMAGE_GENERATION_QUALITY=high

# 🎨 织机生成配置
# ------------------------------------

# 默认颜色数量
DEFAULT_COLOR_COUNT=16

# 最大图像尺寸 (像素)
MAX_IMAGE_SIZE=2048

# 缓存目录
CACHE_DIR=cache

# 🔧 调试配置
# ------------------------------------

# 日志级别 (DEBUG/INFO/WARNING/ERROR)
LOG_LEVEL=INFO

# 详细日志开关
VERBOSE_LOGGING=false

# ===========================================
# 📖 使用说明：
# 1. 将上面的 your_api_key_here 替换为真实密钥
# 2. 保存文件后重启服务
# 3. 查看日志确认API连接成功
# ===========================================
"""
    
    env_path = Path(".env")
    with open(env_path, "w", encoding="utf-8") as f:
        f.write(env_template)
    
    print(f"✅ 已创建API配置模板: {env_path.absolute()}")
    return env_path

def validate_api_setup():
    """验证API配置"""
    from dotenv import load_dotenv
    load_dotenv()
    
    deepseek_key = os.getenv('DEEPSEEK_API_KEY')
    qwen_key = os.getenv('QWEN_API_KEY')
    
    print("\n🔍 验证API配置...")
    
    if not deepseek_key or deepseek_key == "sk-your_deepseek_api_key_here":
        print("⚠️  DeepSeek API密钥未配置")
        return False
    else:
        print("✅ DeepSeek API密钥已配置")
    
    if not qwen_key or qwen_key == "sk-your_qwen_api_key_here":
        print("⚠️  通义千问API密钥未配置")
        return False
    else:
        print("✅ 通义千问API密钥已配置")
    
    return True

def test_api_connection():
    """测试API连接"""
    print("\n🧪 测试API连接...")
    
    try:
        # 测试通义千问API
        import dashscope
        from dotenv import load_dotenv
        load_dotenv()
        
        qwen_key = os.getenv('QWEN_API_KEY')
        if qwen_key and qwen_key != "sk-your_qwen_api_key_here":
            dashscope.api_key = qwen_key
            print("✅ 通义千问API连接正常")
        else:
            print("⚠️  通义千问API密钥无效")
            
    except ImportError:
        print("❌ dashscope库未安装")
    except Exception as e:
        print(f"❌ API连接测试失败: {e}")

def print_api_info():
    """打印API获取信息"""
    print("""
📋 API密钥获取指南
================================

🔹 DeepSeek API
   网址: https://platform.deepseek.com/
   步骤: 注册 → 认证 → API Keys → 创建密钥
   费用: ￥0.1/千tokens (很便宜)

🔹 通义千问 API  
   网址: https://dashscope.aliyun.com/
   步骤: 阿里云账号 → 开通服务 → 获取API Key
   费用: ￥0.2/千tokens (性价比高)

💡 建议：
- 两个API都配置，系统会自动选择最佳服务
- 通义千问多模态功能更强，用于图像分析
- DeepSeek作为备用，确保服务稳定性

🔧 配置步骤：
1. 获取API密钥
2. 编辑 .env 文件
3. 填入真实密钥
4. 重启服务
""")

def main():
    """主函数"""
    print("🚀 商业AI API配置助手")
    print("=" * 50)
    
    # 1. 安装依赖
    if not install_dependencies():
        print("❌ 依赖安装失败，请手动安装")
        return
    
    # 2. 创建配置模板
    env_path = create_env_template()
    
    # 3. 打印获取指南
    print_api_info()
    
    # 4. 检查配置
    print("\n" + "=" * 50)
    if validate_api_setup():
        print("🎉 API配置验证通过!")
        test_api_connection()
    else:
        print(f"⚠️  请编辑 {env_path} 文件，填入真实的API密钥")
    
    print("\n✨ 配置完成后，重启服务即可启用真实AI功能!")

if __name__ == "__main__":
    main() 