#!/bin/bash

# 刺绣图像处理API服务器启动脚本

echo "🚀 启动刺绣图像处理API服务器..."

# 检查Python环境
if ! command -v python &> /dev/null; then
    echo "❌ Python未安装或不在PATH中"
    exit 1
fi

# 检查依赖
echo "📦 检查依赖..."
python -c "import fastapi, uvicorn" 2>/dev/null || {
    echo "❌ 缺少依赖，请运行: pip install -r requirements.txt"
    exit 1
}

# 检查端口是否被占用
PORT=8000
if lsof -Pi :$PORT -sTCP:LISTEN -t >/dev/null ; then
    echo "⚠️  端口 $PORT 已被占用，尝试停止现有进程..."
    lsof -ti:$PORT | xargs kill -9 2>/dev/null
    sleep 2
fi

# 启动服务器
echo "🌐 启动服务器在 http://127.0.0.1:$PORT"
echo "📊 API文档: http://127.0.0.1:$PORT/docs"
echo "🔍 健康检查: http://127.0.0.1:$PORT/health"
echo ""
echo "按 Ctrl+C 停止服务器"
echo ""

uvicorn api_server:app --host 127.0.0.1 --port $PORT --reload --log-level info 