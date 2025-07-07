#!/bin/bash

# 蜀锦蜀绣AI打样图生成工具 - 完整应用启动脚本

echo "🧵 启动蜀锦蜀绣AI打样图生成工具完整应用..."
echo ""

# 检查依赖
echo "🔍 检查系统依赖..."

# 检查Python
if ! command -v python3 &> /dev/null; then
    echo "❌ 错误: Python3 未安装"
    exit 1
fi

# 检查Node.js
if ! command -v node &> /dev/null; then
    echo "❌ 错误: Node.js 未安装"
    exit 1
fi

echo "✅ Python版本: $(python3 --version)"
echo "✅ Node.js版本: $(node --version)"
echo ""

# 创建日志目录
mkdir -p logs

# 启动后端（后台运行）
echo "🚀 启动后端服务..."
cd backend

# 创建虚拟环境和安装依赖
if [ ! -d "venv" ]; then
    echo "📦 创建Python虚拟环境..."
    python3 -m venv venv
fi

source venv/bin/activate
pip install --upgrade pip > /dev/null 2>&1
pip install -r requirements.txt > /dev/null 2>&1

# 创建必要目录
mkdir -p uploads outputs

# 复制环境配置
if [ ! -f ".env" ]; then
    if [ -f ".env.example" ]; then
        cp .env.example .env
    fi
fi

# 启动后端服务（后台）
echo "📍 后端启动中... (http://localhost:8000)"
nohup uvicorn main:app --reload --host 0.0.0.0 --port 8000 > ../logs/backend.log 2>&1 &
BACKEND_PID=$!
echo $BACKEND_PID > ../logs/backend.pid

cd ..

# 等待后端启动
echo "⏳ 等待后端服务启动..."
sleep 5

# 启动前端
echo "🚀 启动前端服务..."
cd frontend

# 安装依赖
if [ ! -d "node_modules" ]; then
    echo "📥 安装前端依赖..."
    npm install > /dev/null 2>&1
fi

# 创建必要文件
if [ ! -f "next-env.d.ts" ]; then
    echo "/// <reference types=\"next\" />" > next-env.d.ts
    echo "/// <reference types=\"next/image-types/global\" />" >> next-env.d.ts
fi

echo "📍 前端启动中... (http://localhost:3000)"
nohup npm run dev > ../logs/frontend.log 2>&1 &
FRONTEND_PID=$!
echo $FRONTEND_PID > ../logs/frontend.pid

cd ..

# 显示信息
echo ""
echo "🎉 蜀锦蜀绣AI打样图生成工具启动完成！"
echo ""
echo "📱 前端地址: http://localhost:3000"
echo "🔧 后端地址: http://localhost:8000"
echo "📖 API文档: http://localhost:8000/docs"
echo ""
echo "📋 进程信息:"
echo "   后端PID: $BACKEND_PID"
echo "   前端PID: $FRONTEND_PID"
echo ""
echo "📄 日志文件:"
echo "   后端日志: logs/backend.log"
echo "   前端日志: logs/frontend.log"
echo ""
echo "❌ 停止服务请运行: ./stop_all.sh"
echo "🔍 查看状态请运行: ./status.sh"
echo ""
echo "🧵 享受蜀锦蜀绣AI打样图生成之旅！" 