#!/bin/bash

# 蜀锦蜀绣AI打样图生成工具 - 停止服务脚本

echo "🛑 停止蜀锦蜀绣AI打样图生成工具..."

# 停止后端服务
if [ -f "logs/backend.pid" ]; then
    BACKEND_PID=$(cat logs/backend.pid)
    if ps -p $BACKEND_PID > /dev/null; then
        echo "🔧 停止后端服务 (PID: $BACKEND_PID)..."
        kill $BACKEND_PID
        rm logs/backend.pid
    else
        echo "⚠️  后端服务未运行"
        rm -f logs/backend.pid
    fi
else
    echo "⚠️  未找到后端进程ID文件"
fi

# 停止前端服务
if [ -f "logs/frontend.pid" ]; then
    FRONTEND_PID=$(cat logs/frontend.pid)
    if ps -p $FRONTEND_PID > /dev/null; then
        echo "📱 停止前端服务 (PID: $FRONTEND_PID)..."
        kill $FRONTEND_PID
        rm logs/frontend.pid
    else
        echo "⚠️  前端服务未运行"
        rm -f logs/frontend.pid
    fi
else
    echo "⚠️  未找到前端进程ID文件"
fi

# 强制清理可能残留的进程
echo "🧹 清理残留进程..."
pkill -f "uvicorn main:app" 2>/dev/null || true
pkill -f "next-server" 2>/dev/null || true
pkill -f "npm run dev" 2>/dev/null || true

echo "✅ 服务已停止" 