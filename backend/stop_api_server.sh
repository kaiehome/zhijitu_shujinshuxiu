#!/bin/bash

# 停止刺绣图像处理API服务器

echo "🛑 停止刺绣图像处理API服务器..."

# 查找并停止uvicorn进程
PIDS=$(pgrep -f "uvicorn.*api_server")

if [ -z "$PIDS" ]; then
    echo "ℹ️  没有找到运行中的API服务器进程"
    exit 0
fi

echo "📊 找到以下进程:"
ps aux | grep "uvicorn.*api_server" | grep -v grep

echo ""
echo "🔄 正在停止进程..."

for PID in $PIDS; do
    echo "停止进程 $PID..."
    kill $PID
done

# 等待进程停止
sleep 2

# 检查是否还有进程在运行
REMAINING_PIDS=$(pgrep -f "uvicorn.*api_server")
if [ -n "$REMAINING_PIDS" ]; then
    echo "⚠️  进程仍在运行，强制停止..."
    for PID in $REMAINING_PIDS; do
        echo "强制停止进程 $PID..."
        kill -9 $PID
    done
fi

# 最终检查
sleep 1
if pgrep -f "uvicorn.*api_server" > /dev/null; then
    echo "❌ 无法停止所有进程"
    exit 1
else
    echo "✅ 所有API服务器进程已停止"
fi

# 检查端口
if lsof -i :8000 > /dev/null 2>&1; then
    echo "⚠️  端口8000仍被占用，尝试释放..."
    lsof -ti:8000 | xargs kill -9 2>/dev/null
fi

echo "🎉 API服务器已完全停止" 