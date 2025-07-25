#!/bin/bash

# 检查API服务器状态

SERVER_URL="http://127.0.0.1:8000"

echo "🔍 检查刺绣图像处理API服务器状态..."
echo ""

# 检查进程
echo "📊 进程状态:"
if pgrep -f "uvicorn.*api_server" > /dev/null; then
    echo "✅ 服务器进程正在运行"
    ps aux | grep "uvicorn.*api_server" | grep -v grep
else
    echo "❌ 服务器进程未运行"
fi

echo ""

# 检查端口
echo "🌐 端口状态:"
if lsof -i :8000 > /dev/null 2>&1; then
    echo "✅ 端口8000正在监听"
    lsof -i :8000
else
    echo "❌ 端口8000未监听"
fi

echo ""

# 检查API响应
echo "🔗 API响应测试:"
if curl -s "$SERVER_URL/health" > /dev/null; then
    echo "✅ 健康检查端点响应正常"
    echo "📋 健康检查详情:"
    curl -s "$SERVER_URL/health" | python -m json.tool 2>/dev/null || curl -s "$SERVER_URL/health"
else
    echo "❌ 健康检查端点无响应"
fi

echo ""

# 检查可用模型
echo "🤖 可用模型:"
if curl -s "$SERVER_URL/models" > /dev/null; then
    echo "✅ 模型端点响应正常"
    echo "📋 可用模型详情:"
    curl -s "$SERVER_URL/models" | python -m json.tool 2>/dev/null || curl -s "$SERVER_URL/models"
else
    echo "❌ 模型端点无响应"
fi

echo ""

# 显示访问信息
echo "📖 访问信息:"
echo "🌐 服务器地址: $SERVER_URL"
echo "📊 API文档: $SERVER_URL/docs"
echo "🔍 健康检查: $SERVER_URL/health"
echo "🤖 可用模型: $SERVER_URL/models"
echo "📈 系统统计: $SERVER_URL/stats" 