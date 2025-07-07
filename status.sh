#!/bin/bash

# 蜀锦蜀绣AI打样图生成工具 - 状态检查脚本

echo "🧵 蜀锦蜀绣AI打样图生成工具状态检查"
echo "=================================="

# 检查后端状态
echo "🔧 后端服务状态:"
if [ -f "logs/backend.pid" ]; then
    BACKEND_PID=$(cat logs/backend.pid)
    if ps -p $BACKEND_PID > /dev/null; then
        echo "   ✅ 运行中 (PID: $BACKEND_PID)"
        echo "   📍 地址: http://localhost:8000"
        
        # 尝试访问后端健康检查
        if command -v curl &> /dev/null; then
            HEALTH_CHECK=$(curl -s http://localhost:8000/api/health 2>/dev/null)
            if [ $? -eq 0 ]; then
                echo "   💚 健康检查: 通过"
            else
                echo "   ❤️  健康检查: 失败"
            fi
        fi
    else
        echo "   ❌ 已停止"
        rm -f logs/backend.pid
    fi
else
    echo "   ❌ 未运行"
fi

echo ""

# 检查前端状态
echo "📱 前端服务状态:"
if [ -f "logs/frontend.pid" ]; then
    FRONTEND_PID=$(cat logs/frontend.pid)
    if ps -p $FRONTEND_PID > /dev/null; then
        echo "   ✅ 运行中 (PID: $FRONTEND_PID)"
        echo "   📍 地址: http://localhost:3000"
        
        # 尝试访问前端
        if command -v curl &> /dev/null; then
            HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:3000 2>/dev/null)
            if [ "$HTTP_CODE" = "200" ]; then
                echo "   💚 页面访问: 正常"
            else
                echo "   ❤️  页面访问: 异常 (HTTP: $HTTP_CODE)"
            fi
        fi
    else
        echo "   ❌ 已停止"
        rm -f logs/frontend.pid
    fi
else
    echo "   ❌ 未运行"
fi

echo ""

# 检查端口占用
echo "🌐 端口占用情况:"
if command -v lsof &> /dev/null; then
    echo "   后端端口 8000:"
    lsof -i :8000 | grep LISTEN || echo "     未占用"
    echo "   前端端口 3000:"
    lsof -i :3000 | grep LISTEN || echo "     未占用"
elif command -v netstat &> /dev/null; then
    echo "   后端端口 8000:"
    netstat -ln | grep :8000 || echo "     未占用"
    echo "   前端端口 3000:"
    netstat -ln | grep :3000 || echo "     未占用"
else
    echo "   无法检查端口占用 (缺少lsof或netstat命令)"
fi

echo ""

# 检查日志文件
echo "📄 日志文件:"
if [ -f "logs/backend.log" ]; then
    BACKEND_LOG_SIZE=$(wc -l < logs/backend.log)
    echo "   后端日志: $BACKEND_LOG_SIZE 行"
else
    echo "   后端日志: 不存在"
fi

if [ -f "logs/frontend.log" ]; then
    FRONTEND_LOG_SIZE=$(wc -l < logs/frontend.log)
    echo "   前端日志: $FRONTEND_LOG_SIZE 行"
else
    echo "   前端日志: 不存在"
fi

echo ""

# 检查目录结构
echo "📁 关键目录:"
[ -d "backend/uploads" ] && echo "   ✅ 上传目录存在" || echo "   ❌ 上传目录不存在"
[ -d "backend/outputs" ] && echo "   ✅ 输出目录存在" || echo "   ❌ 输出目录不存在"
[ -d "frontend/node_modules" ] && echo "   ✅ 前端依赖已安装" || echo "   ❌ 前端依赖未安装"
[ -d "backend/venv" ] && echo "   ✅ Python虚拟环境存在" || echo "   ❌ Python虚拟环境不存在"

echo ""
echo "==================================" 