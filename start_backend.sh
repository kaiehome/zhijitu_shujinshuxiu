#!/bin/bash

# 蜀锦蜀绣AI打样图生成工具 - 后端启动脚本
# 提供完整的环境检查、依赖安装和服务启动功能

set -e  # 遇到错误立即退出

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 日志函数
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 检查命令是否存在
check_command() {
    if ! command -v "$1" &> /dev/null; then
        log_error "$1 未安装或不在PATH中"
        return 1
    fi
    return 0
}

# 检查端口是否被占用
check_port() {
    local port=$1
    if lsof -i :$port &> /dev/null; then
        log_warning "端口 $port 已被占用"
        log_info "占用端口的进程："
        lsof -i :$port
        read -p "是否要杀死占用进程并继续？(y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            local pid=$(lsof -t -i :$port)
            if [ ! -z "$pid" ]; then
                kill -9 $pid
                log_success "已杀死进程 $pid"
            fi
        else
            log_error "端口冲突，启动终止"
            exit 1
        fi
    fi
}

# 创建日志目录
setup_logging() {
    local log_dir="../logs"
    mkdir -p "$log_dir"
    
    # 设置日志文件
    export BACKEND_LOG_FILE="$log_dir/backend.log"
    export ERROR_LOG_FILE="$log_dir/backend_error.log"
    
    log_info "日志目录已准备: $log_dir"
}

# 环境检查
check_environment() {
    log_info "🔍 开始环境检查..."
    
    # 检查Python
    if ! check_command python3; then
        log_error "Python3 未安装，请先安装 Python 3.8+"
        exit 1
    fi
    
    local python_version=$(python3 --version | cut -d' ' -f2)
    log_success "Python版本: $python_version"
    
    # 检查pip
    if ! check_command pip3; then
        log_error "pip3 未安装"
        exit 1
    fi
    
    # 检查系统依赖
    local system_deps=("gcc" "g++")
    for dep in "${system_deps[@]}"; do
        if ! check_command "$dep"; then
            log_warning "系统依赖 $dep 未找到，可能影响某些包的安装"
        fi
    done
    
    log_success "环境检查完成"
}

# 设置虚拟环境
setup_virtual_env() {
    log_info "📦 设置Python虚拟环境..."
    
    if [ ! -d "venv" ]; then
        log_info "创建虚拟环境..."
        python3 -m venv venv
        log_success "虚拟环境已创建"
    else
        log_info "虚拟环境已存在"
    fi
    
    # 激活虚拟环境
    log_info "激活虚拟环境..."
    source venv/bin/activate
    
    # 验证虚拟环境
    if [[ "$VIRTUAL_ENV" != "" ]]; then
        log_success "虚拟环境已激活: $VIRTUAL_ENV"
    else
        log_error "虚拟环境激活失败"
        exit 1
    fi
    
    # 升级pip
    log_info "升级pip..."
    pip install --upgrade pip --quiet
    log_success "pip已升级到最新版本"
}

# 安装依赖
install_dependencies() {
    log_info "📥 安装Python依赖包..."
    
    if [ ! -f "requirements.txt" ]; then
        log_error "requirements.txt 文件不存在"
        exit 1
    fi
    
    # 检查是否需要更新依赖
    local need_install=false
    
    if [ ! -f "venv/installed_packages.txt" ]; then
        need_install=true
    elif ! diff requirements.txt venv/installed_packages.txt &> /dev/null; then
        need_install=true
        log_info "检测到依赖变更，需要重新安装"
    fi
    
    if [ "$need_install" = true ]; then
        log_info "安装依赖包..."
        
        # 安装依赖并记录输出
        if pip install -r requirements.txt --quiet; then
            # 记录已安装的包
            cp requirements.txt venv/installed_packages.txt
            log_success "依赖包安装完成"
            
            # 显示安装的关键包版本
            log_info "关键依赖版本："
            pip show fastapi uvicorn opencv-python Pillow scikit-learn --quiet | grep -E "Name|Version" | paste - -
        else
            log_error "依赖包安装失败"
            exit 1
        fi
    else
        log_info "依赖包已是最新版本"
    fi
}

# 创建必要目录
setup_directories() {
    log_info "📁 创建必要目录..."
    
    local dirs=("uploads" "outputs")
    for dir in "${dirs[@]}"; do
        if [ ! -d "$dir" ]; then
            mkdir -p "$dir"
            log_success "创建目录: $dir"
        else
            log_info "目录已存在: $dir"
        fi
        
        # 设置目录权限
        chmod 755 "$dir"
    done
}

# 检查配置文件
check_configuration() {
    log_info "⚙️ 检查配置文件..."
    
    # 检查环境变量文件
    if [ ! -f ".env" ]; then
        if [ -f ".env.example" ]; then
            cp .env.example .env
            log_success "已创建.env配置文件"
            log_warning "请根据需要修改 .env 文件中的配置"
        else
            log_warning ".env 配置文件不存在，将使用默认配置"
        fi
    else
        log_info ".env 配置文件已存在"
    fi
}

# 健康检查
health_check() {
    log_info "🏥 执行健康检查..."
    
    # 检查Python模块导入
    local test_imports=(
        "fastapi"
        "uvicorn"
        "cv2"
        "PIL"
        "sklearn"
        "numpy"
    )
    
    for module in "${test_imports[@]}"; do
        if python3 -c "import $module" 2>/dev/null; then
            log_success "模块 $module 导入正常"
        else
            log_error "模块 $module 导入失败"
            exit 1
        fi
    done
}

# 启动服务
start_service() {
    log_info "🚀 启动FastAPI服务器..."
    
    # 设置通义千问API密钥
    if [ -n "$TONGYI_API_KEY" ]; then
        export TONGYI_API_KEY="$TONGYI_API_KEY"
        log_success "通义千问API密钥已设置"
    else
        log_warning "通义千问API密钥未设置"
    fi
    
    # 检查端口
    check_port 8000
    
    # 显示启动信息
    echo
    log_success "=== 蜀锦蜀绣AI打样图生成工具后端服务 ==="
    log_info "📍 后端地址: http://localhost:8000"
    log_info "📖 API文档: http://localhost:8000/docs"
    log_info "🔧 Redoc文档: http://localhost:8000/redoc"
    log_info "📊 健康检查: http://localhost:8000/api/health"
    log_info "📝 日志文件: $BACKEND_LOG_FILE"
    echo
    log_warning "❌ 停止服务请按 Ctrl+C"
    echo
    
    # 启动服务
    exec env TONGYI_API_KEY="$TONGYI_API_KEY" uvicorn main:app \
        --reload \
        --host 0.0.0.0 \
        --port 8000 \
        --log-level info \
        --access-log \
        --loop asyncio \
        --workers 1
}

# 清理函数
cleanup() {
    log_info "🧹 清理临时文件..."
    # 这里可以添加清理逻辑
}

# 信号处理
trap cleanup EXIT

# 主函数
main() {
    echo
    log_success "🧵 启动蜀锦蜀绣AI打样图生成工具后端..."
    echo
    
    # 检查是否在正确的目录
    if [ ! -f "main.py" ]; then
        log_error "请在backend目录中运行此脚本"
        exit 1
    fi
    
    # 执行启动流程
    setup_logging
    check_environment
    setup_virtual_env
    install_dependencies
    setup_directories
    check_configuration
    health_check
    start_service
}

# 脚本入口
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    # 进入backend目录
    cd "$(dirname "$0")/backend"
    main "$@"
fi 