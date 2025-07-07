#!/bin/bash

# 蜀锦蜀绣AI打样图生成工具 - 前端启动脚本
# 提供完整的Node.js环境检查、依赖安装和开发服务器启动功能

set -e  # 遇到错误立即退出

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
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

log_highlight() {
    echo -e "${PURPLE}[HIGHLIGHT]${NC} $1"
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
                sleep 2  # 等待进程完全退出
            fi
        else
            log_error "端口冲突，启动终止"
            exit 1
        fi
    fi
}

# 版本比较函数
version_compare() {
    local version1=$1
    local version2=$2
    
    # 移除 'v' 前缀
    version1=${version1#v}
    version2=${version2#v}
    
    # 简单的版本比较
    if [[ "$version1" == "$version2" ]]; then
        return 0
    fi
    
    local IFS=.
    local i ver1=($version1) ver2=($version2)
    
    # 比较版本号
    for ((i=0; i<${#ver1[@]} || i<${#ver2[@]}; i++)); do
        if [[ ${ver1[i]:-0} -gt ${ver2[i]:-0} ]]; then
            return 1
        elif [[ ${ver1[i]:-0} -lt ${ver2[i]:-0} ]]; then
            return 2
        fi
    done
    return 0
}

# 环境检查
check_environment() {
    log_info "🔍 开始环境检查..."
    
    # 检查Node.js
    if ! check_command node; then
        log_error "Node.js 未安装，请先安装 Node.js 16+"
        log_info "推荐使用 nvm 安装: https://github.com/nvm-sh/nvm"
        exit 1
    fi
    
    local node_version=$(node --version)
    log_success "Node.js版本: $node_version"
    
    # 检查Node.js版本（需要16+）
    if version_compare "$node_version" "v16.0.0"; then
        if [[ $? -eq 2 ]]; then
            log_error "Node.js版本过低，需要16.0.0或更高版本"
            log_info "当前版本: $node_version"
            exit 1
        fi
    fi
    
    # 检查包管理器
    local package_manager=""
    if check_command yarn; then
        package_manager="yarn"
        local yarn_version=$(yarn --version)
        log_success "Yarn版本: $yarn_version"
    elif check_command npm; then
        package_manager="npm"
        local npm_version=$(npm --version)
        log_success "npm版本: $npm_version"
    else
        log_error "未找到npm或yarn包管理器"
        exit 1
    fi
    
    export PACKAGE_MANAGER="$package_manager"
    log_success "使用包管理器: $package_manager"
    
    # 检查系统内存
    local available_memory=$(free -m | awk 'NR==2{printf "%.1f", $7/1024}' 2>/dev/null || echo "未知")
    if [[ "$available_memory" != "未知" ]]; then
        log_info "可用内存: ${available_memory}GB"
        if (( $(echo "$available_memory < 1.0" | bc -l) )); then
            log_warning "可用内存较低，编译可能较慢"
        fi
    fi
    
    log_success "环境检查完成"
}

# 检查项目配置
check_project_config() {
    log_info "⚙️ 检查项目配置..."
    
    # 检查package.json
    if [ ! -f "package.json" ]; then
        log_error "package.json 文件不存在"
        exit 1
    fi
    
    # 检查Next.js配置
    if [ ! -f "next.config.js" ] && [ ! -f "next.config.mjs" ]; then
        log_warning "Next.js配置文件不存在，将使用默认配置"
    fi
    
    # 检查TypeScript配置
    if [ -f "tsconfig.json" ]; then
        log_info "TypeScript项目已检测"
    fi
    
    # 检查环境变量文件
    if [ ! -f ".env.local" ]; then
        if [ -f ".env.example" ]; then
            cp .env.example .env.local
            log_success "已创建.env.local配置文件"
            log_warning "请根据需要修改 .env.local 文件中的配置"
        else
            log_info "未找到环境变量配置文件，将使用默认配置"
        fi
    else
        log_info ".env.local 配置文件已存在"
    fi
    
    log_success "项目配置检查完成"
}

# 安装依赖
install_dependencies() {
    log_info "📥 检查和安装依赖包..."
    
    # 检查是否需要安装依赖
    local need_install=false
    
    if [ ! -d "node_modules" ]; then
        need_install=true
        log_info "node_modules目录不存在，需要安装依赖"
    elif [ ! -f "node_modules/.install_timestamp" ]; then
        need_install=true
        log_info "未找到安装时间戳，需要重新安装依赖"
    else
        # 检查package.json是否有更新
        local package_time=$(stat -c %Y package.json 2>/dev/null || stat -f %m package.json)
        local install_time=$(stat -c %Y node_modules/.install_timestamp 2>/dev/null || stat -f %m node_modules/.install_timestamp)
        
        if [ "$package_time" -gt "$install_time" ]; then
            need_install=true
            log_info "package.json已更新，需要重新安装依赖"
        fi
    fi
    
    if [ "$need_install" = true ]; then
        log_info "开始安装依赖包..."
        
        # 清理缓存（如果存在问题）
        if [ -d "node_modules" ]; then
            log_info "清理旧的node_modules..."
            rm -rf node_modules
        fi
        
        # 根据包管理器安装依赖
        local install_cmd=""
        case "$PACKAGE_MANAGER" in
            "yarn")
                install_cmd="yarn install --frozen-lockfile"
                ;;
            "npm")
                install_cmd="npm ci"
                if [ ! -f "package-lock.json" ]; then
                    install_cmd="npm install"
                fi
                ;;
        esac
        
        log_info "执行命令: $install_cmd"
        
        # 执行安装
        if eval "$install_cmd"; then
            # 记录安装时间戳
            touch node_modules/.install_timestamp
            log_success "依赖包安装完成"
            
            # 显示关键依赖信息
            log_info "关键依赖版本："
            if [ -f "package-lock.json" ]; then
                jq -r '.dependencies | to_entries[] | select(.key | test("^(next|react|antd|typescript)")) | "\(.key): \(.value.version)"' package-lock.json 2>/dev/null || true
            elif [ -f "yarn.lock" ]; then
                grep -E "^(next|react|antd|typescript)@" yarn.lock | head -5 || true
            fi
        else
            log_error "依赖包安装失败"
            log_info "尝试清理缓存后重新安装..."
            
            # 清理缓存
            case "$PACKAGE_MANAGER" in
                "yarn")
                    yarn cache clean
                    ;;
                "npm")
                    npm cache clean --force
                    ;;
            esac
            
            exit 1
        fi
    else
        log_info "依赖包已是最新版本"
    fi
}

# 构建检查
build_check() {
    log_info "🔨 执行构建检查..."
    
    # 检查TypeScript类型
    if [ -f "tsconfig.json" ]; then
        log_info "检查TypeScript类型..."
        if $PACKAGE_MANAGER run type-check 2>/dev/null || npx tsc --noEmit; then
            log_success "TypeScript类型检查通过"
        else
            log_warning "TypeScript类型检查发现问题，但不影响开发服务器启动"
        fi
    fi
    
    # 检查ESLint
    if [ -f ".eslintrc.json" ] || [ -f ".eslintrc.js" ]; then
        log_info "运行ESLint检查..."
        if $PACKAGE_MANAGER run lint 2>/dev/null || npx eslint . --ext .ts,.tsx,.js,.jsx --max-warnings 10; then
            log_success "ESLint检查通过"
        else
            log_warning "ESLint检查发现问题，但不影响开发服务器启动"
        fi
    fi
}

# 清理函数
cleanup_dev_files() {
    log_info "🧹 清理开发文件..."
    
    # 清理Next.js缓存
    if [ -d ".next" ]; then
        rm -rf .next
        log_info "已清理.next缓存目录"
    fi
    
    # 清理TypeScript构建信息
    if [ -f "tsconfig.tsbuildinfo" ]; then
        rm -f tsconfig.tsbuildinfo
        log_info "已清理TypeScript构建信息"
    fi
}

# 启动开发服务器
start_dev_server() {
    log_info "🚀 启动Next.js开发服务器..."
    
    # 检查端口
    check_port 3000
    
    # 设置环境变量
    export NODE_ENV=development
    
    # 显示启动信息
    echo
    log_success "=== 蜀锦蜀绣AI打样图生成工具前端服务 ==="
    log_info "🌐 前端地址: http://localhost:3000"
    log_info "📱 移动端预览: http://[您的IP]:3000"
    log_info "🔧 开发工具: React DevTools, Next.js DevTools"
    log_info "🔄 热重载: 已启用"
    echo
    log_warning "❌ 停止服务请按 Ctrl+C"
    echo
    
    # 启动命令
    local dev_cmd=""
    case "$PACKAGE_MANAGER" in
        "yarn")
            dev_cmd="yarn dev"
            ;;
        "npm")
            dev_cmd="npm run dev"
            ;;
    esac
    
    log_highlight "执行命令: $dev_cmd"
    echo
    
    # 启动开发服务器
    exec $dev_cmd
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
    log_success "🧵 启动蜀锦蜀绣AI打样图生成工具前端..."
    echo
    
    # 检查是否在正确的目录
    if [ ! -f "package.json" ]; then
        log_error "请在frontend目录中运行此脚本"
        exit 1
    fi
    
    # 执行启动流程
    check_environment
    check_project_config
    install_dependencies
    build_check
    start_dev_server
}

# 脚本选项处理
while [[ $# -gt 0 ]]; do
    case $1 in
        --clean)
            cleanup_dev_files
            shift
            ;;
        --help|-h)
            echo "蜀锦蜀绣AI打样图生成工具前端启动脚本"
            echo ""
            echo "用法: $0 [选项]"
            echo ""
            echo "选项:"
            echo "  --clean    清理开发文件缓存"
            echo "  --help     显示此帮助信息"
            echo ""
            exit 0
            ;;
        *)
            log_error "未知选项: $1"
            log_info "使用 --help 查看可用选项"
            exit 1
            ;;
    esac
done

# 脚本入口
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    # 进入frontend目录
    cd "$(dirname "$0")/frontend"
    main "$@"
fi 