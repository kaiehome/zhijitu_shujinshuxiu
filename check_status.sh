#!/bin/bash

# 蜀锦蜀绣AI打样图生成工具 - 系统状态检查脚本
# 检查前后端服务运行状态和系统健康状况

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
    echo -e "${PURPLE}[STATUS]${NC} $1"
}

# 检查端口占用
check_port() {
    local port=$1
    local service_name=$2
    
    if lsof -i :$port &> /dev/null; then
        local pid=$(lsof -t -i :$port)
        local process_name=$(ps -p $pid -o comm= 2>/dev/null || echo "Unknown")
        log_success "$service_name (端口 $port): 运行中 [PID: $pid, 进程: $process_name]"
        return 0
    else
        log_error "$service_name (端口 $port): 未运行"
        return 1
    fi
}

# 检查HTTP服务
check_http_service() {
    local url=$1
    local service_name=$2
    local timeout=${3:-5}
    
    if curl -s --max-time $timeout "$url" > /dev/null 2>&1; then
        local response_time=$(curl -o /dev/null -s -w "%{time_total}" --max-time $timeout "$url")
        log_success "$service_name HTTP: 响应正常 (${response_time}s)"
        return 0
    else
        log_error "$service_name HTTP: 无响应"
        return 1
    fi
}

# 检查API健康状况
check_api_health() {
    local api_url="http://localhost:8000/api/health"
    
    log_info "检查后端API健康状况..."
    
    if curl -s --max-time 10 "$api_url" > /dev/null 2>&1; then
        local health_data=$(curl -s --max-time 10 "$api_url")
        local status=$(echo "$health_data" | grep -o '"status":"[^"]*"' | cut -d'"' -f4 2>/dev/null || echo "unknown")
        
        if [ "$status" = "healthy" ]; then
            log_success "API健康检查: 正常"
            return 0
        else
            log_warning "API健康检查: 状态异常 ($status)"
            return 1
        fi
    else
        log_error "API健康检查: 无法连接"
        return 1
    fi
}

# 检查磁盘空间
check_disk_space() {
    log_info "检查磁盘空间..."
    
    local available_space=$(df -h . | awk 'NR==2 {print $4}')
    local used_percent=$(df -h . | awk 'NR==2 {print $5}' | sed 's/%//')
    
    log_info "可用空间: $available_space"
    log_info "已使用: $used_percent%"
    
    if [ "$used_percent" -gt 90 ]; then
        log_error "磁盘空间不足 (使用率: $used_percent%)"
        return 1
    elif [ "$used_percent" -gt 80 ]; then
        log_warning "磁盘空间较少 (使用率: $used_percent%)"
        return 0
    else
        log_success "磁盘空间充足 (使用率: $used_percent%)"
        return 0
    fi
}

# 检查内存使用
check_memory_usage() {
    log_info "检查内存使用情况..."
    
    if command -v free &> /dev/null; then
        local memory_info=$(free -m)
        local total_memory=$(echo "$memory_info" | awk 'NR==2{print $2}')
        local used_memory=$(echo "$memory_info" | awk 'NR==2{print $3}')
        local available_memory=$(echo "$memory_info" | awk 'NR==2{print $7}')
        local used_percent=$((used_memory * 100 / total_memory))
        
        log_info "总内存: ${total_memory}MB"
        log_info "已使用: ${used_memory}MB (${used_percent}%)"
        log_info "可用内存: ${available_memory}MB"
        
        if [ "$used_percent" -gt 90 ]; then
            log_error "内存使用率过高 ($used_percent%)"
            return 1
        elif [ "$used_percent" -gt 80 ]; then
            log_warning "内存使用率较高 ($used_percent%)"
            return 0
        else
            log_success "内存使用正常 ($used_percent%)"
            return 0
        fi
    else
        log_warning "无法获取内存信息 (free命令不可用)"
        return 0
    fi
}

# 检查目录结构
check_directories() {
    log_info "检查项目目录结构..."
    
    local required_dirs=("backend" "frontend")
    local required_files=("backend/main.py" "frontend/package.json" "README.md")
    
    local missing_items=()
    
    # 检查目录
    for dir in "${required_dirs[@]}"; do
        if [ ! -d "$dir" ]; then
            missing_items+=("目录: $dir")
        fi
    done
    
    # 检查文件
    for file in "${required_files[@]}"; do
        if [ ! -f "$file" ]; then
            missing_items+=("文件: $file")
        fi
    done
    
    if [ ${#missing_items[@]} -eq 0 ]; then
        log_success "项目目录结构完整"
        return 0
    else
        log_error "缺少以下项目:"
        for item in "${missing_items[@]}"; do
            echo "  - $item"
        done
        return 1
    fi
}

# 检查Python环境
check_python_env() {
    log_info "检查Python环境..."
    
    if [ -d "backend/venv" ]; then
        log_success "Python虚拟环境: 已创建"
        
        # 检查虚拟环境中的关键包
        if [ -f "backend/venv/bin/python" ]; then
            local python_version=$(backend/venv/bin/python --version 2>&1)
            log_info "Python版本: $python_version"
            
            # 检查关键依赖
            local packages=("fastapi" "uvicorn" "opencv-python" "Pillow")
            for package in "${packages[@]}"; do
                if backend/venv/bin/python -c "import $package" 2>/dev/null; then
                    log_success "Python包 $package: 已安装"
                else
                    log_error "Python包 $package: 未安装"
                fi
            done
        else
            log_error "虚拟环境Python解释器不存在"
        fi
    else
        log_warning "Python虚拟环境: 未创建"
    fi
}

# 检查Node.js环境
check_nodejs_env() {
    log_info "检查Node.js环境..."
    
    if [ -d "frontend/node_modules" ]; then
        log_success "Node.js依赖: 已安装"
        
        # 检查关键依赖
        local packages=("next" "react" "antd" "typescript")
        for package in "${packages[@]}"; do
            if [ -d "frontend/node_modules/$package" ]; then
                log_success "Node.js包 $package: 已安装"
            else
                log_error "Node.js包 $package: 未安装"
            fi
        done
    else
        log_warning "Node.js依赖: 未安装"
    fi
}

# 检查日志文件
check_logs() {
    log_info "检查日志文件..."
    
    local log_files=("logs/backend.log" "logs/backend_error.log" "test_results.log")
    
    for log_file in "${log_files[@]}"; do
        if [ -f "$log_file" ]; then
            local file_size=$(du -h "$log_file" | cut -f1)
            local last_modified=$(stat -c %y "$log_file" 2>/dev/null || stat -f %Sm "$log_file" 2>/dev/null || echo "未知")
            log_info "日志文件 $log_file: 存在 (大小: $file_size, 最后修改: ${last_modified:0:19})"
        else
            log_info "日志文件 $log_file: 不存在"
        fi
    done
}

# 生成状态报告
generate_status_report() {
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    local report_file="status_report_$(date '+%Y%m%d_%H%M%S').txt"
    
    {
        echo "蜀锦蜀绣AI打样图生成工具 - 系统状态报告"
        echo "生成时间: $timestamp"
        echo "========================================"
        echo ""
        
        echo "服务状态:"
        check_port 8000 "后端服务" && echo "  ✅ 后端服务: 运行中" || echo "  ❌ 后端服务: 未运行"
        check_port 3000 "前端服务" && echo "  ✅ 前端服务: 运行中" || echo "  ❌ 前端服务: 未运行"
        echo ""
        
        echo "系统资源:"
        echo "  磁盘空间: $(df -h . | awk 'NR==2 {print $4}') 可用"
        if command -v free &> /dev/null; then
            echo "  内存使用: $(free -m | awk 'NR==2{printf "%.1f%%", $3*100/$2}')"
        fi
        echo ""
        
        echo "环境检查:"
        [ -d "backend/venv" ] && echo "  ✅ Python虚拟环境: 已创建" || echo "  ❌ Python虚拟环境: 未创建"
        [ -d "frontend/node_modules" ] && echo "  ✅ Node.js依赖: 已安装" || echo "  ❌ Node.js依赖: 未安装"
        echo ""
        
    } > "$report_file"
    
    log_info "状态报告已保存到: $report_file"
}

# 主函数
main() {
    echo
    log_highlight "🧵 蜀锦蜀绣AI打样图生成工具 - 系统状态检查"
    echo "================================================"
    echo
    
    local total_checks=0
    local passed_checks=0
    
    # 检查项目结构
    log_highlight "📁 检查项目结构..."
    if check_directories; then
        ((passed_checks++))
    fi
    ((total_checks++))
    echo
    
    # 检查服务状态
    log_highlight "🚀 检查服务状态..."
    if check_port 8000 "后端服务"; then
        ((passed_checks++))
    fi
    ((total_checks++))
    
    if check_port 3000 "前端服务"; then
        ((passed_checks++))
    fi
    ((total_checks++))
    echo
    
    # 检查HTTP服务
    log_highlight "🌐 检查HTTP服务..."
    if check_http_service "http://localhost:8000" "后端API"; then
        ((passed_checks++))
    fi
    ((total_checks++))
    
    if check_http_service "http://localhost:3000" "前端页面"; then
        ((passed_checks++))
    fi
    ((total_checks++))
    echo
    
    # 检查API健康状况
    log_highlight "🏥 检查API健康状况..."
    if check_api_health; then
        ((passed_checks++))
    fi
    ((total_checks++))
    echo
    
    # 检查系统资源
    log_highlight "💻 检查系统资源..."
    if check_disk_space; then
        ((passed_checks++))
    fi
    ((total_checks++))
    
    if check_memory_usage; then
        ((passed_checks++))
    fi
    ((total_checks++))
    echo
    
    # 检查环境
    log_highlight "🔧 检查开发环境..."
    check_python_env
    check_nodejs_env
    echo
    
    # 检查日志
    log_highlight "📝 检查日志文件..."
    check_logs
    echo
    
    # 生成报告
    generate_status_report
    echo
    
    # 总结
    local success_rate=$((passed_checks * 100 / total_checks))
    log_highlight "📊 状态检查总结:"
    log_info "总检查项: $total_checks"
    log_info "通过检查: $passed_checks"
    log_info "成功率: $success_rate%"
    
    if [ "$success_rate" -ge 80 ]; then
        log_success "🎉 系统状态良好！"
        echo
        log_info "📍 访问地址:"
        log_info "  前端: http://localhost:3000"
        log_info "  后端API: http://localhost:8000"
        log_info "  API文档: http://localhost:8000/docs"
    else
        log_warning "⚠️  系统存在问题，请检查失败的项目"
        echo
        log_info "🔧 建议操作:"
        log_info "  1. 检查服务是否启动: ./start_backend.sh 和 ./start_frontend.sh"
        log_info "  2. 查看错误日志: tail -f logs/backend_error.log"
        log_info "  3. 运行系统测试: python test_system.py"
    fi
    
    echo
    echo "================================================"
}

# 脚本入口
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi 