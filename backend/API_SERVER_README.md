# 🎨 刺绣图像处理API服务器使用指南

## 📋 概述

刺绣图像处理API服务器提供了完整的AI驱动刺绣图像生成和优化服务，包括：

- 🖼️ 图像处理和优化
- 🎨 多种艺术风格支持
- 🤖 智能参数调优
- 📊 质量分析和评估
- 🔄 批量处理支持

## 🚀 快速启动

### 方法1：使用启动脚本（推荐）

```bash
# 启动服务器
./start_api_server.sh

# 检查状态
./check_server_status.sh

# 停止服务器
./stop_api_server.sh
```

### 方法2：手动启动

```bash
# 启动服务器
uvicorn api_server:app --host 127.0.0.1 --port 8000 --reload --log-level info

# 或者使用Python直接运行
python api_server.py
```

## 🌐 访问地址

- **服务器地址**: http://127.0.0.1:8000
- **API文档**: http://127.0.0.1:8000/docs
- **健康检查**: http://127.0.0.1:8000/health
- **可用模型**: http://127.0.0.1:8000/models
- **系统统计**: http://127.0.0.1:8000/stats

## 📚 API端点

### 基础端点

| 端点 | 方法 | 描述 |
|------|------|------|
| `/` | GET | 服务器根路径 |
| `/health` | GET | 健康检查 |
| `/models` | GET | 获取可用模型 |
| `/stats` | GET | 获取系统统计 |

### 图像处理端点

| 端点 | 方法 | 描述 |
|------|------|------|
| `/generate` | POST | 生成刺绣图像 |
| `/batch-generate` | POST | 批量生成刺绣图像 |
| `/job/{job_id}` | GET | 获取任务状态 |
| `/jobs` | GET | 列出所有任务 |
| `/download/{job_id}/{filename}` | GET | 下载结果文件 |

### 配置端点

| 端点 | 方法 | 描述 |
|------|------|------|
| `/styles` | GET | 获取可用风格 |
| `/optimization-levels` | GET | 获取优化级别 |

## 🎨 支持的处理风格

1. **basic** - 基础风格：标准图像处理
2. **embroidery** - 刺绣风格：专为刺绣优化
3. **traditional** - 传统风格：传统艺术风格
4. **modern** - 现代风格：现代艺术风格

## ⚙️ 优化级别

1. **conservative** - 保守优化：轻微优化，保持原图特征
2. **balanced** - 平衡优化：平衡质量和处理速度
3. **aggressive** - 激进优化：最大程度优化，可能改变原图特征

## 📝 使用示例

### 1. 检查服务器状态

```bash
curl http://127.0.0.1:8000/health
```

### 2. 获取可用模型

```bash
curl http://127.0.0.1:8000/models
```

### 3. 生成刺绣图像

```bash
curl -X POST "http://127.0.0.1:8000/generate" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@your_image.jpg" \
  -F "style=embroidery" \
  -F "color_count=16" \
  -F "optimization_level=balanced"
```

### 4. 检查任务状态

```bash
curl http://127.0.0.1:8000/job/{job_id}
```

## 🔧 配置选项

服务器支持以下配置选项：

- `cache_dir`: 缓存目录
- `max_batch_size`: 最大批处理大小
- `enable_optimization`: 启用优化
- `enable_quality_analysis`: 启用质量分析
- `default_style`: 默认风格
- `timeout_seconds`: 超时时间

## 🛠️ 故障排除

### 常见问题

1. **端口被占用**
   ```bash
   # 检查端口占用
   lsof -i :8000
   
   # 停止占用进程
   ./stop_api_server.sh
   ```

2. **依赖缺失**
   ```bash
   # 安装依赖
   pip install -r requirements.txt
   ```

3. **权限问题**
   ```bash
   # 给脚本执行权限
   chmod +x *.sh
   ```

### 日志查看

服务器日志会显示在控制台中，包括：
- 请求日志
- 错误信息
- 性能指标

## 📊 性能监控

服务器提供以下监控指标：

- 活动任务数
- 已完成任务数
- 可用模型状态
- 系统配置信息

## 🔒 安全注意事项

- 服务器默认只监听本地地址 (127.0.0.1)
- 生产环境部署时请配置适当的安全措施
- 定期更新依赖包以修复安全漏洞

## 📞 技术支持

如遇到问题，请检查：

1. 服务器日志输出
2. 依赖包版本兼容性
3. 系统资源使用情况
4. 网络连接状态 