# 🧵✨ 蜀锦蜀绣 AI 打样图生成工具

专业的织机识别图像处理工具，传承千年蜀锦工艺，融合现代AI技术。

## 📋 项目概述

蜀锦蜀绣AI打样图生成工具是一个专门为传统织机设计的图像处理系统，能够将普通图像转换为适合蜀锦蜀绣工艺的高质量打样图。

### ✨ 核心功能

- **🎨 智能颜色降色**: 使用K-means聚类算法将图像颜色数量降至6-12种
- **🔍 边缘增强处理**: 多层次边缘增强，突出图案轮廓
- **🧹 噪声清理优化**: 双边滤波和形态学操作，提升图像质量
- **🏮 蜀锦风格化**: 专门针对蜀锦蜀绣传统特色的色彩调整
- **📥 多格式输出**: 高质量PNG主图 + SVG矢量辅助文件
- **⚡ 实时处理**: 支持实时状态查询和进度跟踪

### 🏗️ 技术架构

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   前端 (Next.js) │ ←→ │  后端 (FastAPI)  │ ←→ │ 图像处理 (OpenCV)│
│                 │    │                 │    │                 │
│ • React 18      │    │ • Python 3.8+  │    │ • OpenCV        │
│ • TypeScript    │    │ • FastAPI       │    │ • PIL/Pillow    │
│ • Ant Design    │    │ • Uvicorn       │    │ • scikit-learn  │
│ • Liquid Glass  │    │ • Async/Await   │    │ • NumPy         │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🚀 快速开始

### 📋 系统要求

- **操作系统**: Windows 10+, macOS 10.15+, Ubuntu 18.04+
- **Python**: 3.8+ (推荐 3.9 或 3.10)
- **Node.js**: 16.0+ (推荐 18.x LTS)
- **内存**: 最少 4GB，推荐 8GB+
- **磁盘空间**: 至少 2GB 可用空间

### 🔧 安装指南

#### 1. 克隆项目

```bash
git clone https://github.com/your-org/sichuan-brocade-ai-tool.git
cd sichuan-brocade-ai-tool
```

#### 2. 后端安装

```bash
# 给启动脚本执行权限
chmod +x start_backend.sh

# 启动后端服务（自动安装依赖）
./start_backend.sh
```

#### 3. 前端安装

```bash
# 给启动脚本执行权限  
chmod +x start_frontend.sh

# 启动前端服务（自动安装依赖）
./start_frontend.sh
```

### 🎯 使用方法

1. **访问应用**: 打开浏览器访问 `http://localhost:3000`
2. **上传图片**: 点击上传区域，选择JPG或PNG格式图像（最大10MB）
3. **配置参数**: 
   - 颜色数量：6-12种（推荐8种）
   - 边缘增强：开启可突出图案轮廓
   - 噪声清理：开启可提升图像质量
4. **开始处理**: 点击"开始处理"按钮，等待10-60秒
5. **下载结果**: 处理完成后下载PNG主图和SVG辅助文件

## 🛠️ 开发指南

### 📁 项目结构

```
sichuan-brocade-ai-tool/
├── backend/                 # 后端服务
│   ├── main.py             # FastAPI 主应用
│   ├── models.py           # 数据模型和验证
│   ├── image_processor.py  # 图像处理核心
│   ├── requirements.txt    # Python 依赖
│   └── uploads/            # 上传文件目录
├── frontend/               # 前端应用
│   ├── src/                # 源代码
│   │   ├── pages/          # 页面组件
│   │   ├── components/     # 通用组件
│   │   └── styles/         # 样式文件
│   ├── package.json        # Node.js 依赖
│   └── next.config.js      # Next.js 配置
├── scripts/                # 脚本文件
├── docs/                   # 文档
├── start_backend.sh        # 后端启动脚本
├── start_frontend.sh       # 前端启动脚本
├── test_system.py          # 系统测试脚本
└── README.md              # 项目文档
```

### 🔧 开发环境配置

#### 后端开发

```bash
cd backend

# 创建虚拟环境
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# 或 venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt

# 启动开发服务器
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

#### 前端开发

```bash
cd frontend

# 安装依赖
npm install
# 或 yarn install

# 启动开发服务器
npm run dev
# 或 yarn dev
```

### 🧪 测试

#### 系统测试

```bash
# 运行完整系统测试
python test_system.py

# 跳过性能测试
python test_system.py --no-performance

# 指定服务地址
python test_system.py --backend-url http://localhost:8000 --frontend-url http://localhost:3000

# 输出测试结果到文件
python test_system.py --output test_results.json
```

#### 单元测试

```bash
# 后端测试
cd backend
python -m pytest tests/ -v

# 前端测试
cd frontend
npm test
# 或 yarn test
```

### 📊 API 文档

启动后端服务后，访问以下地址查看API文档：

- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`
- **健康检查**: `http://localhost:8000/api/health`

#### 主要API端点

| 端点 | 方法 | 描述 |
|------|------|------|
| `/api/health` | GET | 健康检查 |
| `/api/upload` | POST | 上传图像文件 |
| `/api/process` | POST | 开始图像处理 |
| `/api/status/{job_id}` | GET | 查询处理状态 |
| `/api/download/{job_id}/{filename}` | GET | 下载处理结果 |

## 🎨 界面设计

### 设计特色

- **液态玻璃效果**: 现代化的毛玻璃背景和液态动画
- **传统色彩搭配**: 融合蜀锦传统色彩元素
- **响应式设计**: 支持桌面端和移动端访问
- **无障碍支持**: 符合WCAG 2.1标准

### 主要组件

- **Header**: 顶部导航栏，包含标题和功能标签
- **UploadSection**: 文件上传区域，支持拖拽上传
- **ProcessSection**: 处理参数配置面板
- **ResultSection**: 结果展示和下载区域

## 🚀 部署指南

### 🐳 Docker 部署

```bash
# 构建镜像
docker-compose build

# 启动服务
docker-compose up -d

# 查看日志
docker-compose logs -f
```

### ☁️ 云服务部署

#### 后端部署 (推荐使用 Railway/Render)

1. 连接 GitHub 仓库
2. 设置环境变量
3. 选择 Python 运行时
4. 设置启动命令: `uvicorn main:app --host 0.0.0.0 --port $PORT`

#### 前端部署 (推荐使用 Vercel/Netlify)

1. 连接 GitHub 仓库
2. 设置构建命令: `npm run build`
3. 设置输出目录: `out` 或 `.next`
4. 配置环境变量

### 🔧 环境变量配置

#### 后端环境变量

```bash
# .env
ENVIRONMENT=production
MAX_UPLOAD_SIZE=10485760
CORS_ORIGINS=["http://localhost:3000", "https://your-frontend-domain.com"]
LOG_LEVEL=info
```

#### 前端环境变量

```bash
# .env.local
NEXT_PUBLIC_API_BASE_URL=https://your-backend-domain.com
NEXT_PUBLIC_MAX_FILE_SIZE=10485760
```

## 🔒 安全性

### 安全特性

- **文件类型验证**: 严格验证上传文件格式
- **文件大小限制**: 限制上传文件大小（10MB）
- **路径遍历防护**: 防止目录遍历攻击
- **CORS 配置**: 限制跨域请求来源
- **输入验证**: 全面的输入参数验证
- **错误处理**: 安全的错误信息返回

### 安全建议

1. **生产环境部署时更改默认密钥**
2. **使用 HTTPS 协议**
3. **定期更新依赖包**
4. **配置防火墙规则**
5. **启用访问日志监控**

## 📈 性能优化

### 后端优化

- **异步处理**: 使用 asyncio 和 FastAPI 异步特性
- **并发限制**: 控制同时处理的任务数量
- **内存管理**: 及时释放图像处理内存
- **缓存机制**: 对处理结果进行缓存
- **文件清理**: 定期清理临时文件

### 前端优化

- **代码分割**: 使用 Next.js 自动代码分割
- **图片优化**: 使用 Next.js Image 组件
- **静态生成**: 预渲染静态页面
- **CDN 部署**: 使用 CDN 加速静态资源
- **压缩优化**: Gzip/Brotli 压缩

## 🐛 故障排除

### 常见问题

#### 后端问题

**Q: Python 依赖安装失败**
```bash
# 升级 pip
pip install --upgrade pip

# 安装系统依赖 (Ubuntu)
sudo apt-get install python3-dev libgl1-mesa-glx

# 安装系统依赖 (CentOS)
sudo yum install python3-devel mesa-libGL
```

**Q: 端口被占用**
```bash
# 查找占用端口的进程
lsof -i :8000

# 杀死进程
kill -9 <PID>
```

#### 前端问题

**Q: Node.js 版本过低**
```bash
# 使用 nvm 安装最新版本
nvm install 18
nvm use 18
```

**Q: 依赖安装失败**
```bash
# 清理缓存
npm cache clean --force
rm -rf node_modules package-lock.json
npm install
```

### 日志查看

```bash
# 后端日志
tail -f logs/backend.log

# 前端日志
npm run dev  # 开发模式会在终端显示日志
```

## 🤝 贡献指南

### 开发流程

1. **Fork 项目**
2. **创建功能分支**: `git checkout -b feature/new-feature`
3. **提交更改**: `git commit -am 'Add new feature'`
4. **推送分支**: `git push origin feature/new-feature`
5. **创建 Pull Request**

### 代码规范

#### Python 代码规范

- 遵循 PEP 8 标准
- 使用 Black 进行代码格式化
- 使用 flake8 进行代码检查
- 编写详细的文档字符串

#### TypeScript 代码规范

- 遵循 ESLint 配置
- 使用 Prettier 进行代码格式化
- 编写 JSDoc 注释
- 使用严格的 TypeScript 类型

### 测试要求

- 新功能必须包含测试用例
- 测试覆盖率不低于 80%
- 所有测试必须通过
- 包含集成测试

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 👥 团队

- **项目负责人**: [Your Name](mailto:your.email@example.com)
- **后端开发**: [Backend Developer](mailto:backend@example.com)
- **前端开发**: [Frontend Developer](mailto:frontend@example.com)
- **UI/UX 设计**: [Designer](mailto:design@example.com)

## 🙏 致谢

感谢所有为蜀锦蜀绣传统工艺数字化做出贡献的开发者和研究者。

特别感谢：
- OpenCV 社区提供的图像处理算法
- FastAPI 团队提供的高性能 Web 框架
- Next.js 团队提供的前端解决方案
- Ant Design 团队提供的 UI 组件库

## 📞 联系我们

- **项目主页**: https://github.com/your-org/sichuan-brocade-ai-tool
- **问题反馈**: https://github.com/your-org/sichuan-brocade-ai-tool/issues
- **邮箱**: support@sichuan-brocade.com
- **微信群**: [扫码加入开发者群]

---

**🧵✨ 传承千年工艺，融合现代科技 | 专业织机图像处理解决方案** 