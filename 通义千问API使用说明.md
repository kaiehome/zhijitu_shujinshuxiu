# 通义千问API集成使用说明

## 概述

本系统已成功集成通义千问大模型API，用于生成高质量的织机识别图。通义千问是阿里巴巴开发的多模态大模型，具有强大的图像理解和生成能力。

## 功能特性

### 1. 织机识别图生成
- **端点**: `/api/generate-tongyi-qianwen`
- **方法**: POST
- **功能**: 使用通义千问大模型将普通图片转换为专业的织机识别图
- **特点**: 
  - 高饱和度
  - 强对比度
  - 清晰边缘
  - 丰富的色彩层次
  - 像素化效果

### 2. 图片质量增强
- **端点**: `/api/enhance-tongyi-qianwen`
- **方法**: POST
- **功能**: 使用通义千问大模型增强图片质量
- **特点**:
  - 提升清晰度
  - 增强色彩
  - 优化对比度
  - 改善细节

### 3. API状态检查
- **端点**: `/api/tongyi-qianwen-status`
- **方法**: GET
- **功能**: 检查通义千问API的可用状态

## 配置步骤

### 1. 获取通义千问API密钥

1. 访问 [阿里云通义千问控制台](https://dashscope.console.aliyun.com/)
2. 注册/登录阿里云账号
3. 开通通义千问服务
4. 创建API密钥

### 2. 设置环境变量

```bash
# 设置通义千问API密钥
export TONGYI_API_KEY="your_api_key_here"

# 或者在启动服务器时设置
TONGYI_API_KEY="your_api_key_here" python main.py
```

### 3. 验证配置

运行测试脚本验证API配置：

```bash
python test_tongyi_qianwen.py
```

## 使用方法

### 1. 使用curl命令

```bash
# 生成织机识别图
curl -X POST -F "file=@your_image.jpg" \
  http://localhost:8000/api/generate-tongyi-qianwen \
  -o generated_loom_image.jpg

# 增强图片质量
curl -X POST -F "file=@your_image.jpg" \
  http://localhost:8000/api/enhance-tongyi-qianwen \
  -o enhanced_image.jpg

# 检查API状态
curl http://localhost:8000/api/tongyi-qianwen-status
```

### 2. 使用Python requests

```python
import requests

# 生成织机识别图
with open('your_image.jpg', 'rb') as f:
    files = {'file': f}
    response = requests.post(
        'http://localhost:8000/api/generate-tongyi-qianwen',
        files=files
    )
    
if response.status_code == 200:
    with open('generated_loom_image.jpg', 'wb') as f:
        f.write(response.content)
    print("织机识别图生成成功！")
```

### 3. 使用前端界面

在Web界面中：
1. 上传原始图片
2. 选择"通义千问生成"或"通义千问增强"
3. 点击处理按钮
4. 下载生成的图片

## 技术规格

### 支持的模型
- **qwen-vl-plus**: 通义千问视觉语言模型（推荐）
- **qwen-vl-max**: 通义千问视觉语言模型（最大版本）
- **qwen-vl-chat**: 通义千问视觉语言对话模型

### 支持的图片格式
- JPEG (.jpg, .jpeg)
- PNG (.png)
- 最大文件大小: 10MB

### 处理时间
- 图片生成: 30-60秒（取决于图片大小和复杂度）
- 图片增强: 20-40秒
- 状态检查: 1-2秒

## 错误处理

### 常见错误及解决方案

1. **API密钥未设置**
   ```
   错误: 通义千问API未配置，请设置TONGYI_API_KEY环境变量
   解决: 设置TONGYI_API_KEY环境变量
   ```

2. **API密钥无效**
   ```
   错误: 401 Unauthorized
   解决: 检查API密钥是否正确，确认服务已开通
   ```

3. **图片格式不支持**
   ```
   错误: 不支持的图片格式
   解决: 使用JPEG或PNG格式的图片
   ```

4. **文件大小超限**
   ```
   错误: 文件大小超过限制
   解决: 压缩图片或使用较小的图片
   ```

## 性能优化建议

1. **图片预处理**
   - 将图片调整为合适的分辨率（建议1024x1024以下）
   - 使用JPEG格式以减少文件大小

2. **并发控制**
   - 避免同时发送大量请求
   - 建议请求间隔至少5秒

3. **缓存策略**
   - 对相同图片的处理结果进行缓存
   - 避免重复处理

## 成本控制

通义千问API按调用次数计费：
- 图片生成: 约0.1-0.5元/次
- 图片增强: 约0.05-0.2元/次

建议：
- 在开发阶段使用测试图片
- 批量处理时控制并发数量
- 监控API使用量

## 示例效果

### 输入图片
- 普通照片或设计图

### 输出图片
- 高饱和度的织机识别图
- 强对比度的专业效果
- 清晰的边缘和细节
- 丰富的色彩层次

## 技术支持

如遇到问题，请检查：
1. API密钥是否正确设置
2. 网络连接是否正常
3. 服务器是否正常运行
4. 图片格式是否符合要求

## 更新日志

- **v1.0.0**: 初始版本，支持基本的图片生成和增强功能
- **v1.1.0**: 添加状态检查端点
- **v1.2.0**: 优化错误处理和性能 