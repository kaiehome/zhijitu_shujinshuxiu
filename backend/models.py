"""
蜀锦蜀绣AI打样图生成工具 - 数据模型
定义API请求和响应的数据结构，包含验证逻辑
"""

import re
import time
from pathlib import Path
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, validator


class ProcessRequest(BaseModel):
    """图像处理请求模型"""
    filename: str = Field(..., description="上传的文件名", min_length=1, max_length=255)
    color_count: Optional[int] = Field(16, description="目标颜色数量", ge=10, le=20)
    edge_enhancement: Optional[bool] = Field(True, description="是否启用边缘增强")
    noise_reduction: Optional[bool] = Field(True, description="是否启用噪声清理")
    style: Optional[str] = Field("sichuan_brocade", description="处理风格")
    professional_mode: Optional[bool] = Field(True, description="是否使用专业织机模式")
    
    @validator('filename')
    def validate_filename(cls, v):
        """验证文件名安全性"""
        if not v or v.strip() == "":
            raise ValueError("文件名不能为空")
        
        # 检查危险字符
        dangerous_chars = ['..', '/', '\\', ':', '*', '?', '"', '<', '>', '|']
        for char in dangerous_chars:
            if char in v:
                raise ValueError(f"文件名包含非法字符: {char}")
        
        return v.strip()
    
    @validator('color_count')
    def validate_color_count(cls, v):
        """验证颜色数量"""
        allowed_values = [10, 12, 14, 16, 18, 20]
        if v not in allowed_values:
            raise ValueError(f"颜色数量必须是以下值之一: {', '.join(map(str, allowed_values))}")
        return v
    
    @validator('style')
    def validate_style(cls, v):
        """验证处理风格"""
        allowed_styles = ["sichuan_brocade", "traditional", "modern"]
        if v not in allowed_styles:
            raise ValueError(f"不支持的风格: {v}")
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "filename": "example_pattern.png",
                "color_count": 16,
                "edge_enhancement": True,
                "noise_reduction": True,
                "style": "sichuan_brocade",
                "professional_mode": True
            }
        }


class ProcessResponse(BaseModel):
    """图像处理响应模型"""
    job_id: str = Field(..., description="任务唯一标识符")
    status: str = Field(..., description="任务状态: processing/completed/failed")
    message: str = Field(..., description="状态描述信息")
    original_filename: str = Field(..., description="原始文件名")
    processed_files: Optional[List[str]] = Field(None, description="处理结果文件路径列表")
    processing_time: Optional[float] = Field(None, description="处理耗时（秒）")
    
    @validator('status')
    def validate_status(cls, v):
        """验证任务状态"""
        allowed_statuses = ["processing", "completed", "failed", "pending"]
        if v not in allowed_statuses:
            raise ValueError(f"无效的任务状态: {v}")
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "job_id": "1640995200000",
                "status": "completed",
                "message": "图像处理完成",
                "original_filename": "example_pattern.png",
                "processed_files": [
                    "1640995200000/1640995200000_processed.png",
                    "1640995200000/1640995200000_pattern.svg"
                ],
                "processing_time": 15.32
            }
        }


class TextToPatternRequest(BaseModel):
    """文生图请求模型（预留功能）"""
    prompt: str = Field(..., description="文本提示词", min_length=1, max_length=500)
    style: str = Field("sichuan_brocade", description="生成风格")
    size: Optional[str] = Field("1024x1024", description="图像尺寸")
    color_count: Optional[int] = Field(16, description="颜色数量", ge=10, le=20)
    
    @validator('prompt')
    def validate_prompt(cls, v):
        """验证提示词内容"""
        if not v or v.strip() == "":
            raise ValueError("提示词不能为空")
        
        # 过滤敏感内容（简单示例）
        forbidden_words = ["暴力", "色情", "政治"]
        for word in forbidden_words:
            if word in v:
                raise ValueError(f"提示词包含不当内容: {word}")
        
        return v.strip()
    
    @validator('color_count')
    def validate_color_count(cls, v):
        """验证颜色数量"""
        allowed_values = [10, 12, 14, 16, 18, 20]
        if v not in allowed_values:
            raise ValueError(f"颜色数量必须是以下值之一: {', '.join(map(str, allowed_values))}")
        return v
    
    @validator('size')
    def validate_size(cls, v):
        """验证图像尺寸格式"""
        if not re.match(r'^\d+x\d+$', v):
            raise ValueError("图像尺寸格式错误，应为 '宽度x高度'")
        
        width, height = map(int, v.split('x'))
        if width < 256 or height < 256 or width > 2048 or height > 2048:
            raise ValueError("图像尺寸应在256x256到2048x2048之间")
        
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "prompt": "传统蜀锦花鸟图案，富有层次感",
                "style": "sichuan_brocade",
                "size": "1024x1024",
                "color_count": 16
            }
        }


class UploadResponse(BaseModel):
    """文件上传响应模型"""
    filename: str = Field(..., description="服务器存储的文件名")
    size: int = Field(..., description="文件大小（字节）")
    content_type: str = Field(..., description="文件MIME类型")
    upload_time: str = Field(..., description="上传时间（ISO格式）")
    message: str = Field(..., description="上传结果消息")
    
    @validator('size')
    def validate_size(cls, v):
        """验证文件大小"""
        if v <= 0:
            raise ValueError("文件大小必须大于0")
        if v > 10 * 1024 * 1024:  # 10MB
            raise ValueError("文件大小超过10MB限制")
        return v
    
    @validator('content_type')
    def validate_content_type(cls, v):
        """验证文件类型"""
        allowed_types = ["image/jpeg", "image/png", "image/jpg"]
        if v not in allowed_types:
            raise ValueError(f"不支持的文件类型: {v}")
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "filename": "1640995200000.png",
                "size": 2048576,
                "content_type": "image/png",
                "upload_time": "2023-12-01T10:30:00",
                "message": "文件上传成功"
            }
        }


class ErrorResponse(BaseModel):
    """错误响应模型"""
    error: str = Field(..., description="错误类型")
    message: str = Field(..., description="错误详细信息")
    timestamp: str = Field(..., description="错误发生时间")
    request_id: Optional[str] = Field(None, description="请求追踪ID")
    
    class Config:
        schema_extra = {
            "example": {
                "error": "ValidationError",
                "message": "文件格式不支持",
                "timestamp": "2023-12-01T10:30:00",
                "request_id": "req_1640995200000"
            }
        }


# 工具函数
def generate_job_id() -> str:
    """
    生成用户友好的任务ID
    格式：YYMMDD_HHMMSS
    例子：241225_153045 (24年12月25日 15:30:45)
    
    Returns:
        str: 人类可读的任务ID
    """
    from datetime import datetime
    
    now = datetime.now()
    return now.strftime("%y%m%d_%H%M%S")


def validate_image_file(filename: str, content_type: str) -> Dict[str, Any]:
    """
    验证图像文件的合法性
    
    Args:
        filename: 文件名
        content_type: MIME类型
        
    Returns:
        dict: 包含验证结果的字典
    """
    result = {"valid": False, "error": None}
    
    try:
        # 检查文件名
        if not filename:
            result["error"] = "文件名不能为空"
            return result
        
        # 检查文件扩展名
        allowed_extensions = {".jpg", ".jpeg", ".png"}
        file_extension = Path(filename).suffix.lower()
        
        if file_extension not in allowed_extensions:
            result["error"] = f"不支持的文件格式: {file_extension}，仅支持 JPG, PNG"
            return result
        
        # 检查MIME类型
        allowed_mime_types = {"image/jpeg", "image/png", "image/jpg"}
        if content_type not in allowed_mime_types:
            result["error"] = f"不支持的文件类型: {content_type}"
            return result
        
        # 检查文件名长度
        if len(filename) > 255:
            result["error"] = "文件名过长"
            return result
        
        # 检查危险字符
        dangerous_chars = ['..', '/', '\\', ':', '*', '?', '"', '<', '>', '|']
        for char in dangerous_chars:
            if char in filename:
                result["error"] = f"文件名包含非法字符: {char}"
                return result
        
        result["valid"] = True
        return result
        
    except Exception as e:
        result["error"] = f"文件验证失败: {str(e)}"
        return result


def sanitize_filename(filename: str) -> str:
    """
    清理文件名，移除危险字符
    
    Args:
        filename: 原始文件名
        
    Returns:
        str: 清理后的安全文件名
    """
    if not filename:
        return ""
    
    # 移除危险字符
    dangerous_chars = ['..', '/', '\\', ':', '*', '?', '"', '<', '>', '|']
    clean_name = filename
    
    for char in dangerous_chars:
        clean_name = clean_name.replace(char, '_')
    
    # 移除多余的空格和点
    clean_name = re.sub(r'[.\s]+', '.', clean_name.strip())
    
    # 确保不以点开头或结尾
    clean_name = clean_name.strip('.')
    
    # 限制长度
    if len(clean_name) > 255:
        name_part = Path(clean_name).stem[:200]
        extension = Path(clean_name).suffix
        clean_name = f"{name_part}{extension}"
    
    return clean_name


def validate_job_id(job_id: str) -> bool:
    """
    验证任务ID格式
    支持两种格式：
    1. 新格式：YYMMDD_HHMMSS (例：241225_153045)
    2. 旧格式：13位时间戳 (例：1750836509370)
    
    Args:
        job_id: 任务ID
        
    Returns:
        bool: 是否为有效的任务ID
    """
    if not job_id:
        return False
    
    # 检查新格式：YYMMDD_HHMMSS
    if re.match(r'^\d{6}_\d{6}$', job_id):
        try:
            # 解析日期时间部分
            date_part, time_part = job_id.split('_')
            
            # 验证日期格式
            year = int('20' + date_part[:2])  # YY -> YYYY
            month = int(date_part[2:4])
            day = int(date_part[4:6])
            
            hour = int(time_part[:2])
            minute = int(time_part[2:4])
            second = int(time_part[4:6])
            
            # 基本范围检查
            if not (2024 <= year <= 2030):  # 合理的年份范围
                return False
            if not (1 <= month <= 12):
                return False
            if not (1 <= day <= 31):
                return False
            if not (0 <= hour <= 23):
                return False
            if not (0 <= minute <= 59):
                return False
            if not (0 <= second <= 59):
                return False
                
            # 尝试创建datetime对象验证日期有效性
            from datetime import datetime
            task_time = datetime(year, month, day, hour, minute, second)
            
            # 检查是否为未来时间（允许5分钟误差）
            from datetime import datetime, timedelta
            current_time = datetime.now()
            if task_time > current_time + timedelta(minutes=5):
                return False
                
            return True
            
        except (ValueError, OverflowError):
            return False
    
    # 检查旧格式：13位数字（时间戳）- 兼容性支持
    elif re.match(r'^\d{13}$', job_id):
        try:
            timestamp = int(job_id) / 1000
            current_time = time.time()
            
            # 不能是未来时间
            if timestamp > current_time:
                return False
            
            # 不能超过30天前
            if current_time - timestamp > 30 * 24 * 3600:
                return False
            
            return True
            
        except (ValueError, OverflowError):
            return False
    
    return False


def get_file_info(file_path: Path) -> Dict[str, Any]:
    """
    获取文件基本信息
    
    Args:
        file_path: 文件路径
        
    Returns:
        dict: 文件信息字典
    """
    try:
        if not file_path.exists():
            return {"exists": False}
        
        stat_info = file_path.stat()
        
        return {
            "exists": True,
            "size": stat_info.st_size,
            "created": stat_info.st_ctime,
            "modified": stat_info.st_mtime,
            "extension": file_path.suffix.lower(),
            "name": file_path.name
        }
        
    except Exception as e:
        return {"exists": False, "error": str(e)}


# 常量定义
class Constants:
    """应用常量"""
    
    # 文件限制
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
    ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
    ALLOWED_MIME_TYPES = {"image/jpeg", "image/png", "image/jpg"}
    
    # 处理参数限制
    MIN_COLOR_COUNT = 10
    MAX_COLOR_COUNT = 20
    DEFAULT_COLOR_COUNT = 16
    
    # 任务管理
    MAX_CONCURRENT_JOBS = 10
    JOB_CLEANUP_HOURS = 24
    MAX_JOB_ID_AGE_DAYS = 30
    
    # 安全设置
    MAX_FILENAME_LENGTH = 255
    MAX_PROMPT_LENGTH = 500
    
    # 支持的处理风格
    SUPPORTED_STYLES = ["sichuan_brocade", "traditional", "modern"]
    
    # 状态码
    STATUS_PROCESSING = "processing"
    STATUS_COMPLETED = "completed"
    STATUS_FAILED = "failed"
    STATUS_PENDING = "pending" 