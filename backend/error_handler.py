"""
统一错误处理模块
提供分层异常体系、结构化错误响应和错误上下文管理
"""

import logging
import traceback
import sys
import time
from typing import Dict, Any, Optional, List, Union
from enum import Enum
import json
import os
from dataclasses import dataclass, asdict
from contextlib import contextmanager


class ErrorSeverity(Enum):
    """错误严重程度"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """错误分类"""
    VALIDATION = "validation"
    PROCESSING = "processing"
    SYSTEM = "system"
    NETWORK = "network"
    RESOURCE = "resource"
    GPU = "gpu"
    MEMORY = "memory"
    UNKNOWN = "unknown"


@dataclass
class ErrorContext:
    """错误上下文信息"""
    timestamp: float
    function_name: str
    line_number: int
    file_name: str
    module_name: str
    thread_id: Optional[int] = None
    process_id: Optional[int] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    request_id: Optional[str] = None
    additional_data: Optional[Dict[str, Any]] = None


@dataclass
class ErrorInfo:
    """错误信息"""
    error_type: str
    error_message: str
    severity: ErrorSeverity
    category: ErrorCategory
    context: ErrorContext
    stack_trace: str
    original_exception: Optional[Exception] = None
    error_code: Optional[str] = None
    suggestions: Optional[List[str]] = None


class BaseError(Exception):
    """基础错误类"""
    
    def __init__(self, message: str, severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                 category: ErrorCategory = ErrorCategory.UNKNOWN, 
                 error_code: Optional[str] = None, suggestions: Optional[List[str]] = None):
        super().__init__(message)
        self.severity = severity
        self.category = category
        self.error_code = error_code
        self.suggestions = suggestions or []
        self.context = None
        self.stack_trace = None


class ValidationError(BaseError):
    """验证错误"""
    def __init__(self, message: str, error_code: Optional[str] = None, suggestions: Optional[List[str]] = None):
        super().__init__(message, ErrorSeverity.MEDIUM, ErrorCategory.VALIDATION, error_code, suggestions)


class ProcessingError(BaseError):
    """处理错误"""
    def __init__(self, message: str, error_code: Optional[str] = None, suggestions: Optional[List[str]] = None):
        super().__init__(message, ErrorSeverity.HIGH, ErrorCategory.PROCESSING, error_code, suggestions)


class SystemError(BaseError):
    """系统错误"""
    def __init__(self, message: str, error_code: Optional[str] = None, suggestions: Optional[List[str]] = None):
        super().__init__(message, ErrorSeverity.CRITICAL, ErrorCategory.SYSTEM, error_code, suggestions)


class ResourceError(BaseError):
    """资源错误"""
    def __init__(self, message: str, error_code: Optional[str] = None, suggestions: Optional[List[str]] = None):
        super().__init__(message, ErrorSeverity.HIGH, ErrorCategory.RESOURCE, error_code, suggestions)


class GPUError(BaseError):
    """GPU错误"""
    def __init__(self, message: str, error_code: Optional[str] = None, suggestions: Optional[List[str]] = None):
        super().__init__(message, ErrorSeverity.MEDIUM, ErrorCategory.GPU, error_code, suggestions)


class MemoryError(BaseError):
    """内存错误"""
    def __init__(self, message: str, error_code: Optional[str] = None, suggestions: Optional[List[str]] = None):
        super().__init__(message, ErrorSeverity.HIGH, ErrorCategory.MEMORY, error_code, suggestions)


class ErrorHandler:
    """
    统一错误处理器
    
    提供错误捕获、分类、记录和响应功能
    """
    
    def __init__(self, log_file: Optional[str] = None, 
                 enable_console_logging: bool = True,
                 enable_file_logging: bool = True):
        self.logger = logging.getLogger(__name__)
        self.log_file = log_file
        self.enable_console_logging = enable_console_logging
        self.enable_file_logging = enable_file_logging
        self.error_history: List[ErrorInfo] = []
        self.error_stats = {
            "total_errors": 0,
            "errors_by_category": {},
            "errors_by_severity": {},
            "errors_by_type": {}
        }
        
        self._setup_logging()
    
    def _setup_logging(self):
        """设置日志记录"""
        if not self.logger.handlers:
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            
            if self.enable_console_logging:
                console_handler = logging.StreamHandler()
                console_handler.setFormatter(formatter)
                self.logger.addHandler(console_handler)
            
            if self.enable_file_logging and self.log_file:
                file_handler = logging.FileHandler(self.log_file)
                file_handler.setFormatter(formatter)
                self.logger.addHandler(file_handler)
            
            self.logger.setLevel(logging.DEBUG)
    
    def capture_error(self, exception: Exception, 
                     function_name: Optional[str] = None,
                     additional_data: Optional[Dict[str, Any]] = None) -> ErrorInfo:
        """
        捕获并分析错误
        
        Args:
            exception: 异常对象
            function_name: 函数名称
            additional_data: 额外数据
            
        Returns:
            ErrorInfo: 错误信息对象
        """
        # 获取错误上下文
        context = self._get_error_context(function_name, additional_data)
        
        # 分析错误类型
        error_type, severity, category, error_code, suggestions = self._analyze_error(exception)
        
        # 获取堆栈跟踪
        stack_trace = self._get_stack_trace(exception)
        
        # 创建错误信息
        error_info = ErrorInfo(
            error_type=error_type,
            error_message=str(exception),
            severity=severity,
            category=category,
            context=context,
            stack_trace=stack_trace,
            original_exception=exception,
            error_code=error_code,
            suggestions=suggestions
        )
        
        # 记录错误
        self._log_error(error_info)
        
        # 更新统计
        self._update_stats(error_info)
        
        # 添加到历史
        self.error_history.append(error_info)
        
        return error_info
    
    def _get_error_context(self, function_name: Optional[str], 
                          additional_data: Optional[Dict[str, Any]]) -> ErrorContext:
        """获取错误上下文"""
        frame = sys.exc_info()[2]
        if frame:
            filename = frame.tb_frame.f_code.co_filename
            line_number = frame.tb_lineno
            module_name = frame.tb_frame.f_globals.get('__name__', 'unknown')
        else:
            filename = 'unknown'
            line_number = 0
            module_name = 'unknown'
        
        return ErrorContext(
            timestamp=time.time(),
            function_name=function_name or 'unknown',
            line_number=line_number,
            file_name=os.path.basename(filename),
            module_name=module_name,
            additional_data=additional_data
        )
    
    def _analyze_error(self, exception: Exception) -> tuple:
        """分析错误类型和严重程度"""
        error_type = type(exception).__name__
        
        # 检查是否是自定义错误
        if isinstance(exception, BaseError):
            return (error_type, exception.severity, exception.category, 
                   exception.error_code, exception.suggestions)
        
        # 根据异常类型分类
        if isinstance(exception, (ValueError, TypeError, AttributeError)):
            return (error_type, ErrorSeverity.MEDIUM, ErrorCategory.VALIDATION, 
                   "VAL_001", ["检查输入参数类型和值", "验证数据格式"])
        
        elif isinstance(exception, (FileNotFoundError, PermissionError, OSError)):
            return (error_type, ErrorSeverity.HIGH, ErrorCategory.RESOURCE, 
                   "RES_001", ["检查文件路径和权限", "确保资源可用"])
        
        elif isinstance(exception, (MemoryError, OverflowError)):
            return (error_type, ErrorSeverity.HIGH, ErrorCategory.MEMORY, 
                   "MEM_001", ["减少内存使用", "优化算法", "增加系统内存"])
        
        elif isinstance(exception, (ConnectionError, TimeoutError)):
            return (error_type, ErrorSeverity.MEDIUM, ErrorCategory.NETWORK, 
                   "NET_001", ["检查网络连接", "重试操作", "增加超时时间"])
        
        elif isinstance(exception, (ImportError, ModuleNotFoundError)):
            return (error_type, ErrorSeverity.HIGH, ErrorCategory.SYSTEM, 
                   "SYS_001", ["安装缺失的依赖", "检查Python环境", "更新包版本"])
        
        else:
            return (error_type, ErrorSeverity.MEDIUM, ErrorCategory.UNKNOWN, 
                   "UNK_001", ["检查错误日志", "联系技术支持"])
    
    def _get_stack_trace(self, exception: Exception) -> str:
        """获取堆栈跟踪"""
        return ''.join(traceback.format_exception(type(exception), exception, exception.__traceback__))
    
    def _log_error(self, error_info: ErrorInfo):
        """记录错误"""
        log_message = (f"错误类型: {error_info.error_type}, "
                      f"严重程度: {error_info.severity.value}, "
                      f"分类: {error_info.category.value}, "
                      f"消息: {error_info.error_message}")
        
        if error_info.severity == ErrorSeverity.CRITICAL:
            self.logger.critical(log_message)
        elif error_info.severity == ErrorSeverity.HIGH:
            self.logger.error(log_message)
        elif error_info.severity == ErrorSeverity.MEDIUM:
            self.logger.warning(log_message)
        else:
            self.logger.info(log_message)
        
        # 记录详细信息
        self.logger.debug(f"错误详情: {json.dumps(asdict(error_info), indent=2, default=str)}")
    
    def _update_stats(self, error_info: ErrorInfo):
        """更新错误统计"""
        self.error_stats["total_errors"] += 1
        
        # 按分类统计
        category = error_info.category.value
        self.error_stats["errors_by_category"][category] = \
            self.error_stats["errors_by_category"].get(category, 0) + 1
        
        # 按严重程度统计
        severity = error_info.severity.value
        self.error_stats["errors_by_severity"][severity] = \
            self.error_stats["errors_by_severity"].get(severity, 0) + 1
        
        # 按类型统计
        error_type = error_info.error_type
        self.error_stats["errors_by_type"][error_type] = \
            self.error_stats["errors_by_type"].get(error_type, 0) + 1
    
    def get_error_response(self, error_info: ErrorInfo) -> Dict[str, Any]:
        """获取错误响应"""
        return {
            "success": False,
            "error": {
                "type": error_info.error_type,
                "message": error_info.error_message,
                "code": error_info.error_code,
                "severity": error_info.severity.value,
                "category": error_info.category.value,
                "suggestions": error_info.suggestions,
                "timestamp": error_info.context.timestamp
            }
        }
    
    def get_error_stats(self) -> Dict[str, Any]:
        """获取错误统计"""
        return self.error_stats.copy()
    
    def get_recent_errors(self, limit: int = 10) -> List[ErrorInfo]:
        """获取最近的错误"""
        return self.error_history[-limit:]
    
    def clear_history(self):
        """清除错误历史"""
        self.error_history.clear()
    
    @contextmanager
    def error_context(self, function_name: Optional[str] = None,
                     additional_data: Optional[Dict[str, Any]] = None):
        """
        错误上下文管理器
        
        用法:
        with error_handler.error_context("process_image", {"image_size": "1024x1024"}):
            # 可能出错的代码
            result = process_image(image)
        """
        try:
            yield
        except Exception as e:
            error_info = self.capture_error(e, function_name, additional_data)
            raise error_info.original_exception from e


class ErrorDecorator:
    """错误处理装饰器"""
    
    def __init__(self, error_handler: ErrorHandler):
        self.error_handler = error_handler
    
    def __call__(self, func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                error_info = self.error_handler.capture_error(
                    e, func.__name__, 
                    {"args_count": len(args), "kwargs_keys": list(kwargs.keys())}
                )
                raise error_info.original_exception from e
        return wrapper


# 全局错误处理器实例
_global_error_handler = None

def get_global_error_handler() -> ErrorHandler:
    """获取全局错误处理器"""
    global _global_error_handler
    if _global_error_handler is None:
        _global_error_handler = ErrorHandler(
            log_file="error_logs.txt",
            enable_console_logging=True,
            enable_file_logging=True
        )
    return _global_error_handler

def set_global_error_handler(error_handler: ErrorHandler):
    """设置全局错误处理器"""
    global _global_error_handler
    _global_error_handler = error_handler

def handle_error(func):
    """错误处理装饰器"""
    return ErrorDecorator(get_global_error_handler())(func)


# 测试函数
def test_error_handler():
    """测试错误处理器"""
    error_handler = ErrorHandler()
    
    print("=== 错误处理器测试 ===")
    
    # 测试不同类型的错误
    try:
        raise ValidationError("输入参数无效", "VAL_001", ["检查参数类型", "验证数据格式"])
    except Exception as e:
        error_info = error_handler.capture_error(e, "test_validation")
        print(f"验证错误: {error_info.error_message}")
    
    try:
        raise ProcessingError("图像处理失败", "PROC_001", ["检查图像格式", "重试处理"])
    except Exception as e:
        error_info = error_handler.capture_error(e, "test_processing")
        print(f"处理错误: {error_info.error_message}")
    
    try:
        raise ResourceError("内存不足", "RES_001", ["减少内存使用", "优化算法"])
    except Exception as e:
        error_info = error_handler.capture_error(e, "test_resource")
        print(f"资源错误: {error_info.error_message}")
    
    # 显示统计
    print(f"错误统计: {error_handler.get_error_stats()}")


if __name__ == "__main__":
    test_error_handler() 