"""
统一性能测量框架
提供装饰器式性能测量器、性能指标收集分析和性能监控仪表板
"""

import time
import functools
import logging
import json
import os
import threading
from typing import Dict, Any, Optional, List, Callable, Union
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import statistics
import psutil
import gc
from contextlib import contextmanager
from datetime import datetime, timedelta


@dataclass
class PerformanceMetric:
    """性能指标"""
    function_name: str
    execution_time: float
    memory_usage: float
    cpu_usage: float
    timestamp: float
    success: bool
    error_message: Optional[str] = None
    additional_data: Optional[Dict[str, Any]] = None


@dataclass
class PerformanceSummary:
    """性能摘要"""
    function_name: str
    total_calls: int
    successful_calls: int
    failed_calls: int
    avg_execution_time: float
    min_execution_time: float
    max_execution_time: float
    std_execution_time: float
    avg_memory_usage: float
    avg_cpu_usage: float
    success_rate: float
    last_execution: float
    first_execution: float


class PerformanceMonitor:
    """
    性能监控器
    
    提供函数级性能监控、内存使用跟踪和CPU使用率监控
    """
    
    def __init__(self, max_history: int = 1000, 
                 enable_memory_monitoring: bool = True,
                 enable_cpu_monitoring: bool = True):
        self.max_history = max_history
        self.enable_memory_monitoring = enable_memory_monitoring
        self.enable_cpu_monitoring = enable_cpu_monitoring
        
        self.metrics: deque = deque(maxlen=max_history)
        self.function_stats: Dict[str, List[PerformanceMetric]] = defaultdict(list)
        self.global_stats = {
            "total_calls": 0,
            "total_execution_time": 0,
            "total_memory_usage": 0,
            "total_cpu_usage": 0,
            "start_time": time.time()
        }
        
        self.logger = logging.getLogger(__name__)
        self._lock = threading.Lock()
        
        # 初始化系统监控
        self._init_system_monitoring()
    
    def _init_system_monitoring(self):
        """初始化系统监控"""
        try:
            # 获取初始系统状态
            self.initial_memory = psutil.virtual_memory().used
            self.initial_cpu_percent = psutil.cpu_percent()
        except Exception as e:
            self.logger.warning(f"系统监控初始化失败: {e}")
            self.enable_memory_monitoring = False
            self.enable_cpu_monitoring = False
    
    def record_metric(self, metric: PerformanceMetric):
        """记录性能指标"""
        with self._lock:
            self.metrics.append(metric)
            self.function_stats[metric.function_name].append(metric)
            
            # 更新全局统计
            self.global_stats["total_calls"] += 1
            self.global_stats["total_execution_time"] += metric.execution_time
            self.global_stats["total_memory_usage"] += metric.memory_usage
            self.global_stats["total_cpu_usage"] += metric.cpu_usage
    
    def get_function_summary(self, function_name: str) -> Optional[PerformanceSummary]:
        """获取函数性能摘要"""
        if function_name not in self.function_stats:
            return None
        
        metrics = self.function_stats[function_name]
        if not metrics:
            return None
        
        execution_times = [m.execution_time for m in metrics]
        memory_usages = [m.memory_usage for m in metrics]
        cpu_usages = [m.cpu_usage for m in metrics]
        successful_calls = sum(1 for m in metrics if m.success)
        
        return PerformanceSummary(
            function_name=function_name,
            total_calls=len(metrics),
            successful_calls=successful_calls,
            failed_calls=len(metrics) - successful_calls,
            avg_execution_time=statistics.mean(execution_times),
            min_execution_time=min(execution_times),
            max_execution_time=max(execution_times),
            std_execution_time=statistics.stdev(execution_times) if len(execution_times) > 1 else 0,
            avg_memory_usage=statistics.mean(memory_usages) if memory_usages else 0,
            avg_cpu_usage=statistics.mean(cpu_usages) if cpu_usages else 0,
            success_rate=successful_calls / len(metrics),
            last_execution=metrics[-1].timestamp,
            first_execution=metrics[0].timestamp
        )
    
    def get_global_summary(self) -> Dict[str, Any]:
        """获取全局性能摘要"""
        with self._lock:
            total_calls = self.global_stats["total_calls"]
            if total_calls == 0:
                return {"message": "无性能数据"}
            
            avg_execution_time = self.global_stats["total_execution_time"] / total_calls
            avg_memory_usage = self.global_stats["total_memory_usage"] / total_calls
            avg_cpu_usage = self.global_stats["total_cpu_usage"] / total_calls
            
            return {
                "total_calls": total_calls,
                "avg_execution_time": avg_execution_time,
                "avg_memory_usage": avg_memory_usage,
                "avg_cpu_usage": avg_cpu_usage,
                "uptime": time.time() - self.global_stats["start_time"],
                "function_count": len(self.function_stats)
            }
    
    def get_slowest_functions(self, limit: int = 10) -> List[PerformanceSummary]:
        """获取最慢的函数"""
        summaries = []
        for function_name in self.function_stats:
            summary = self.get_function_summary(function_name)
            if summary:
                summaries.append(summary)
        
        return sorted(summaries, key=lambda x: x.avg_execution_time, reverse=True)[:limit]
    
    def get_most_called_functions(self, limit: int = 10) -> List[PerformanceSummary]:
        """获取调用最频繁的函数"""
        summaries = []
        for function_name in self.function_stats:
            summary = self.get_function_summary(function_name)
            if summary:
                summaries.append(summary)
        
        return sorted(summaries, key=lambda x: x.total_calls, reverse=True)[:limit]
    
    def clear_history(self):
        """清除历史数据"""
        with self._lock:
            self.metrics.clear()
            self.function_stats.clear()
            self.global_stats = {
                "total_calls": 0,
                "total_execution_time": 0,
                "total_memory_usage": 0,
                "total_cpu_usage": 0,
                "start_time": time.time()
            }
    
    def export_data(self, filepath: str):
        """导出性能数据"""
        try:
            data = {
                "global_stats": self.global_stats,
                "function_summaries": {},
                "recent_metrics": [asdict(m) for m in list(self.metrics)[-100:]]
            }
            
            for function_name in self.function_stats:
                summary = self.get_function_summary(function_name)
                if summary:
                    data["function_summaries"][function_name] = asdict(summary)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, default=str)
            
            self.logger.info(f"性能数据已导出到: {filepath}")
            
        except Exception as e:
            self.logger.error(f"导出性能数据失败: {e}")


class PerformanceDecorator:
    """性能监控装饰器"""
    
    def __init__(self, monitor: PerformanceMonitor, 
                 function_name: Optional[str] = None,
                 enable_memory_tracking: bool = True,
                 enable_cpu_tracking: bool = True):
        self.monitor = monitor
        self.function_name = function_name
        self.enable_memory_tracking = enable_memory_tracking
        self.enable_cpu_tracking = enable_cpu_tracking
    
    def __call__(self, func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            function_name = self.function_name or func.__name__
            
            # 记录开始状态
            start_time = time.perf_counter()
            start_memory = self._get_memory_usage() if self.enable_memory_tracking else 0
            start_cpu = self._get_cpu_usage() if self.enable_cpu_tracking else 0
            
            success = True
            error_message = None
            
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                success = False
                error_message = str(e)
                raise
            finally:
                # 记录结束状态
                end_time = time.perf_counter()
                end_memory = self._get_memory_usage() if self.enable_memory_tracking else 0
                end_cpu = self._get_cpu_usage() if self.enable_cpu_tracking else 0
                
                # 计算指标
                execution_time = end_time - start_time
                memory_usage = end_memory - start_memory if self.enable_memory_tracking else 0
                cpu_usage = end_cpu - start_cpu if self.enable_cpu_tracking else 0
                
                # 创建性能指标
                metric = PerformanceMetric(
                    function_name=function_name,
                    execution_time=execution_time,
                    memory_usage=memory_usage,
                    cpu_usage=cpu_usage,
                    timestamp=time.time(),
                    success=success,
                    error_message=error_message,
                    additional_data={
                        "args_count": len(args),
                        "kwargs_keys": list(kwargs.keys())
                    }
                )
                
                # 记录指标
                self.monitor.record_metric(metric)
        
        return wrapper
    
    def _get_memory_usage(self) -> float:
        """获取内存使用量（MB）"""
        try:
            process = psutil.Process()
            return process.memory_info().rss / (1024 * 1024)
        except:
            return 0
    
    def _get_cpu_usage(self) -> float:
        """获取CPU使用率"""
        try:
            return psutil.cpu_percent()
        except:
            return 0


@contextmanager
def performance_context(monitor: PerformanceMonitor, 
                       context_name: str,
                       enable_memory_tracking: bool = True,
                       enable_cpu_tracking: bool = True):
    """
    性能上下文管理器
    
    用法:
    with performance_context(monitor, "image_processing"):
        # 需要监控的代码块
        result = process_image(image)
    """
    start_time = time.perf_counter()
    start_memory = monitor._get_memory_usage() if enable_memory_tracking else 0
    start_cpu = monitor._get_cpu_usage() if enable_cpu_tracking else 0
    
    success = True
    error_message = None
    
    try:
        yield
    except Exception as e:
        success = False
        error_message = str(e)
        raise
    finally:
        end_time = time.perf_counter()
        end_memory = monitor._get_memory_usage() if enable_memory_tracking else 0
        end_cpu = monitor._get_cpu_usage() if enable_cpu_tracking else 0
        
        execution_time = end_time - start_time
        memory_usage = end_memory - start_memory if enable_memory_tracking else 0
        cpu_usage = end_cpu - start_cpu if enable_cpu_tracking else 0
        
        metric = PerformanceMetric(
            function_name=context_name,
            execution_time=execution_time,
            memory_usage=memory_usage,
            cpu_usage=cpu_usage,
            timestamp=time.time(),
            success=success,
            error_message=error_message
        )
        
        monitor.record_metric(metric)


class PerformanceBenchmark:
    """性能基准测试"""
    
    def __init__(self, monitor: PerformanceMonitor):
        self.monitor = monitor
        self.benchmarks = {}
    
    def benchmark_function(self, func: Callable, 
                          test_cases: List[tuple],
                          iterations: int = 1,
                          warmup_iterations: int = 0) -> Dict[str, Any]:
        """
        对函数进行基准测试
        
        Args:
            func: 要测试的函数
            test_cases: 测试用例列表
            iterations: 每个测试用例的迭代次数
            warmup_iterations: 预热迭代次数
            
        Returns:
            Dict: 基准测试结果
        """
        function_name = func.__name__
        results = {
            "function_name": function_name,
            "test_cases": len(test_cases),
            "iterations": iterations,
            "warmup_iterations": warmup_iterations,
            "results": []
        }
        
        # 预热
        if warmup_iterations > 0:
            for _ in range(warmup_iterations):
                for args in test_cases:
                    try:
                        func(*args)
                    except:
                        pass
        
        # 执行测试
        for i, args in enumerate(test_cases):
            case_results = []
            
            for j in range(iterations):
                start_time = time.perf_counter()
                start_memory = self._get_memory_usage()
                
                try:
                    result = func(*args)
                    success = True
                    error = None
                except Exception as e:
                    result = None
                    success = False
                    error = str(e)
                
                end_time = time.perf_counter()
                end_memory = self._get_memory_usage()
                
                case_results.append({
                    "iteration": j + 1,
                    "execution_time": end_time - start_time,
                    "memory_usage": end_memory - start_memory,
                    "success": success,
                    "error": error,
                    "result": result
                })
            
            # 计算统计信息
            execution_times = [r["execution_time"] for r in case_results if r["success"]]
            memory_usages = [r["memory_usage"] for r in case_results if r["success"]]
            success_count = sum(1 for r in case_results if r["success"])
            
            results["results"].append({
                "test_case": i + 1,
                "args": str(args),
                "avg_execution_time": statistics.mean(execution_times) if execution_times else 0,
                "min_execution_time": min(execution_times) if execution_times else 0,
                "max_execution_time": max(execution_times) if execution_times else 0,
                "avg_memory_usage": statistics.mean(memory_usages) if memory_usages else 0,
                "success_rate": success_count / len(case_results),
                "iterations": case_results
            })
        
        self.benchmarks[function_name] = results
        return results
    
    def get_benchmark_results(self, function_name: str) -> Optional[Dict[str, Any]]:
        """获取基准测试结果"""
        return self.benchmarks.get(function_name)
    
    def export_benchmark_results(self, filepath: str):
        """导出基准测试结果"""
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(self.benchmarks, f, indent=2, default=str)
        except Exception as e:
            logging.error(f"导出基准测试结果失败: {e}")
    
    def _get_memory_usage(self) -> float:
        """获取内存使用量（MB）"""
        try:
            process = psutil.Process()
            return process.memory_info().rss / (1024 * 1024)
        except:
            return 0
    
    def _get_cpu_usage(self) -> float:
        """获取CPU使用率"""
        try:
            return psutil.cpu_percent()
        except:
            return 0


# 全局性能监控器实例
_global_monitor = None

def get_global_monitor() -> PerformanceMonitor:
    """获取全局性能监控器"""
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = PerformanceMonitor()
    return _global_monitor

def set_global_monitor(monitor: PerformanceMonitor):
    """设置全局性能监控器"""
    global _global_monitor
    _global_monitor = monitor

def monitor_performance(function_name: Optional[str] = None,
                       enable_memory_tracking: bool = True,
                       enable_cpu_tracking: bool = True):
    """性能监控装饰器"""
    return PerformanceDecorator(
        get_global_monitor(),
        function_name,
        enable_memory_tracking,
        enable_cpu_tracking
    )


# 测试函数
def test_performance_monitor():
    """测试性能监控器"""
    monitor = PerformanceMonitor()
    
    print("=== 性能监控器测试 ===")
    
    # 测试装饰器
    @monitor_performance("test_function")
    def test_function(x):
        time.sleep(0.1)  # 模拟工作
        return x * 2
    
    # 执行测试
    for i in range(5):
        try:
            result = test_function(i)
            print(f"测试 {i}: 结果 = {result}")
        except Exception as e:
            print(f"测试 {i} 失败: {e}")
    
    # 显示结果
    summary = monitor.get_function_summary("test_function")
    if summary:
        print(f"函数摘要: {asdict(summary)}")
    
    global_summary = monitor.get_global_summary()
    print(f"全局摘要: {global_summary}")
    
    # 测试基准测试
    benchmark = PerformanceBenchmark(monitor)
    
    def benchmark_function(x):
        time.sleep(0.05)
        return x * x
    
    test_cases = [(1,), (2,), (3,)]
    results = benchmark.benchmark_function(benchmark_function, test_cases, iterations=3)
    print(f"基准测试结果: {results}")


if __name__ == "__main__":
    test_performance_monitor() 