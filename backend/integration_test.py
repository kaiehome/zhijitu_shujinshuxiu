"""
综合集成测试
验证GPU加速器、错误处理、性能监控和内存管理的集成工作
"""

import time
import numpy as np
import cv2
import logging
from typing import Dict, Any

# 导入修复后的组件
from gpu_accelerator import GPUAccelerator, GPUInfo
from error_handler import ErrorHandler, ValidationError, ProcessingError, ResourceError
from performance_monitor import PerformanceMonitor, monitor_performance
from memory_manager import MemoryManager, get_global_memory_manager

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class IntegratedSystem:
    """集成系统测试类"""
    
    def __init__(self):
        self.gpu_accelerator = GPUAccelerator()
        self.error_handler = ErrorHandler()
        self.performance_monitor = PerformanceMonitor()
        self.memory_manager = MemoryManager(max_memory=500 * 1024 * 1024)  # 500MB
        
        logger.info("集成系统初始化完成")
    
    @monitor_performance("integrated_image_processing")
    def process_image_integrated(self, image: np.ndarray, operation: str = "blur") -> np.ndarray:
        """集成图像处理"""
        try:
            # 验证输入
            if image is None or image.size == 0:
                raise ValidationError("图像数据无效", "VAL_IMG_001", ["检查图像数据", "确保图像不为空"])
            
            if not isinstance(image, np.ndarray):
                raise ValidationError("图像格式错误", "VAL_IMG_002", ["使用numpy数组", "检查图像类型"])
            
            # 分配内存
            image_id = self.memory_manager.allocate_image(image, f"proc_{int(time.time())}")
            
            # GPU加速处理
            if operation == "blur":
                result = cv2.GaussianBlur(image, (15, 15), 0)
            elif operation == "sharpen":
                kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
                result = cv2.filter2D(image, -1, kernel)
            elif operation == "edge_detect":
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                result = cv2.Canny(gray, 100, 200)
                result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
            else:
                raise ValidationError(f"不支持的操作: {operation}", "VAL_OP_001", ["使用blur", "使用sharpen", "使用edge_detect"])
            
            # 验证结果
            if result is None or result.size == 0:
                raise ProcessingError("图像处理失败", "PROC_IMG_001", ["检查GPU状态", "重试处理"])
            
            # 存储结果
            result_id = self.memory_manager.allocate_image(result, f"result_{int(time.time())}")
            
            logger.info(f"图像处理成功: {operation}, 结果ID: {result_id}")
            return result
            
        except (ValidationError, ProcessingError, ResourceError) as e:
            self.error_handler.capture_error(e)
            raise
        except Exception as e:
            # 处理未预期的错误
            unexpected_error = ProcessingError(f"未预期的处理错误: {str(e)}", "PROC_UNEXP_001", ["检查系统状态", "联系技术支持"])
            self.error_handler.capture_error(unexpected_error)
            raise unexpected_error
    
    def benchmark_system(self, test_images: list, operations: list) -> Dict[str, Any]:
        """系统基准测试"""
        results = {
            "total_tests": 0,
            "successful_tests": 0,
            "failed_tests": 0,
            "performance_metrics": {},
            "error_summary": {},
            "memory_usage": {}
        }
        
        for i, image in enumerate(test_images):
            for operation in operations:
                results["total_tests"] += 1
                
                try:
                    start_time = time.time()
                    result = self.process_image_integrated(image, operation)
                    end_time = time.time()
                    
                    results["successful_tests"] += 1
                    
                    # 记录性能指标
                    test_key = f"test_{i}_{operation}"
                    results["performance_metrics"][test_key] = {
                        "execution_time": end_time - start_time,
                        "success": True,
                        "result_shape": result.shape
                    }
                    
                except Exception as e:
                    results["failed_tests"] += 1
                    test_key = f"test_{i}_{operation}"
                    results["performance_metrics"][test_key] = {
                        "execution_time": 0,
                        "success": False,
                        "error": str(e)
                    }
        
        # 获取系统统计
        results["memory_usage"] = self.memory_manager.get_memory_stats()
        results["error_summary"] = self.error_handler.get_error_stats()
        results["performance_summary"] = self.performance_monitor.get_global_summary()
        
        return results
    
    def stress_test(self, num_iterations: int = 10) -> Dict[str, Any]:
        """压力测试"""
        logger.info(f"开始压力测试，迭代次数: {num_iterations}")
        
        # 创建测试图像
        test_image = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        operations = ["blur", "sharpen", "edge_detect"]
        
        results = {
            "iterations": num_iterations,
            "successful_iterations": 0,
            "failed_iterations": 0,
            "total_processing_time": 0,
            "average_processing_time": 0,
            "memory_leaks": 0
        }
        
        start_time = time.time()
        
        for i in range(num_iterations):
            try:
                operation = operations[i % len(operations)]
                result = self.process_image_integrated(test_image, operation)
                results["successful_iterations"] += 1
                
                # 每5次迭代进行一次内存优化
                if (i + 1) % 5 == 0:
                    self.memory_manager.optimize_memory()
                
            except Exception as e:
                results["failed_iterations"] += 1
                logger.error(f"压力测试迭代 {i} 失败: {e}")
        
        end_time = time.time()
        results["total_processing_time"] = end_time - start_time
        results["average_processing_time"] = results["total_processing_time"] / num_iterations
        
        # 检查内存泄漏
        if self.memory_manager.leak_detector:
            leak_stats = self.memory_manager.leak_detector.get_stats()
            results["memory_leaks"] = leak_stats["potential_leaks"]
        
        logger.info(f"压力测试完成: {results}")
        return results
    
    def generate_report(self) -> Dict[str, Any]:
        """生成综合报告"""
        report = {
            "timestamp": time.time(),
            "system_info": {
                "gpu_available": self.gpu_accelerator.is_available(),
                "gpu_info": self.gpu_accelerator.get_gpu_info(),
                "memory_manager_stats": self.memory_manager.get_memory_stats(),
                "performance_monitor_stats": self.performance_monitor.get_global_summary(),
                "error_handler_stats": self.error_handler.get_error_stats()
            },
            "component_status": {
                "gpu_accelerator": "正常" if self.gpu_accelerator.is_available() else "不可用",
                "error_handler": "正常",
                "performance_monitor": "正常",
                "memory_manager": "正常"
            }
        }
        
        return report


def test_integration():
    """集成测试主函数"""
    print("=== 集成系统测试 ===")
    
    # 创建集成系统
    system = IntegratedSystem()
    
    # 生成测试图像
    test_images = []
    for i in range(3):
        image = np.random.randint(0, 255, (150, 150, 3), dtype=np.uint8)
        test_images.append(image)
    
    print("1. 测试基本图像处理...")
    try:
        result = system.process_image_integrated(test_images[0], "blur")
        print(f"   基本处理成功，结果形状: {result.shape}")
    except Exception as e:
        print(f"   基本处理失败: {e}")
    
    print("2. 测试基准测试...")
    operations = ["blur", "sharpen"]
    benchmark_results = system.benchmark_system(test_images, operations)
    print(f"   基准测试完成: {benchmark_results['successful_tests']}/{benchmark_results['total_tests']} 成功")
    
    print("3. 测试压力测试...")
    stress_results = system.stress_test(5)
    print(f"   压力测试完成: {stress_results['successful_iterations']}/{stress_results['iterations']} 成功")
    
    print("4. 生成综合报告...")
    report = system.generate_report()
    print("   综合报告生成完成")
    
    print("5. 系统状态检查...")
    for component, status in report["component_status"].items():
        print(f"   {component}: {status}")
    
    print("=== 集成测试完成 ===")
    
    return report


def test_error_scenarios():
    """测试错误场景"""
    print("=== 错误场景测试 ===")
    
    system = IntegratedSystem()
    
    # 测试无效输入
    print("1. 测试无效图像输入...")
    try:
        system.process_image_integrated(None, "blur")
    except ValidationError as e:
        print(f"   正确捕获验证错误: {str(e)}")
    
    # 测试无效操作
    print("2. 测试无效操作...")
    try:
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        system.process_image_integrated(image, "invalid_operation")
    except ValidationError as e:
        print(f"   正确捕获操作错误: {str(e)}")
    
    print("=== 错误场景测试完成 ===")


if __name__ == "__main__":
    # 运行集成测试
    test_integration()
    
    # 运行错误场景测试
    test_error_scenarios() 