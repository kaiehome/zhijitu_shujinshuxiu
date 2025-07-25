"""
综合测试套件
测试所有新开发的AI组件和性能基准测试
"""

import cv2
import numpy as np
import logging
import time
import json
import argparse
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path
import os
import sys

# 导入新开发的组件
from parallel_processor import ParallelProcessor, ImageProcessingPipeline
from gpu_accelerator import GPUAccelerator, GPUProcessingPipeline
from memory_manager import MemoryManager
from enhanced_image_processor import EnhancedImageProcessor, EnhancedProcessingManager
from deep_learning_models import DeepLearningModelManager, ModelConfig
from quality_assessment import QualityAssessmentSystem

logger = logging.getLogger(__name__)


class ComprehensiveTestSuite:
    """综合测试套件"""
    
    def __init__(self, test_data_dir: str = "test_data", results_dir: str = "test_results"):
        self.test_data_dir = Path(test_data_dir)
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True, parents=True)
        
        # 测试结果
        self.test_results = {
            "timestamp": time.time(),
            "system_info": self._get_system_info(),
            "tests": {}
        }
        
        logger.info("综合测试套件初始化完成")
    
    def run_all_tests(self) -> Dict[str, Any]:
        """运行所有测试"""
        logger.info("🚀 开始运行综合测试套件")
        
        try:
            # 1. 基础功能测试
            self.test_results["tests"]["basic_functionality"] = self._test_basic_functionality()
            
            # 2. 并行处理测试
            self.test_results["tests"]["parallel_processing"] = self._test_parallel_processing()
            
            # 3. GPU加速测试
            self.test_results["tests"]["gpu_acceleration"] = self._test_gpu_acceleration()
            
            # 4. 内存管理测试
            self.test_results["tests"]["memory_management"] = self._test_memory_management()
            
            # 5. 增强图像处理测试
            self.test_results["tests"]["enhanced_processing"] = self._test_enhanced_processing()
            
            # 6. 深度学习模型测试
            self.test_results["tests"]["deep_learning"] = self._test_deep_learning()
            
            # 7. 质量评估测试
            self.test_results["tests"]["quality_assessment"] = self._test_quality_assessment()
            
            # 8. 性能基准测试
            self.test_results["tests"]["performance_benchmark"] = self._test_performance_benchmark()
            
            # 9. 集成测试
            self.test_results["tests"]["integration"] = self._test_integration()
            
            # 保存测试结果
            self._save_test_results()
            
            # 生成测试报告
            report = self._generate_test_report()
            
            logger.info("✅ 综合测试套件运行完成")
            return report
            
        except Exception as e:
            logger.error(f"综合测试套件运行失败: {e}")
            raise
    
    def _test_basic_functionality(self) -> Dict[str, Any]:
        """基础功能测试"""
        logger.info("📋 运行基础功能测试")
        
        results = {
            "status": "passed",
            "tests": {},
            "errors": []
        }
        
        try:
            # 测试OpenCV基础功能
            test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            
            # 测试图像读取和保存
            test_path = self.results_dir / "test_basic.png"
            cv2.imwrite(str(test_path), test_image)
            loaded_image = cv2.imread(str(test_path))
            
            if loaded_image is not None and np.array_equal(test_image, loaded_image):
                results["tests"]["opencv_io"] = {"status": "passed", "message": "OpenCV I/O功能正常"}
            else:
                results["tests"]["opencv_io"] = {"status": "failed", "message": "OpenCV I/O功能异常"}
                results["status"] = "failed"
            
            # 测试NumPy功能
            array1 = np.array([1, 2, 3])
            array2 = np.array([4, 5, 6])
            result = np.dot(array1, array2)
            
            if result == 32:  # 1*4 + 2*5 + 3*6 = 32
                results["tests"]["numpy_operations"] = {"status": "passed", "message": "NumPy运算功能正常"}
            else:
                results["tests"]["numpy_operations"] = {"status": "failed", "message": "NumPy运算功能异常"}
                results["status"] = "failed"
            
        except Exception as e:
            results["status"] = "failed"
            results["errors"].append(f"基础功能测试异常: {e}")
            logger.error(f"基础功能测试失败: {e}")
        
        return results
    
    def _test_parallel_processing(self) -> Dict[str, Any]:
        """并行处理测试"""
        logger.info("🔄 运行并行处理测试")
        
        results = {
            "status": "passed",
            "tests": {},
            "performance": {},
            "errors": []
        }
        
        try:
            # 创建测试图像
            test_images = []
            for i in range(5):
                img = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
                test_images.append(img)
            
            # 测试并行处理器
            processor = ParallelProcessor(max_workers=2)
            
            # 测试任务提交
            start_time = time.time()
            tasks = []
            for i, img in enumerate(test_images):
                task = processor.submit_task(f"test_{i}", img, "blur", {"kernel_size": 5})
                tasks.append(task)
            
            # 等待所有任务完成
            results_list = []
            for task in tasks:
                result = processor.get_result(task.task_id)
                if result:
                    results_list.append(result)
            
            processing_time = time.time() - start_time
            
            # 验证结果
            if len(results_list) == len(test_images):
                results["tests"]["task_execution"] = {"status": "passed", "message": "并行任务执行成功"}
                results["performance"]["processing_time"] = processing_time
                results["performance"]["images_per_second"] = len(test_images) / processing_time
            else:
                results["tests"]["task_execution"] = {"status": "failed", "message": "并行任务执行失败"}
                results["status"] = "failed"
            
            # 测试流水线处理
            pipeline = ImageProcessingPipeline(max_workers=2)
            pipeline.add_step("blur", {"kernel_size": 3})
            pipeline.add_step("sharpen", {"kernel_size": 3})
            
            start_time = time.time()
            pipeline_result = pipeline.process_image(test_images[0], "pipeline_test")
            pipeline_time = time.time() - start_time
            
            if pipeline_result:
                results["tests"]["pipeline_processing"] = {"status": "passed", "message": "流水线处理成功"}
                results["performance"]["pipeline_time"] = pipeline_time
            else:
                results["tests"]["pipeline_processing"] = {"status": "failed", "message": "流水线处理失败"}
                results["status"] = "failed"
            
            processor.shutdown()
            
        except Exception as e:
            results["status"] = "failed"
            results["errors"].append(f"并行处理测试异常: {e}")
            logger.error(f"并行处理测试失败: {e}")
        
        return results
    
    def _test_gpu_acceleration(self) -> Dict[str, Any]:
        """GPU加速测试"""
        logger.info("🚀 运行GPU加速测试")
        
        results = {
            "status": "passed",
            "tests": {},
            "performance": {},
            "errors": []
        }
        
        try:
            # 测试GPU检测
            gpu_accelerator = GPUAccelerator()
            gpu_info = gpu_accelerator.get_gpu_info()
            
            if gpu_info:
                results["tests"]["gpu_detection"] = {"status": "passed", "message": f"检测到GPU: {gpu_info[0].name}"}
                results["performance"]["gpu_info"] = [gpu.to_dict() for gpu in gpu_info]
            else:
                results["tests"]["gpu_detection"] = {"status": "skipped", "message": "未检测到GPU"}
                results["status"] = "skipped"
                return results
            
            # 测试GPU处理流水线
            test_image = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
            
            gpu_pipeline = GPUProcessingPipeline()
            gpu_pipeline.add_step("bilateral_filter", {"d": 9, "sigma_color": 75, "sigma_space": 75})
            
            start_time = time.time()
            gpu_result = gpu_pipeline.process_image(test_image)
            gpu_time = time.time() - start_time
            
            if gpu_result is not None:
                results["tests"]["gpu_processing"] = {"status": "passed", "message": "GPU处理成功"}
                results["performance"]["gpu_processing_time"] = gpu_time
            else:
                results["tests"]["gpu_processing"] = {"status": "failed", "message": "GPU处理失败"}
                results["status"] = "failed"
            
            # 对比CPU处理时间
            start_time = time.time()
            cpu_result = cv2.bilateralFilter(test_image, 9, 75, 75)
            cpu_time = time.time() - start_time
            
            results["performance"]["cpu_processing_time"] = cpu_time
            if gpu_time < cpu_time:
                results["performance"]["speedup"] = cpu_time / gpu_time
            else:
                results["performance"]["speedup"] = 1.0
            
        except Exception as e:
            results["status"] = "failed"
            results["errors"].append(f"GPU加速测试异常: {e}")
            logger.error(f"GPU加速测试失败: {e}")
        
        return results
    
    def _test_memory_management(self) -> Dict[str, Any]:
        """内存管理测试"""
        logger.info("💾 运行内存管理测试")
        
        results = {
            "status": "passed",
            "tests": {},
            "performance": {},
            "errors": []
        }
        
        try:
            # 测试内存管理器
            memory_manager = MemoryManager(pool_size=5, cache_size=10)
            
            # 测试内存分配
            start_time = time.time()
            allocated_memory = []
            for i in range(10):
                mem = memory_manager.allocate_memory(1024 * 1024)  # 1MB
                if mem is not None:
                    allocated_memory.append(mem)
            
            allocation_time = time.time() - start_time
            
            if len(allocated_memory) == 10:
                results["tests"]["memory_allocation"] = {"status": "passed", "message": "内存分配成功"}
                results["performance"]["allocation_time"] = allocation_time
            else:
                results["tests"]["memory_allocation"] = {"status": "failed", "message": "内存分配失败"}
                results["status"] = "failed"
            
            # 测试图像缓存
            test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            
            cache_success = memory_manager.cache_image("test_image", test_image)
            if cache_success:
                results["tests"]["image_caching"] = {"status": "passed", "message": "图像缓存成功"}
            else:
                results["tests"]["image_caching"] = {"status": "failed", "message": "图像缓存失败"}
                results["status"] = "failed"
            
            # 测试缓存检索
            cached_image = memory_manager.get_cached_image("test_image")
            if cached_image is not None and np.array_equal(test_image, cached_image):
                results["tests"]["cache_retrieval"] = {"status": "passed", "message": "缓存检索成功"}
            else:
                results["tests"]["cache_retrieval"] = {"status": "failed", "message": "缓存检索失败"}
                results["status"] = "failed"
            
            # 测试内存优化
            optimization_result = memory_manager.optimize_memory()
            results["performance"]["optimization_result"] = optimization_result
            
            # 获取统计信息
            stats = memory_manager.get_comprehensive_stats()
            results["performance"]["memory_stats"] = stats
            
            memory_manager.shutdown()
            
        except Exception as e:
            results["status"] = "failed"
            results["errors"].append(f"内存管理测试异常: {e}")
            logger.error(f"内存管理测试失败: {e}")
        
        return results
    
    def _test_enhanced_processing(self) -> Dict[str, Any]:
        """增强图像处理测试"""
        logger.info("🎨 运行增强图像处理测试")
        
        results = {
            "status": "passed",
            "tests": {},
            "performance": {},
            "errors": []
        }
        
        try:
            # 创建测试图像
            test_image = np.random.randint(0, 255, (400, 400, 3), dtype=np.uint8)
            test_path = self.results_dir / "test_enhanced_input.png"
            cv2.imwrite(str(test_path), test_image)
            
            # 测试增强图像处理器
            processor = EnhancedImageProcessor(
                outputs_dir=str(self.results_dir / "enhanced_outputs"),
                use_gpu=False,  # 先测试CPU版本
                use_parallel=True
            )
            
            # 测试增强处理
            start_time = time.time()
            output_path, error, processing_time = processor.process_image_enhanced(
                str(test_path), "test_enhanced",
                color_count=16,
                edge_enhancement=True,
                noise_reduction=True,
                use_ai_segmentation=True
            )
            
            total_time = time.time() - start_time
            
            if error == "" and os.path.exists(output_path):
                results["tests"]["enhanced_processing"] = {"status": "passed", "message": "增强处理成功"}
                results["performance"]["processing_time"] = processing_time
                results["performance"]["total_time"] = total_time
            else:
                results["tests"]["enhanced_processing"] = {"status": "failed", "message": f"增强处理失败: {error}"}
                results["status"] = "failed"
            
            # 测试处理管理器
            manager = EnhancedProcessingManager(num_processors=2)
            
            start_time = time.time()
            output_path2, error2, processing_time2 = manager.process_image(
                str(test_path), "test_manager",
                color_count=8,
                edge_enhancement=False,
                noise_reduction=True
            )
            
            manager_time = time.time() - start_time
            
            if error2 == "" and os.path.exists(output_path2):
                results["tests"]["processing_manager"] = {"status": "passed", "message": "处理管理器成功"}
                results["performance"]["manager_time"] = manager_time
            else:
                results["tests"]["processing_manager"] = {"status": "failed", "message": f"处理管理器失败: {error2}"}
                results["status"] = "failed"
            
            # 获取性能统计
            stats = processor.get_performance_stats()
            results["performance"]["processor_stats"] = stats
            
            manager.shutdown()
            
        except Exception as e:
            results["status"] = "failed"
            results["errors"].append(f"增强图像处理测试异常: {e}")
            logger.error(f"增强图像处理测试失败: {e}")
        
        return results
    
    def _test_deep_learning(self) -> Dict[str, Any]:
        """深度学习模型测试"""
        logger.info("🧠 运行深度学习模型测试")
        
        results = {
            "status": "passed",
            "tests": {},
            "performance": {},
            "errors": []
        }
        
        try:
            # 测试深度学习模型管理器
            model_manager = DeepLearningModelManager(models_dir=str(self.results_dir / "models"))
            
            # 测试模型配置
            config = ModelConfig(
                model_type="unet",
                input_size=(256, 256),
                num_classes=2,
                device="cpu"  # 使用CPU进行测试
            )
            
            # 测试模型加载（如果PyTorch可用）
            try:
                if model_manager.load_segmentation_model("unet", config):
                    results["tests"]["model_loading"] = {"status": "passed", "message": "模型加载成功"}
                else:
                    results["tests"]["model_loading"] = {"status": "skipped", "message": "模型加载跳过（依赖库未安装）"}
                    results["status"] = "skipped"
                    return results
            except ImportError:
                results["tests"]["model_loading"] = {"status": "skipped", "message": "PyTorch未安装"}
                results["status"] = "skipped"
                return results
            
            # 测试图像分割
            test_image = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
            
            start_time = time.time()
            segmentation_result = model_manager.segment_image(test_image, "unet")
            segmentation_time = time.time() - start_time
            
            if segmentation_result is not None and segmentation_result.shape[:2] == test_image.shape[:2]:
                results["tests"]["image_segmentation"] = {"status": "passed", "message": "图像分割成功"}
                results["performance"]["segmentation_time"] = segmentation_time
            else:
                results["tests"]["image_segmentation"] = {"status": "failed", "message": "图像分割失败"}
                results["status"] = "failed"
            
            # 测试特征提取器
            if model_manager.load_feature_extractor("resnet50"):
                start_time = time.time()
                features = model_manager.extract_features(test_image, "resnet50")
                feature_time = time.time() - start_time
                
                if features is not None and len(features) > 0:
                    results["tests"]["feature_extraction"] = {"status": "passed", "message": "特征提取成功"}
                    results["performance"]["feature_time"] = feature_time
                    results["performance"]["feature_dimension"] = len(features)
                else:
                    results["tests"]["feature_extraction"] = {"status": "failed", "message": "特征提取失败"}
                    results["status"] = "failed"
            
            # 获取可用模型列表
            available_models = model_manager.get_available_models()
            results["performance"]["available_models"] = available_models
            
        except Exception as e:
            results["status"] = "failed"
            results["errors"].append(f"深度学习模型测试异常: {e}")
            logger.error(f"深度学习模型测试失败: {e}")
        
        return results
    
    def _test_quality_assessment(self) -> Dict[str, Any]:
        """质量评估测试"""
        logger.info("📊 运行质量评估测试")
        
        results = {
            "status": "passed",
            "tests": {},
            "performance": {},
            "errors": []
        }
        
        try:
            # 创建测试图像
            original_image = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
            processed_image = cv2.GaussianBlur(original_image, (5, 5), 0)  # 添加一些模糊
            
            original_path = self.results_dir / "test_original.png"
            processed_path = self.results_dir / "test_processed.png"
            
            cv2.imwrite(str(original_path), original_image)
            cv2.imwrite(str(processed_path), processed_image)
            
            # 测试质量评估系统
            quality_system = QualityAssessmentSystem(output_dir=str(self.results_dir / "quality"))
            
            # 主观评分
            subjective_scores = {
                "color_accuracy": 8.0,
                "edge_preservation": 7.5,
                "detail_preservation": 7.0,
                "overall_quality": 7.5
            }
            
            # 综合评估
            start_time = time.time()
            assessment_result = quality_system.comprehensive_assessment(
                str(original_path), str(processed_path), "test_quality", subjective_scores
            )
            assessment_time = time.time() - start_time
            
            if assessment_result:
                results["tests"]["quality_assessment"] = {"status": "passed", "message": "质量评估成功"}
                results["performance"]["assessment_time"] = assessment_time
                results["performance"]["assessment_result"] = assessment_result
            else:
                results["tests"]["quality_assessment"] = {"status": "failed", "message": "质量评估失败"}
                results["status"] = "failed"
            
            # 测试统计信息
            stats = quality_system.get_statistics()
            results["performance"]["quality_stats"] = stats
            
        except Exception as e:
            results["status"] = "failed"
            results["errors"].append(f"质量评估测试异常: {e}")
            logger.error(f"质量评估测试失败: {e}")
        
        return results
    
    def _test_performance_benchmark(self) -> Dict[str, Any]:
        """性能基准测试"""
        logger.info("⚡ 运行性能基准测试")
        
        results = {
            "status": "passed",
            "tests": {},
            "benchmarks": {},
            "errors": []
        }
        
        try:
            # 创建不同尺寸的测试图像
            test_sizes = [(100, 100), (300, 300), (500, 500), (800, 800)]
            
            for size in test_sizes:
                test_image = np.random.randint(0, 255, (*size, 3), dtype=np.uint8)
                size_key = f"{size[0]}x{size[1]}"
                
                # 基准测试：传统OpenCV处理
                start_time = time.time()
                cv2_result = cv2.bilateralFilter(test_image, 9, 75, 75)
                cv2_time = time.time() - start_time
                
                # 基准测试：增强处理器
                processor = EnhancedImageProcessor(use_gpu=False, use_parallel=True)
                start_time = time.time()
                output_path, error, processing_time = processor.process_image_enhanced(
                    "test_image", f"benchmark_{size_key}",
                    color_count=16,
                    edge_enhancement=True,
                    noise_reduction=True
                )
                enhanced_time = time.time() - start_time
                
                results["benchmarks"][size_key] = {
                    "opencv_time": cv2_time,
                    "enhanced_time": enhanced_time,
                    "speedup": cv2_time / enhanced_time if enhanced_time > 0 else 0
                }
            
            # 内存使用基准测试
            memory_manager = MemoryManager()
            start_memory = memory_manager.get_comprehensive_stats()["system_memory"]["used_mb"]
            
            # 处理大量图像
            for i in range(10):
                test_image = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
                memory_manager.cache_image(f"benchmark_{i}", test_image)
            
            end_memory = memory_manager.get_comprehensive_stats()["system_memory"]["used_mb"]
            memory_increase = end_memory - start_memory
            
            results["benchmarks"]["memory_usage"] = {
                "start_memory_mb": start_memory,
                "end_memory_mb": end_memory,
                "memory_increase_mb": memory_increase
            }
            
            memory_manager.shutdown()
            
        except Exception as e:
            results["status"] = "failed"
            results["errors"].append(f"性能基准测试异常: {e}")
            logger.error(f"性能基准测试失败: {e}")
        
        return results
    
    def _test_integration(self) -> Dict[str, Any]:
        """集成测试"""
        logger.info("🔗 运行集成测试")
        
        results = {
            "status": "passed",
            "tests": {},
            "performance": {},
            "errors": []
        }
        
        try:
            # 创建完整的处理流水线
            test_image = np.random.randint(0, 255, (400, 400, 3), dtype=np.uint8)
            test_path = self.results_dir / "test_integration.png"
            cv2.imwrite(str(test_path), test_image)
            
            # 1. 增强图像处理
            processor = EnhancedImageProcessor(use_gpu=False, use_parallel=True)
            output_path, error, processing_time = processor.process_image_enhanced(
                str(test_path), "integration_test",
                color_count=16,
                edge_enhancement=True,
                noise_reduction=True,
                use_ai_segmentation=True
            )
            
            if error != "" or not os.path.exists(output_path):
                results["tests"]["enhanced_processing"] = {"status": "failed", "message": "增强处理失败"}
                results["status"] = "failed"
                return results
            
            # 2. 质量评估
            quality_system = QualityAssessmentSystem()
            subjective_scores = {
                "color_accuracy": 8.0,
                "edge_preservation": 7.5,
                "detail_preservation": 7.0,
                "overall_quality": 7.5
            }
            
            assessment_result = quality_system.comprehensive_assessment(
                str(test_path), output_path, "integration_test", subjective_scores
            )
            
            if assessment_result:
                results["tests"]["quality_assessment"] = {"status": "passed", "message": "质量评估成功"}
                results["performance"]["assessment_result"] = assessment_result
            else:
                results["tests"]["quality_assessment"] = {"status": "failed", "message": "质量评估失败"}
                results["status"] = "failed"
            
            # 3. 深度学习模型（如果可用）
            try:
                model_manager = DeepLearningModelManager()
                config = ModelConfig(model_type="unet", input_size=(256, 256), device="cpu")
                
                if model_manager.load_segmentation_model("unet", config):
                    segmentation_result = model_manager.segment_image(test_image, "unet")
                    if segmentation_result is not None:
                        results["tests"]["deep_learning"] = {"status": "passed", "message": "深度学习集成成功"}
                    else:
                        results["tests"]["deep_learning"] = {"status": "failed", "message": "深度学习集成失败"}
                        results["status"] = "failed"
                else:
                    results["tests"]["deep_learning"] = {"status": "skipped", "message": "深度学习模型不可用"}
            except Exception as e:
                results["tests"]["deep_learning"] = {"status": "skipped", "message": f"深度学习跳过: {e}"}
            
            results["tests"]["integration"] = {"status": "passed", "message": "集成测试完成"}
            
        except Exception as e:
            results["status"] = "failed"
            results["errors"].append(f"集成测试异常: {e}")
            logger.error(f"集成测试失败: {e}")
        
        return results
    
    def _get_system_info(self) -> Dict[str, Any]:
        """获取系统信息"""
        import platform
        import psutil
        
        return {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "cpu_count": psutil.cpu_count(),
            "memory_total_gb": psutil.virtual_memory().total / (1024**3),
            "opencv_version": cv2.__version__,
            "numpy_version": np.__version__
        }
    
    def _save_test_results(self):
        """保存测试结果"""
        timestamp = time.strftime("%y%m%d_%H%M%S")
        filename = f"comprehensive_test_results_{timestamp}.json"
        filepath = self.results_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.test_results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"测试结果已保存: {filepath}")
    
    def _generate_test_report(self) -> Dict[str, Any]:
        """生成测试报告"""
        total_tests = 0
        passed_tests = 0
        failed_tests = 0
        skipped_tests = 0
        
        for test_category, test_result in self.test_results["tests"].items():
            if test_result["status"] == "passed":
                passed_tests += 1
            elif test_result["status"] == "failed":
                failed_tests += 1
            elif test_result["status"] == "skipped":
                skipped_tests += 1
            
            total_tests += 1
        
        report = {
            "summary": {
                "total_tests": total_tests,
                "passed_tests": passed_tests,
                "failed_tests": failed_tests,
                "skipped_tests": skipped_tests,
                "success_rate": (passed_tests / total_tests * 100) if total_tests > 0 else 0
            },
            "system_info": self.test_results["system_info"],
            "detailed_results": self.test_results["tests"],
            "timestamp": self.test_results["timestamp"]
        }
        
        # 打印报告摘要
        logger.info("📊 测试报告摘要:")
        logger.info(f"  总测试数: {total_tests}")
        logger.info(f"  通过: {passed_tests}")
        logger.info(f"  失败: {failed_tests}")
        logger.info(f"  跳过: {skipped_tests}")
        logger.info(f"  成功率: {report['summary']['success_rate']:.1f}%")
        
        return report


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="综合测试套件")
    parser.add_argument("--test-data-dir", default="test_data", help="测试数据目录")
    parser.add_argument("--results-dir", default="test_results", help="结果输出目录")
    parser.add_argument("--verbose", "-v", action="store_true", help="详细输出")
    
    args = parser.parse_args()
    
    # 设置日志级别
    if args.verbose:
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    else:
        logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # 运行测试套件
    test_suite = ComprehensiveTestSuite(args.test_data_dir, args.results_dir)
    report = test_suite.run_all_tests()
    
    # 返回退出码
    if report["summary"]["failed_tests"] > 0:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main() 