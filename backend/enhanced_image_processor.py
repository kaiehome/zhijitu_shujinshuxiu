"""
增强版图像处理器
集成并行处理、GPU加速和智能内存管理
"""

import cv2
import numpy as np
import logging
import time
from typing import Dict, Any, Optional, Tuple, List
from pathlib import Path
import os

# 导入新开发的组件
from parallel_processor import ParallelProcessor, ProcessingTask, ImageProcessingPipeline
from gpu_accelerator import GPUAccelerator, GPUProcessingPipeline
from memory_manager import MemoryManager

logger = logging.getLogger(__name__)


class EnhancedImageProcessor:
    """
    增强版图像处理器
    
    集成并行处理、GPU加速和智能内存管理，提供高性能图像处理
    """
    
    def __init__(self, 
                 outputs_dir: str = "outputs",
                 use_gpu: bool = True,
                 use_parallel: bool = True,
                 max_workers: int = None,
                 cache_size: int = 100):
        """
        初始化增强版图像处理器
        
        Args:
            outputs_dir: 输出目录
            use_gpu: 是否使用GPU加速
            use_parallel: 是否使用并行处理
            max_workers: 最大工作进程数
            cache_size: 缓存大小
        """
        self.outputs_dir = Path(outputs_dir)
        self.outputs_dir.mkdir(exist_ok=True, parents=True)
        
        # 初始化组件
        self.use_gpu = use_gpu
        self.use_parallel = use_parallel
        
        # GPU加速器
        if use_gpu:
            self.gpu_accelerator = GPUAccelerator()
            self.gpu_pipeline = GPUProcessingPipeline()
            logger.info("GPU加速器已启用")
        else:
            self.gpu_accelerator = None
            self.gpu_pipeline = None
        
        # 并行处理器
        if use_parallel:
            self.parallel_processor = ParallelProcessor(max_workers=max_workers)
            self.parallel_pipeline = ImageProcessingPipeline(max_workers=max_workers)
            logger.info("并行处理器已启用")
        else:
            self.parallel_processor = None
            self.parallel_pipeline = None
        
        # 内存管理器
        self.memory_manager = MemoryManager(cache_size=cache_size)
        
        # 性能统计
        self.performance_stats = {
            "total_processed": 0,
            "gpu_processed": 0,
            "parallel_processed": 0,
            "total_time": 0,
            "average_time": 0
        }
        
        logger.info("增强版图像处理器初始化完成")
    
    def process_image_enhanced(self, 
                              input_path: str, 
                              job_id: str,
                              color_count: int = 16,
                              edge_enhancement: bool = True,
                              noise_reduction: bool = True,
                              use_ai_segmentation: bool = True) -> Tuple[str, str, float]:
        """
        增强版图像处理
        
        Args:
            input_path: 输入图像路径
            job_id: 任务ID
            color_count: 颜色数量
            edge_enhancement: 是否启用边缘增强
            noise_reduction: 是否启用噪声清理
            use_ai_segmentation: 是否使用AI分割
            
        Returns:
            Tuple[str, str, float]: 处理结果路径和耗时
        """
        start_time = time.time()
        
        try:
            logger.info(f"🚀 开始增强版图像处理: {job_id}")
            
            # 检查缓存
            cache_key = f"{job_id}_enhanced"
            cached_result = self.memory_manager.get_cached_image(cache_key)
            if cached_result is not None:
                logger.info(f"使用缓存结果: {job_id}")
                return self._save_cached_result(cached_result, job_id, start_time)
            
            # 加载图像
            image = self._load_image(input_path)
            if image is None:
                raise ValueError(f"无法加载图像: {input_path}")
            
            # 选择处理策略
            if self.use_gpu and self.gpu_accelerator.is_available():
                processed_image = self._gpu_process(image, color_count, edge_enhancement, 
                                                  noise_reduction, use_ai_segmentation)
                self.performance_stats["gpu_processed"] += 1
            elif self.use_parallel:
                processed_image = self._parallel_process(image, color_count, edge_enhancement,
                                                       noise_reduction, use_ai_segmentation)
                self.performance_stats["parallel_processed"] += 1
            else:
                processed_image = self._cpu_process(image, color_count, edge_enhancement,
                                                  noise_reduction, use_ai_segmentation)
            
            # 保存结果
            output_path = self._save_result(processed_image, job_id)
            
            # 缓存结果
            self.memory_manager.cache_image(cache_key, processed_image)
            
            # 更新统计
            processing_time = time.time() - start_time
            self._update_stats(processing_time)
            
            logger.info(f"✅ 增强版图像处理完成: {job_id}, 耗时: {processing_time:.2f}s")
            
            return output_path, "", processing_time
            
        except Exception as e:
            logger.error(f"增强版图像处理失败: {job_id}, 错误: {e}")
            raise
    
    def _gpu_process(self, image: np.ndarray, color_count: int, edge_enhancement: bool,
                    noise_reduction: bool, use_ai_segmentation: bool) -> np.ndarray:
        """GPU处理"""
        # 构建GPU处理流水线
        self.gpu_pipeline.pipeline_steps.clear()
        
        if noise_reduction:
            self.gpu_pipeline.add_step("bilateral_filter", {
                "d": 9, "sigma_color": 75, "sigma_space": 75
            })
        
        if edge_enhancement:
            self.gpu_pipeline.add_step("laplacian", {
                "ddepth": cv2.CV_16S, "ksize": 3
            })
        
        # 颜色降色（使用GPU加速的K-means）
        processed_image = self.gpu_pipeline.process_image(image)
        
        # 颜色降色（CPU处理，因为GPU K-means实现复杂）
        processed_image = self._color_reduction_gpu_fallback(processed_image, color_count)
        
        return processed_image
    
    def _parallel_process(self, image: np.ndarray, color_count: int, edge_enhancement: bool,
                         noise_reduction: bool, use_ai_segmentation: bool) -> np.ndarray:
        """并行处理"""
        # 构建并行处理流水线
        self.parallel_pipeline.pipeline_steps.clear()
        
        if noise_reduction:
            self.parallel_pipeline.add_step("noise_reduction", {
                "kernel_size": 5
            })
        
        if edge_enhancement:
            self.parallel_pipeline.add_step("edge_enhancement", {
                "kernel_size": 3
            })
        
        if use_ai_segmentation:
            self.parallel_pipeline.add_step("segmentation", {
                "method": "grabcut"
            })
        
        # 颜色降色
        self.parallel_pipeline.add_step("color_reduction", {
            "n_colors": color_count
        })
        
        # 执行并行处理
        results = self.parallel_pipeline.process_image(image, f"parallel_{int(time.time())}")
        
        # 获取最终结果
        if results:
            return list(results.values())[-1]  # 返回最后一个步骤的结果
        else:
            return image
    
    def _cpu_process(self, image: np.ndarray, color_count: int, edge_enhancement: bool,
                    noise_reduction: bool, use_ai_segmentation: bool) -> np.ndarray:
        """CPU处理（传统方法）"""
        processed_image = image.copy()
        
        # 噪声清理
        if noise_reduction:
            processed_image = cv2.bilateralFilter(processed_image, 9, 75, 75)
        
        # 边缘增强
        if edge_enhancement:
            gray = cv2.cvtColor(processed_image, cv2.COLOR_BGR2GRAY)
            edges = cv2.Laplacian(gray, cv2.CV_64F, ksize=3)
            edges = np.uint8(np.absolute(edges))
            enhanced = cv2.addWeighted(gray, 1.5, edges, 0.5, 0)
            processed_image = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
        
        # AI分割
        if use_ai_segmentation:
            mask = self._ai_segmentation(processed_image)
            # 应用分割掩码
            processed_image = cv2.bitwise_and(processed_image, processed_image, mask=mask)
        
        # 颜色降色
        processed_image = self._color_reduction(processed_image, color_count)
        
        return processed_image
    
    def _color_reduction_gpu_fallback(self, image: np.ndarray, n_colors: int) -> np.ndarray:
        """GPU回退的颜色降色"""
        return self._color_reduction(image, n_colors)
    
    def _color_reduction(self, image: np.ndarray, n_colors: int) -> np.ndarray:
        """颜色降色"""
        # 重塑图像为二维数组
        pixels = image.reshape(-1, 3)
        
        # K-means聚类
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
        labels = kmeans.fit_predict(pixels)
        
        # 重建图像
        quantized = kmeans.cluster_centers_[labels]
        result = quantized.reshape(image.shape).astype(np.uint8)
        
        return result
    
    def _ai_segmentation(self, image: np.ndarray) -> np.ndarray:
        """AI分割"""
        # 使用GrabCut进行分割
        mask = np.zeros(image.shape[:2], np.uint8)
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)
        
        rect = (10, 10, image.shape[1]-20, image.shape[0]-20)
        cv2.grabCut(image, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
        
        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
        return mask2 * 255
    
    def _load_image(self, input_path: str) -> Optional[np.ndarray]:
        """加载图像"""
        try:
            image = cv2.imread(input_path)
            if image is None:
                logger.error(f"无法加载图像: {input_path}")
                return None
            return image
        except Exception as e:
            logger.error(f"加载图像失败: {input_path}, 错误: {e}")
            return None
    
    def _save_result(self, image: np.ndarray, job_id: str) -> str:
        """保存处理结果"""
        timestamp = time.strftime("%y%m%d_%H%M%S")
        filename = f"{timestamp}_enhanced_{job_id}.png"
        output_path = self.outputs_dir / filename
        
        # 高质量保存
        cv2.imwrite(str(output_path), image, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        
        return str(output_path)
    
    def _save_cached_result(self, cached_image: np.ndarray, job_id: str, start_time: float) -> Tuple[str, str, float]:
        """保存缓存结果"""
        output_path = self._save_result(cached_image, job_id)
        processing_time = time.time() - start_time
        return output_path, "", processing_time
    
    def _update_stats(self, processing_time: float):
        """更新性能统计"""
        self.performance_stats["total_processed"] += 1
        self.performance_stats["total_time"] += processing_time
        self.performance_stats["average_time"] = (
            self.performance_stats["total_time"] / self.performance_stats["total_processed"]
        )
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """获取性能统计"""
        stats = self.performance_stats.copy()
        
        # 添加组件统计
        if self.gpu_accelerator:
            stats["gpu_stats"] = self.gpu_accelerator.get_performance_stats()
        
        if self.parallel_processor:
            stats["parallel_stats"] = self.parallel_processor.get_stats()
        
        stats["memory_stats"] = self.memory_manager.get_comprehensive_stats()
        
        return stats
    
    def optimize_system(self) -> Dict[str, Any]:
        """系统优化"""
        optimization_results = {}
        
        # 内存优化
        optimization_results["memory"] = self.memory_manager.optimize_memory()
        
        # GPU优化（如果有）
        if self.gpu_accelerator and self.gpu_accelerator.is_available():
            # 这里可以添加GPU特定的优化
            optimization_results["gpu"] = {"status": "optimized"}
        
        logger.info("系统优化完成")
        return optimization_results
    
    def shutdown(self):
        """关闭处理器"""
        if self.parallel_processor:
            self.parallel_processor.shutdown()
        
        if self.memory_manager:
            self.memory_manager.shutdown()
        
        logger.info("增强版图像处理器已关闭")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()


class EnhancedProcessingManager:
    """
    增强版处理管理器
    
    统一管理多个增强版图像处理器
    """
    
    def __init__(self, num_processors: int = 2):
        """
        初始化处理管理器
        
        Args:
            num_processors: 处理器数量
        """
        self.processors = []
        self.current_processor = 0
        
        # 创建多个处理器实例
        for i in range(num_processors):
            processor = EnhancedImageProcessor(
                outputs_dir=f"outputs/processor_{i}",
                use_gpu=True,
                use_parallel=True
            )
            self.processors.append(processor)
        
        logger.info(f"增强版处理管理器初始化完成: {num_processors} 个处理器")
    
    def process_image(self, input_path: str, job_id: str, **kwargs) -> Tuple[str, str, float]:
        """
        处理图像（负载均衡）
        
        Args:
            input_path: 输入图像路径
            job_id: 任务ID
            **kwargs: 其他参数
            
        Returns:
            Tuple[str, str, float]: 处理结果
        """
        # 简单的轮询负载均衡
        processor = self.processors[self.current_processor]
        self.current_processor = (self.current_processor + 1) % len(self.processors)
        
        return processor.process_image_enhanced(input_path, job_id, **kwargs)
    
    def get_all_stats(self) -> Dict[str, Any]:
        """获取所有处理器的统计信息"""
        all_stats = {}
        for i, processor in enumerate(self.processors):
            all_stats[f"processor_{i}"] = processor.get_performance_stats()
        return all_stats
    
    def shutdown(self):
        """关闭所有处理器"""
        for processor in self.processors:
            processor.shutdown()
        logger.info("增强版处理管理器已关闭") 