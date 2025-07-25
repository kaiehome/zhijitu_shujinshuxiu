"""
并行图像处理系统
提供多进程/多线程的图像处理流水线，显著提升处理性能
"""

import multiprocessing as mp
import threading
import queue
import time
import logging
from typing import Dict, List, Any, Callable, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import numpy as np
import cv2
from pathlib import Path
import os

logger = logging.getLogger(__name__)


class ProcessingTask:
    """处理任务数据类"""
    
    def __init__(self, task_id: str, input_data: Any, task_type: str, 
                 parameters: Dict[str, Any] = None):
        self.task_id = task_id
        self.input_data = input_data
        self.task_type = task_type
        self.parameters = parameters or {}
        self.created_time = time.time()
        self.status = "pending"
        self.result = None
        self.error = None


class ProcessingResult:
    """处理结果数据类"""
    
    def __init__(self, task_id: str, result: Any = None, error: str = None):
        self.task_id = task_id
        self.result = result
        self.error = error
        self.completion_time = time.time()


class ParallelProcessor:
    """
    并行图像处理器
    
    支持多进程和多线程的图像处理，提供高性能的并行计算能力
    """
    
    def __init__(self, 
                 max_workers: int = None,
                 use_multiprocessing: bool = True,
                 queue_size: int = 100):
        """
        初始化并行处理器
        
        Args:
            max_workers: 最大工作进程/线程数
            use_multiprocessing: 是否使用多进程（True）或多线程（False）
            queue_size: 任务队列大小
        """
        self.max_workers = max_workers or min(32, (os.cpu_count() or 1) + 4)
        self.use_multiprocessing = use_multiprocessing
        self.queue_size = queue_size
        
        # 任务队列
        self.task_queue = queue.Queue(maxsize=queue_size)
        self.result_queue = queue.Queue()
        
        # 执行器
        self.executor = None
        self._setup_executor()
        
        # 任务跟踪
        self.active_tasks = {}
        self.completed_tasks = {}
        
        # 性能统计
        self.stats = {
            "total_tasks": 0,
            "completed_tasks": 0,
            "failed_tasks": 0,
            "total_processing_time": 0,
            "average_processing_time": 0
        }
        
        logger.info(f"并行处理器初始化完成: {self.max_workers} workers, "
                   f"{'多进程' if use_multiprocessing else '多线程'}")
    
    def _setup_executor(self):
        """设置执行器"""
        if self.use_multiprocessing:
            self.executor = ProcessPoolExecutor(max_workers=self.max_workers)
        else:
            self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
    
    def submit_task(self, task: ProcessingTask) -> bool:
        """
        提交处理任务
        
        Args:
            task: 处理任务
            
        Returns:
            bool: 提交是否成功
        """
        try:
            self.task_queue.put(task, timeout=1.0)
            self.active_tasks[task.task_id] = task
            self.stats["total_tasks"] += 1
            logger.debug(f"任务提交成功: {task.task_id}")
            return True
        except queue.Full:
            logger.warning(f"任务队列已满，任务提交失败: {task.task_id}")
            return False
    
    def process_batch(self, tasks: List[ProcessingTask]) -> Dict[str, ProcessingResult]:
        """
        批量处理任务
        
        Args:
            tasks: 任务列表
            
        Returns:
            Dict[str, ProcessingResult]: 处理结果字典
        """
        logger.info(f"开始批量处理 {len(tasks)} 个任务")
        
        # 提交所有任务
        futures = {}
        for task in tasks:
            future = self.executor.submit(self._process_single_task, task)
            futures[future] = task.task_id
        
        # 收集结果
        results = {}
        start_time = time.time()
        
        for future in as_completed(futures):
            task_id = futures[future]
            try:
                result = future.result()
                results[task_id] = result
                self.completed_tasks[task_id] = result
                self.stats["completed_tasks"] += 1
                logger.debug(f"任务完成: {task_id}")
            except Exception as e:
                error_result = ProcessingResult(task_id, error=str(e))
                results[task_id] = error_result
                self.completed_tasks[task_id] = error_result
                self.stats["failed_tasks"] += 1
                logger.error(f"任务失败: {task_id}, 错误: {e}")
        
        # 更新统计信息
        processing_time = time.time() - start_time
        self.stats["total_processing_time"] += processing_time
        self.stats["average_processing_time"] = (
            self.stats["total_processing_time"] / self.stats["completed_tasks"]
            if self.stats["completed_tasks"] > 0 else 0
        )
        
        logger.info(f"批量处理完成: {len(results)} 个结果, 耗时: {processing_time:.2f}s")
        return results
    
    def _process_single_task(self, task: ProcessingTask) -> ProcessingResult:
        """
        处理单个任务
        
        Args:
            task: 处理任务
            
        Returns:
            ProcessingResult: 处理结果
        """
        try:
            start_time = time.time()
            
            # 根据任务类型选择处理函数
            if task.task_type == "color_reduction":
                result = self._process_color_reduction(task)
            elif task.task_type == "edge_enhancement":
                result = self._process_edge_enhancement(task)
            elif task.task_type == "noise_reduction":
                result = self._process_noise_reduction(task)
            elif task.task_type == "segmentation":
                result = self._process_segmentation(task)
            else:
                raise ValueError(f"不支持的任务类型: {task.task_type}")
            
            processing_time = time.time() - start_time
            logger.debug(f"任务 {task.task_id} 处理完成，耗时: {processing_time:.2f}s")
            
            return ProcessingResult(task.task_id, result=result)
            
        except Exception as e:
            logger.error(f"任务 {task.task_id} 处理失败: {e}")
            return ProcessingResult(task.task_id, error=str(e))
    
    def _process_color_reduction(self, task: ProcessingTask) -> np.ndarray:
        """颜色降色处理"""
        image = task.input_data
        n_colors = task.parameters.get("n_colors", 16)
        
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
    
    def _process_edge_enhancement(self, task: ProcessingTask) -> np.ndarray:
        """边缘增强处理"""
        image = task.input_data
        kernel_size = task.parameters.get("kernel_size", 3)
        
        # 转换为灰度图
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # 边缘检测
        edges = cv2.Laplacian(gray, cv2.CV_64F, ksize=kernel_size)
        edges = np.uint8(np.absolute(edges))
        
        # 增强边缘
        enhanced = cv2.addWeighted(gray, 1.5, edges, 0.5, 0)
        
        return enhanced
    
    def _process_noise_reduction(self, task: ProcessingTask) -> np.ndarray:
        """噪声清理处理"""
        image = task.input_data
        kernel_size = task.parameters.get("kernel_size", 5)
        
        # 双边滤波
        denoised = cv2.bilateralFilter(image, kernel_size, 75, 75)
        
        return denoised
    
    def _process_segmentation(self, task: ProcessingTask) -> np.ndarray:
        """图像分割处理"""
        image = task.input_data
        method = task.parameters.get("method", "grabcut")
        
        if method == "grabcut":
            # GrabCut分割
            mask = np.zeros(image.shape[:2], np.uint8)
            bgd_model = np.zeros((1, 65), np.float64)
            fgd_model = np.zeros((1, 65), np.float64)
            
            rect = (10, 10, image.shape[1]-20, image.shape[0]-20)
            cv2.grabCut(image, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
            
            mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
            return mask2 * 255
        
        elif method == "watershed":
            # 分水岭分割
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            kernel = np.ones((3, 3), np.uint8)
            opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
            
            sure_bg = cv2.dilate(opening, kernel, iterations=3)
            dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
            _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
            
            sure_fg = np.uint8(sure_fg)
            unknown = cv2.subtract(sure_bg, sure_fg)
            
            _, markers = cv2.connectedComponents(sure_fg)
            markers = markers + 1
            markers[unknown == 255] = 0
            
            markers = cv2.watershed(image, markers)
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            mask[markers > 1] = 255
            
            return mask
        
        else:
            raise ValueError(f"不支持的分割方法: {method}")
    
    def get_stats(self) -> Dict[str, Any]:
        """获取性能统计信息"""
        return self.stats.copy()
    
    def shutdown(self):
        """关闭处理器"""
        if self.executor:
            self.executor.shutdown(wait=True)
        logger.info("并行处理器已关闭")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()


class ImageProcessingPipeline:
    """
    图像处理流水线
    
    使用并行处理器构建高效的图像处理流水线
    """
    
    def __init__(self, max_workers: int = None):
        self.processor = ParallelProcessor(max_workers=max_workers)
        self.pipeline_steps = []
        
    def add_step(self, step_name: str, step_function: Callable, 
                 parameters: Dict[str, Any] = None):
        """添加处理步骤"""
        self.pipeline_steps.append({
            "name": step_name,
            "function": step_function,
            "parameters": parameters or {}
        })
    
    def process_image(self, image: np.ndarray, job_id: str) -> Dict[str, Any]:
        """
        处理图像
        
        Args:
            image: 输入图像
            job_id: 任务ID
            
        Returns:
            Dict[str, Any]: 处理结果
        """
        current_image = image.copy()
        results = {}
        
        for i, step in enumerate(self.pipeline_steps):
            step_id = f"{job_id}_step_{i}_{step['name']}"
            
            # 创建任务
            task = ProcessingTask(
                task_id=step_id,
                input_data=current_image,
                task_type=step['name'],
                parameters=step['parameters']
            )
            
            # 处理任务
            result = self.processor.process_batch([task])
            step_result = result[step_id]
            
            if step_result.error:
                raise Exception(f"步骤 {step['name']} 失败: {step_result.error}")
            
            # 更新当前图像和结果
            current_image = step_result.result
            results[step['name']] = step_result.result
        
        return results 