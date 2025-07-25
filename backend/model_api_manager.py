"""
统一模型API管理器
集成深度学习模型、图像优化器和质量分析器，提供完整的AI刺绣生成服务
"""

import cv2
import numpy as np
import logging
import time
import json
import os
from typing import Dict, Any, Optional, List, Tuple, Union
from pathlib import Path
from dataclasses import dataclass
import asyncio
from concurrent.futures import ThreadPoolExecutor

# 导入我们的优化组件
from embroidery_quality_analyzer import EmbroideryQualityAnalyzer
from embroidery_optimizer import EmbroideryOptimizer, OptimizationParams
from parameter_optimizer import ParameterOptimizer, ParameterSpace

# 导入现有的模型组件
try:
    from deep_learning_models import DeepLearningModelManager, ModelConfig
    from image_processor import SichuanBrocadeProcessor
    from simple_professional_generator import SimpleProfessionalGenerator
    from structural_professional_generator import StructuralProfessionalGenerator
    MODELS_AVAILABLE = True
except ImportError as e:
    logging.warning(f"部分模型组件不可用: {e}")
    MODELS_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class ModelAPIConfig:
    """模型API配置"""
    models_dir: str = "models"
    cache_dir: str = "cache"
    max_batch_size: int = 4
    enable_gpu: bool = True
    enable_optimization: bool = True
    enable_quality_analysis: bool = True
    default_style: str = "sichuan_brocade"
    timeout_seconds: int = 300


@dataclass
class GenerationRequest:
    """生成请求"""
    image_path: str
    style: str = "sichuan_brocade"
    color_count: int = 16
    edge_enhancement: bool = True
    noise_reduction: bool = True
    professional_mode: bool = True
    optimization_level: str = "balanced"  # conservative, balanced, aggressive
    enable_auto_tune: bool = True
    reference_image: Optional[str] = None


@dataclass
class GenerationResult:
    """生成结果"""
    job_id: str
    status: str  # processing, completed, failed
    original_image: np.ndarray
    generated_image: Optional[np.ndarray] = None
    optimized_image: Optional[np.ndarray] = None
    quality_metrics: Optional[Dict[str, Any]] = None
    optimization_params: Optional[Dict[str, Any]] = None
    processing_time: float = 0.0
    error_message: Optional[str] = None
    output_files: List[str] = None


class ModelAPIManager:
    """统一模型API管理器"""

    def __init__(self, config: ModelAPIConfig = None):
        self.config = config or ModelAPIConfig()
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # 初始化组件
        self._initialize_components()
        
        # 任务管理
        self.active_jobs: Dict[str, GenerationResult] = {}
        self.job_history: List[GenerationResult] = []
        
        logger.info("模型API管理器初始化完成")

    def _initialize_components(self):
        """初始化各个组件"""
        try:
            # 初始化质量分析器
            self.quality_analyzer = EmbroideryQualityAnalyzer()
            logger.info("质量分析器初始化完成")
        except Exception as e:
            logger.error(f"质量分析器初始化失败: {e}")
            self.quality_analyzer = None

        try:
            # 初始化图像优化器
            self.embroidery_optimizer = EmbroideryOptimizer()
            logger.info("图像优化器初始化完成")
        except Exception as e:
            logger.error(f"图像优化器初始化失败: {e}")
            self.embroidery_optimizer = None

        # 初始化深度学习模型管理器
        if MODELS_AVAILABLE:
            try:
                self.model_manager = DeepLearningModelManager(self.config.models_dir)
                logger.info("深度学习模型管理器初始化完成")
            except Exception as e:
                logger.error(f"深度学习模型管理器初始化失败: {e}")
                self.model_manager = None
        else:
            self.model_manager = None

        # 初始化图像处理器
        if MODELS_AVAILABLE:
            try:
                self.brocade_processor = SichuanBrocadeProcessor()
                self.simple_generator = SimpleProfessionalGenerator()
                self.structural_generator = StructuralProfessionalGenerator()
                logger.info("图像处理器初始化完成")
            except Exception as e:
                logger.error(f"图像处理器初始化失败: {e}")
                self.brocade_processor = None
                self.simple_generator = None
                self.structural_generator = None

    async def generate_embroidery(self, request: GenerationRequest) -> str:
        """生成刺绣图像（异步）"""
        job_id = self._generate_job_id()
        
        # 创建结果对象
        result = GenerationResult(
            job_id=job_id,
            status="processing",
            original_image=None,
            output_files=[]
        )
        
        self.active_jobs[job_id] = result
        
        # 异步执行生成任务
        asyncio.create_task(self._process_generation_job(job_id, request))
        
        return job_id

    async def _process_generation_job(self, job_id: str, request: GenerationRequest):
        """处理生成任务"""
        result = self.active_jobs[job_id]
        start_time = time.time()
        
        try:
            logger.info(f"开始处理任务 {job_id}")
            
            # 1. 加载原始图像
            original_image = cv2.imread(request.image_path)
            if original_image is None:
                raise ValueError(f"无法加载图像: {request.image_path}")
            
            result.original_image = original_image
            
            # 2. 使用深度学习模型生成基础图像
            generated_image = await self._generate_base_image(original_image, request)
            result.generated_image = generated_image
            
            # 3. 质量分析（如果有参考图像）
            if request.reference_image and self.quality_analyzer:
                quality_metrics = await self._analyze_quality(generated_image, request.reference_image)
                result.quality_metrics = quality_metrics
            
            # 4. 图像优化
            if self.embroidery_optimizer and self.config.enable_optimization:
                optimized_image, opt_params = await self._optimize_image(generated_image, request)
                result.optimized_image = optimized_image
                result.optimization_params = opt_params
                
                # 保存优化后的图像
                output_path = self._save_result_image(optimized_image, job_id, "optimized")
                result.output_files.append(output_path)
            else:
                # 保存生成的图像
                output_path = self._save_result_image(generated_image, job_id, "generated")
                result.output_files.append(output_path)
            
            # 5. 更新状态
            result.status = "completed"
            result.processing_time = time.time() - start_time
            
            logger.info(f"任务 {job_id} 处理完成，耗时: {result.processing_time:.2f}秒")
            
        except Exception as e:
            logger.error(f"任务 {job_id} 处理失败: {e}")
            result.status = "failed"
            result.error_message = str(e)
            result.processing_time = time.time() - start_time
        
        # 移动到历史记录
        self.job_history.append(result)
        if job_id in self.active_jobs:
            del self.active_jobs[job_id]

    async def _generate_base_image(self, image: np.ndarray, request: GenerationRequest) -> np.ndarray:
        """生成基础图像"""
        logger.info(f"使用风格 {request.style} 生成基础图像")
        
        if not MODELS_AVAILABLE:
            # 如果没有模型，返回原图
            return image
        
        try:
            if request.style == "sichuan_brocade" and self.brocade_processor:
                # 使用蜀锦处理器
                return self.brocade_processor.process_image(
                    image, 
                    color_count=request.color_count,
                    edge_enhancement=request.edge_enhancement,
                    noise_reduction=request.noise_reduction
                )
            
            elif request.style == "structural" and self.structural_generator:
                # 使用结构化生成器
                return self.structural_generator.generate(
                    image,
                    color_count=request.color_count,
                    professional_mode=request.professional_mode
                )
            
            elif self.simple_generator:
                # 使用简单专业生成器
                return self.simple_generator.generate(
                    image,
                    color_count=request.color_count,
                    edge_enhancement=request.edge_enhancement,
                    noise_reduction=request.noise_reduction
                )
            
            else:
                # 降级到基础处理
                return self._basic_image_processing(image, request)
                
        except Exception as e:
            logger.warning(f"模型生成失败，使用基础处理: {e}")
            return self._basic_image_processing(image, request)

    def _basic_image_processing(self, image: np.ndarray, request: GenerationRequest) -> np.ndarray:
        """基础图像处理"""
        processed = image.copy()
        
        # 颜色量化
        if request.color_count < 256:
            processed = self._quantize_colors(processed, request.color_count)
        
        # 边缘增强
        if request.edge_enhancement:
            processed = self._enhance_edges(processed)
        
        # 噪声减少
        if request.noise_reduction:
            processed = self._reduce_noise(processed)
        
        return processed

    def _quantize_colors(self, image: np.ndarray, color_count: int) -> np.ndarray:
        """颜色量化"""
        # 重塑图像为像素列表
        pixels = image.reshape(-1, 3)
        
        # 使用K-means聚类
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=color_count, random_state=42, n_init=10)
        labels = kmeans.fit_predict(pixels)
        
        # 使用聚类中心替换像素
        quantized_pixels = kmeans.cluster_centers_[labels]
        quantized_image = quantized_pixels.reshape(image.shape).astype(np.uint8)
        
        return quantized_image

    def _enhance_edges(self, image: np.ndarray) -> np.ndarray:
        """边缘增强"""
        # 使用拉普拉斯算子增强边缘
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        enhanced = cv2.filter2D(image, -1, kernel)
        return enhanced

    def _reduce_noise(self, image: np.ndarray) -> np.ndarray:
        """噪声减少"""
        # 使用双边滤波减少噪声
        denoised = cv2.bilateralFilter(image, 9, 75, 75)
        return denoised

    async def _analyze_quality(self, generated_image: np.ndarray, reference_path: str) -> Dict[str, Any]:
        """分析图像质量"""
        if not self.quality_analyzer:
            return {}
        
        try:
            analysis = self.quality_analyzer.analyze_embroidery_quality(
                generated_image, reference_path
            )
            return {
                "overall_score": analysis.quality_metrics.overall_score,
                "color_accuracy": analysis.quality_metrics.color_accuracy,
                "edge_quality": analysis.quality_metrics.edge_quality,
                "texture_detail": analysis.quality_metrics.texture_detail,
                "pattern_continuity": analysis.quality_metrics.pattern_continuity,
                "recommendations": analysis.quality_metrics.recommendations
            }
        except Exception as e:
            logger.error(f"质量分析失败: {e}")
            return {}

    async def _optimize_image(self, image: np.ndarray, request: GenerationRequest) -> Tuple[np.ndarray, Dict[str, Any]]:
        """优化图像"""
        if not self.embroidery_optimizer:
            return image, {}
        
        try:
            # 根据优化级别选择参数
            if request.optimization_level == "conservative":
                params = OptimizationParams(
                    color_clusters=12,
                    edge_smoothness=1.0,
                    texture_detail_level=0.6,
                    pattern_continuity_weight=0.4,
                    noise_reduction=0.2,
                    contrast_enhancement=1.1
                )
            elif request.optimization_level == "aggressive":
                params = OptimizationParams(
                    color_clusters=20,
                    edge_smoothness=2.0,
                    texture_detail_level=1.0,
                    pattern_continuity_weight=0.8,
                    noise_reduction=0.4,
                    contrast_enhancement=1.4
                )
            else:  # balanced
                params = OptimizationParams(
                    color_clusters=16,
                    edge_smoothness=1.5,
                    texture_detail_level=0.8,
                    pattern_continuity_weight=0.6,
                    noise_reduction=0.3,
                    contrast_enhancement=1.2
                )
            
            # 执行优化
            result = self.embroidery_optimizer.optimize_embroidery(image, params)
            
            return result.optimized_image, {
                "params": result.optimization_params.__dict__,
                "improvements": result.quality_improvements
            }
            
        except Exception as e:
            logger.error(f"图像优化失败: {e}")
            return image, {}

    def _save_result_image(self, image: np.ndarray, job_id: str, suffix: str) -> str:
        """保存结果图像"""
        output_dir = Path("outputs") / job_id
        output_dir.mkdir(parents=True, exist_ok=True)
        
        filename = f"{job_id}_{suffix}.png"
        output_path = output_dir / filename
        
        cv2.imwrite(str(output_path), image)
        return str(output_path)

    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """获取任务状态"""
        # 检查活动任务
        if job_id in self.active_jobs:
            result = self.active_jobs[job_id]
            return {
                "job_id": result.job_id,
                "status": result.status,
                "processing_time": result.processing_time,
                "error_message": result.error_message,
                "output_files": result.output_files
            }
        
        # 检查历史任务
        for result in self.job_history:
            if result.job_id == job_id:
                return {
                    "job_id": result.job_id,
                    "status": result.status,
                    "processing_time": result.processing_time,
                    "error_message": result.error_message,
                    "output_files": result.output_files,
                    "quality_metrics": result.quality_metrics,
                    "optimization_params": result.optimization_params
                }
        
        return None

    def get_available_models(self) -> Dict[str, Any]:
        """获取可用模型信息"""
        models_info = {
            "deep_learning_models": [],
            "image_processors": [],
            "optimizers": [],
            "quality_analyzers": []
        }
        
        if self.model_manager:
            models_info["deep_learning_models"] = self.model_manager.get_available_models()
        
        if self.brocade_processor:
            models_info["image_processors"].append("sichuan_brocade")
        
        if self.simple_generator:
            models_info["image_processors"].append("simple_professional")
        
        if self.structural_generator:
            models_info["image_processors"].append("structural_professional")
        
        if self.embroidery_optimizer:
            models_info["optimizers"].append("embroidery_optimizer")
        
        if self.quality_analyzer:
            models_info["quality_analyzers"].append("embroidery_quality_analyzer")
        
        return models_info

    def get_system_stats(self) -> Dict[str, Any]:
        """获取系统统计信息"""
        return {
            "active_jobs": len(self.active_jobs),
            "completed_jobs": len(self.job_history),
            "available_models": self.get_available_models(),
            "config": self.config.__dict__
        }

    def _generate_job_id(self) -> str:
        """生成任务ID"""
        return str(int(time.time() * 1000))


# 创建全局API管理器实例
api_manager = ModelAPIManager()


# FastAPI路由函数
async def generate_embroidery_api(request: GenerationRequest) -> str:
    """生成刺绣图像API"""
    return await api_manager.generate_embroidery(request)


def get_job_status_api(job_id: str) -> Optional[Dict[str, Any]]:
    """获取任务状态API"""
    return api_manager.get_job_status(job_id)


def get_available_models_api() -> Dict[str, Any]:
    """获取可用模型API"""
    return api_manager.get_available_models()


def get_system_stats_api() -> Dict[str, Any]:
    """获取系统统计API"""
    return api_manager.get_system_stats() 