"""
质量评估系统
实现PSNR、SSIM等客观质量指标和主观质量评估
"""

import numpy as np
import cv2
import logging
import time
from typing import Dict, Any, Optional, Tuple, List, Union
from pathlib import Path
import json
import math
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class QualityMetric(Enum):
    """质量指标枚举"""
    PSNR = "psnr"
    SSIM = "ssim"
    MSE = "mse"
    MAE = "mae"
    SNR = "snr"
    UIQI = "uiqi"
    VIF = "vif"


@dataclass
class QualityScore:
    """质量评分数据类"""
    metric: str
    value: float
    unit: str
    description: str
    timestamp: float
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "metric": self.metric,
            "value": self.value,
            "unit": self.unit,
            "description": self.description,
            "timestamp": self.timestamp
        }


class ObjectiveQualityMetrics:
    """客观质量指标计算"""
    
    @staticmethod
    def calculate_psnr(original: np.ndarray, processed: np.ndarray) -> float:
        """
        计算峰值信噪比 (Peak Signal-to-Noise Ratio)
        
        Args:
            original: 原始图像
            processed: 处理后的图像
            
        Returns:
            float: PSNR值 (dB)
        """
        try:
            # 确保图像类型一致
            original = original.astype(np.float64)
            processed = processed.astype(np.float64)
            
            # 计算MSE
            mse = np.mean((original - processed) ** 2)
            
            if mse == 0:
                return float('inf')  # 完美匹配
            
            # 计算PSNR
            max_pixel = 255.0
            psnr = 20 * math.log10(max_pixel / math.sqrt(mse))
            
            return psnr
            
        except Exception as e:
            logger.error(f"PSNR计算失败: {e}")
            return 0.0
    
    @staticmethod
    def calculate_ssim(original: np.ndarray, processed: np.ndarray, 
                      window_size: int = 11) -> float:
        """
        计算结构相似性指数 (Structural Similarity Index)
        
        Args:
            original: 原始图像
            processed: 处理后的图像
            window_size: 窗口大小
            
        Returns:
            float: SSIM值 (0-1)
        """
        try:
            # 转换为灰度图
            if len(original.shape) == 3:
                original_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
            else:
                original_gray = original
            
            if len(processed.shape) == 3:
                processed_gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
            else:
                processed_gray = processed
            
            # 确保尺寸一致
            if original_gray.shape != processed_gray.shape:
                processed_gray = cv2.resize(processed_gray, 
                                          (original_gray.shape[1], original_gray.shape[0]))
            
            # 计算SSIM
            ssim = ObjectiveQualityMetrics._ssim_simple(original_gray, processed_gray)
            
            return ssim
            
        except Exception as e:
            logger.error(f"SSIM计算失败: {e}")
            return 0.0
    
    @staticmethod
    def _ssim_simple(img1: np.ndarray, img2: np.ndarray, 
                    window_size: int = 11, sigma: float = 1.5) -> float:
        """简化的SSIM计算"""
        # 创建高斯窗口
        window = ObjectiveQualityMetrics._create_gaussian_window(window_size, sigma)
        
        # 计算均值
        mu1 = cv2.filter2D(img1.astype(np.float64), -1, window)
        mu2 = cv2.filter2D(img2.astype(np.float64), -1, window)
        
        # 计算方差和协方差
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = cv2.filter2D(img1.astype(np.float64) ** 2, -1, window) - mu1_sq
        sigma2_sq = cv2.filter2D(img2.astype(np.float64) ** 2, -1, window) - mu2_sq
        sigma12 = cv2.filter2D(img1.astype(np.float64) * img2.astype(np.float64), -1, window) - mu1_mu2
        
        # SSIM参数
        C1 = (0.01 * 255) ** 2
        C2 = (0.03 * 255) ** 2
        
        # 计算SSIM
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        
        return np.mean(ssim_map)
    
    @staticmethod
    def _create_gaussian_window(size: int, sigma: float) -> np.ndarray:
        """创建高斯窗口"""
        ax = np.arange(-size // 2 + 1., size // 2 + 1.)
        xx, yy = np.meshgrid(ax, ax)
        kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        return kernel / np.sum(kernel)
    
    @staticmethod
    def calculate_mse(original: np.ndarray, processed: np.ndarray) -> float:
        """
        计算均方误差 (Mean Squared Error)
        
        Args:
            original: 原始图像
            processed: 处理后的图像
            
        Returns:
            float: MSE值
        """
        try:
            original = original.astype(np.float64)
            processed = processed.astype(np.float64)
            
            mse = np.mean((original - processed) ** 2)
            return mse
            
        except Exception as e:
            logger.error(f"MSE计算失败: {e}")
            return float('inf')
    
    @staticmethod
    def calculate_mae(original: np.ndarray, processed: np.ndarray) -> float:
        """
        计算平均绝对误差 (Mean Absolute Error)
        
        Args:
            original: 原始图像
            processed: 处理后的图像
            
        Returns:
            float: MAE值
        """
        try:
            original = original.astype(np.float64)
            processed = processed.astype(np.float64)
            
            mae = np.mean(np.abs(original - processed))
            return mae
            
        except Exception as e:
            logger.error(f"MAE计算失败: {e}")
            return float('inf')
    
    @staticmethod
    def calculate_snr(original: np.ndarray, processed: np.ndarray) -> float:
        """
        计算信噪比 (Signal-to-Noise Ratio)
        
        Args:
            original: 原始图像
            processed: 处理后的图像
            
        Returns:
            float: SNR值 (dB)
        """
        try:
            original = original.astype(np.float64)
            processed = processed.astype(np.float64)
            
            # 计算信号功率
            signal_power = np.mean(original ** 2)
            
            # 计算噪声功率
            noise = original - processed
            noise_power = np.mean(noise ** 2)
            
            if noise_power == 0:
                return float('inf')
            
            snr = 10 * math.log10(signal_power / noise_power)
            return snr
            
        except Exception as e:
            logger.error(f"SNR计算失败: {e}")
            return 0.0


class SubjectiveQualityAssessment:
    """主观质量评估"""
    
    def __init__(self):
        self.assessment_criteria = {
            "color_accuracy": {
                "weight": 0.3,
                "description": "颜色准确性"
            },
            "edge_preservation": {
                "weight": 0.25,
                "description": "边缘保持"
            },
            "detail_preservation": {
                "weight": 0.25,
                "description": "细节保持"
            },
            "overall_quality": {
                "weight": 0.2,
                "description": "整体质量"
            }
        }
    
    def assess_image_quality(self, original: np.ndarray, processed: np.ndarray,
                           scores: Dict[str, float]) -> Dict[str, Any]:
        """
        主观质量评估
        
        Args:
            original: 原始图像
            processed: 处理后的图像
            scores: 主观评分 (1-10)
            
        Returns:
            Dict[str, Any]: 评估结果
        """
        try:
            # 验证评分
            for criterion in self.assessment_criteria:
                if criterion not in scores:
                    raise ValueError(f"缺少评分标准: {criterion}")
                if not (1 <= scores[criterion] <= 10):
                    raise ValueError(f"评分超出范围 (1-10): {criterion}")
            
            # 计算加权总分
            total_score = 0
            for criterion, config in self.assessment_criteria.items():
                total_score += scores[criterion] * config["weight"]
            
            # 计算客观指标作为参考
            objective_metrics = {
                "psnr": ObjectiveQualityMetrics.calculate_psnr(original, processed),
                "ssim": ObjectiveQualityMetrics.calculate_ssim(original, processed),
                "mse": ObjectiveQualityMetrics.calculate_mse(original, processed)
            }
            
            # 质量等级评估
            quality_level = self._get_quality_level(total_score, objective_metrics)
            
            result = {
                "subjective_scores": scores,
                "total_score": total_score,
                "objective_metrics": objective_metrics,
                "quality_level": quality_level,
                "timestamp": time.time(),
                "assessment_criteria": self.assessment_criteria
            }
            
            return result
            
        except Exception as e:
            logger.error(f"主观质量评估失败: {e}")
            raise
    
    def _get_quality_level(self, subjective_score: float, 
                          objective_metrics: Dict[str, float]) -> str:
        """获取质量等级"""
        # 基于主观评分和客观指标的综合评估
        if subjective_score >= 8.5 and objective_metrics["psnr"] >= 30:
            return "优秀"
        elif subjective_score >= 7.0 and objective_metrics["psnr"] >= 25:
            return "良好"
        elif subjective_score >= 5.5 and objective_metrics["psnr"] >= 20:
            return "一般"
        else:
            return "较差"


class QualityAssessmentSystem:
    """质量评估系统"""
    
    def __init__(self, output_dir: str = "quality_assessment"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        self.objective_metrics = ObjectiveQualityMetrics()
        self.subjective_assessment = SubjectiveQualityAssessment()
        
        # 评估历史
        self.assessment_history = []
        
        logger.info("质量评估系统初始化完成")
    
    def comprehensive_assessment(self, original_path: str, processed_path: str,
                               job_id: str, subjective_scores: Dict[str, float] = None) -> Dict[str, Any]:
        """
        综合质量评估
        
        Args:
            original_path: 原始图像路径
            processed_path: 处理后图像路径
            job_id: 任务ID
            subjective_scores: 主观评分
            
        Returns:
            Dict[str, Any]: 综合评估结果
        """
        try:
            logger.info(f"开始综合质量评估: {job_id}")
            
            # 加载图像
            original = cv2.imread(original_path)
            processed = cv2.imread(processed_path)
            
            if original is None or processed is None:
                raise ValueError("无法加载图像")
            
            # 确保尺寸一致
            if original.shape != processed.shape:
                processed = cv2.resize(processed, (original.shape[1], original.shape[0]))
            
            # 客观指标评估
            objective_results = self._objective_assessment(original, processed)
            
            # 主观质量评估
            subjective_results = None
            if subjective_scores:
                subjective_results = self.subjective_assessment.assess_image_quality(
                    original, processed, subjective_scores
                )
            
            # 综合评估结果
            assessment_result = {
                "job_id": job_id,
                "timestamp": time.time(),
                "objective_metrics": objective_results,
                "subjective_assessment": subjective_results,
                "overall_quality": self._calculate_overall_quality(objective_results, subjective_results)
            }
            
            # 保存评估结果
            self._save_assessment_result(assessment_result, job_id)
            
            # 添加到历史记录
            self.assessment_history.append(assessment_result)
            
            logger.info(f"综合质量评估完成: {job_id}")
            return assessment_result
            
        except Exception as e:
            logger.error(f"综合质量评估失败: {job_id}, 错误: {e}")
            raise
    
    def _objective_assessment(self, original: np.ndarray, processed: np.ndarray) -> Dict[str, QualityScore]:
        """客观指标评估"""
        results = {}
        
        # 计算各种客观指标
        metrics = [
            ("psnr", ObjectiveQualityMetrics.calculate_psnr(original, processed), "dB", "峰值信噪比"),
            ("ssim", ObjectiveQualityMetrics.calculate_ssim(original, processed), "", "结构相似性"),
            ("mse", ObjectiveQualityMetrics.calculate_mse(original, processed), "", "均方误差"),
            ("mae", ObjectiveQualityMetrics.calculate_mae(original, processed), "", "平均绝对误差"),
            ("snr", ObjectiveQualityMetrics.calculate_snr(original, processed), "dB", "信噪比")
        ]
        
        for metric_name, value, unit, description in metrics:
            score = QualityScore(
                metric=metric_name,
                value=value,
                unit=unit,
                description=description,
                timestamp=time.time()
            )
            results[metric_name] = score
        
        return results
    
    def _calculate_overall_quality(self, objective_results: Dict[str, QualityScore],
                                 subjective_results: Dict[str, Any]) -> Dict[str, Any]:
        """计算整体质量评分"""
        overall_quality = {
            "score": 0.0,
            "level": "未知",
            "confidence": 0.0
        }
        
        # 基于客观指标计算基础分数
        if objective_results:
            # PSNR权重最高
            psnr_score = objective_results.get("psnr", QualityScore("psnr", 0, "dB", "", 0))
            ssim_score = objective_results.get("ssim", QualityScore("ssim", 0, "", "", 0))
            
            # 归一化PSNR (假设30dB为满分)
            normalized_psnr = min(psnr_score.value / 30.0, 1.0)
            normalized_ssim = ssim_score.value  # SSIM已经是0-1范围
            
            # 加权计算
            objective_score = 0.6 * normalized_psnr + 0.4 * normalized_ssim
            overall_quality["score"] = objective_score * 10  # 转换为1-10分制
        
        # 如果有主观评分，进行加权
        if subjective_results:
            subjective_score = subjective_results.get("total_score", 0)
            # 客观指标权重0.4，主观评分权重0.6
            overall_quality["score"] = 0.4 * overall_quality["score"] + 0.6 * subjective_score
        
        # 确定质量等级
        score = overall_quality["score"]
        if score >= 8.5:
            overall_quality["level"] = "优秀"
            overall_quality["confidence"] = 0.9
        elif score >= 7.0:
            overall_quality["level"] = "良好"
            overall_quality["confidence"] = 0.8
        elif score >= 5.5:
            overall_quality["level"] = "一般"
            overall_quality["confidence"] = 0.7
        else:
            overall_quality["level"] = "较差"
            overall_quality["confidence"] = 0.6
        
        return overall_quality
    
    def _save_assessment_result(self, result: Dict[str, Any], job_id: str):
        """保存评估结果"""
        timestamp = time.strftime("%y%m%d_%H%M%S")
        filename = f"quality_assessment_{timestamp}_{job_id}.json"
        filepath = self.output_dir / filename
        
        # 转换QualityScore对象为字典
        serializable_result = self._make_serializable(result)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(serializable_result, f, indent=2, ensure_ascii=False)
        
        logger.info(f"评估结果已保存: {filepath}")
    
    def _make_serializable(self, obj: Any) -> Any:
        """使对象可序列化"""
        if isinstance(obj, QualityScore):
            return obj.to_dict()
        elif isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        else:
            return obj
    
    def get_assessment_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """获取评估历史"""
        return self.assessment_history[-limit:]
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取评估统计信息"""
        if not self.assessment_history:
            return {"message": "暂无评估历史"}
        
        # 计算统计信息
        psnr_values = []
        ssim_values = []
        overall_scores = []
        
        for assessment in self.assessment_history:
            objective = assessment.get("objective_metrics", {})
            if "psnr" in objective:
                psnr_values.append(objective["psnr"].value)
            if "ssim" in objective:
                ssim_values.append(objective["ssim"].value)
            
            overall = assessment.get("overall_quality", {})
            if "score" in overall:
                overall_scores.append(overall["score"])
        
        stats = {
            "total_assessments": len(self.assessment_history),
            "psnr": {
                "mean": np.mean(psnr_values) if psnr_values else 0,
                "std": np.std(psnr_values) if psnr_values else 0,
                "min": np.min(psnr_values) if psnr_values else 0,
                "max": np.max(psnr_values) if psnr_values else 0
            },
            "ssim": {
                "mean": np.mean(ssim_values) if ssim_values else 0,
                "std": np.std(ssim_values) if ssim_values else 0,
                "min": np.min(ssim_values) if ssim_values else 0,
                "max": np.max(ssim_values) if ssim_values else 0
            },
            "overall_score": {
                "mean": np.mean(overall_scores) if overall_scores else 0,
                "std": np.std(overall_scores) if overall_scores else 0,
                "min": np.min(overall_scores) if overall_scores else 0,
                "max": np.max(overall_scores) if overall_scores else 0
            }
        }
        
        return stats 