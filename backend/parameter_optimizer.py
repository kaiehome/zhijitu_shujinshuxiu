"""
参数自动调优系统
智能发现最佳参数组合，实现自动化优化
"""

import cv2
import numpy as np
import logging
from typing import Dict, Any, Tuple, List, Optional, Callable
from dataclasses import dataclass
import json
import time
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import optuna
from optuna.samplers import TPESampler

logger = logging.getLogger(__name__)


@dataclass
class ParameterSpace:
    """参数搜索空间"""
    color_clusters: Tuple[int, int] = (8, 32)  # (min, max)
    edge_smoothness: Tuple[float, float] = (0.5, 3.0)
    texture_detail_level: Tuple[float, float] = (0.3, 1.2)
    pattern_continuity_weight: Tuple[float, float] = (0.3, 1.0)
    noise_reduction: Tuple[float, float] = (0.1, 0.8)
    contrast_enhancement: Tuple[float, float] = (0.8, 1.8)


@dataclass
class OptimizationTrial:
    """优化试验"""
    trial_id: int
    parameters: Dict[str, Any]
    quality_score: float
    processing_time: float
    timestamp: float


class ParameterOptimizer:
    """参数自动调优器"""
    
    def __init__(self, 
                 optimization_function: Callable,
                 parameter_space: ParameterSpace = None,
                 n_trials: int = 100,
                 n_jobs: int = 4):
        """
        初始化参数优化器
        
        Args:
            optimization_function: 优化函数，接受参数字典，返回质量评分
            parameter_space: 参数搜索空间
            n_trials: 试验次数
            n_jobs: 并行作业数
        """
        self.optimization_function = optimization_function
        self.parameter_space = parameter_space or ParameterSpace()
        self.n_trials = n_trials
        self.n_jobs = n_jobs
        self.trials = []
        self.best_trial = None
        self.optimization_history = []
        
    def optimize(self, target_score: float = 0.9, timeout: int = 3600) -> Dict[str, Any]:
        """执行参数优化"""
        logger.info(f"开始参数优化，目标评分: {target_score}, 超时: {timeout}秒")
        
        start_time = time.time()
        
        # 创建Optuna研究
        study = optuna.create_study(
            direction="maximize",
            sampler=TPESampler(seed=42)
        )
        
        # 定义目标函数
        def objective(trial):
            # 生成参数
            params = self._generate_parameters(trial)
            
            try:
                # 执行优化
                quality_score = self.optimization_function(params)
                
                # 记录试验
                trial_result = OptimizationTrial(
                    trial_id=len(self.trials),
                    parameters=params,
                    quality_score=quality_score,
                    processing_time=time.time() - start_time,
                    timestamp=time.time()
                )
                self.trials.append(trial_result)
                
                # 更新最佳试验
                if self.best_trial is None or quality_score > self.best_trial.quality_score:
                    self.best_trial = trial_result
                
                # 检查是否达到目标
                if quality_score >= target_score:
                    logger.info(f"达到目标评分: {quality_score:.3f}")
                    study.stop()
                
                return quality_score
                
            except Exception as e:
                logger.warning(f"试验失败: {e}")
                return 0.0
        
        # 执行优化
        study.optimize(objective, n_trials=self.n_trials, timeout=timeout)
        
        # 记录优化历史
        self.optimization_history.append({
            'timestamp': time.time(),
            'n_trials': len(self.trials),
            'best_score': self.best_trial.quality_score if self.best_trial else 0.0,
            'best_parameters': self.best_trial.parameters if self.best_trial else {},
            'total_time': time.time() - start_time
        })
        
        logger.info(f"参数优化完成，最佳评分: {self.best_trial.quality_score:.3f}")
        
        return self.best_trial.parameters if self.best_trial else {}
    
    def _generate_parameters(self, trial) -> Dict[str, Any]:
        """生成试验参数"""
        params = {}
        
        # 颜色聚类数
        params['color_clusters'] = trial.suggest_int(
            'color_clusters',
            self.parameter_space.color_clusters[0],
            self.parameter_space.color_clusters[1]
        )
        
        # 边缘平滑度
        params['edge_smoothness'] = trial.suggest_float(
            'edge_smoothness',
            self.parameter_space.edge_smoothness[0],
            self.parameter_space.edge_smoothness[1]
        )
        
        # 纹理细节级别
        params['texture_detail_level'] = trial.suggest_float(
            'texture_detail_level',
            self.parameter_space.texture_detail_level[0],
            self.parameter_space.texture_detail_level[1]
        )
        
        # 图案连续性权重
        params['pattern_continuity_weight'] = trial.suggest_float(
            'pattern_continuity_weight',
            self.parameter_space.pattern_continuity_weight[0],
            self.parameter_space.pattern_continuity_weight[1]
        )
        
        # 噪声减少
        params['noise_reduction'] = trial.suggest_float(
            'noise_reduction',
            self.parameter_space.noise_reduction[0],
            self.parameter_space.noise_reduction[1]
        )
        
        # 对比度增强
        params['contrast_enhancement'] = trial.suggest_float(
            'contrast_enhancement',
            self.parameter_space.contrast_enhancement[0],
            self.parameter_space.contrast_enhancement[1]
        )
        
        return params
    
    def parallel_optimize(self, 
                         test_images: List[np.ndarray],
                         target_score: float = 0.9) -> Dict[str, Any]:
        """并行参数优化"""
        logger.info(f"开始并行参数优化，图像数量: {len(test_images)}")
        
        def evaluate_on_image(params: Dict[str, Any], image: np.ndarray) -> float:
            """在单个图像上评估参数"""
            try:
                from embroidery_optimizer import EmbroideryOptimizer, OptimizationParams
                
                optimizer = EmbroideryOptimizer()
                opt_params = OptimizationParams(**params)
                result = optimizer.optimize_embroidery(image, opt_params)
                
                # 计算平均改进
                improvements = result.quality_improvements
                return sum(improvements.values()) / len(improvements)
                
            except Exception as e:
                logger.warning(f"图像评估失败: {e}")
                return 0.0
        
        def objective(trial):
            params = self._generate_parameters(trial)
            
            # 在所有测试图像上评估
            scores = []
            with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
                futures = [
                    executor.submit(evaluate_on_image, params, image)
                    for image in test_images
                ]
                
                for future in as_completed(futures):
                    score = future.result()
                    scores.append(score)
            
            # 返回平均评分
            avg_score = np.mean(scores) if scores else 0.0
            
            # 记录试验
            trial_result = OptimizationTrial(
                trial_id=len(self.trials),
                parameters=params,
                quality_score=avg_score,
                processing_time=time.time(),
                timestamp=time.time()
            )
            self.trials.append(trial_result)
            
            return avg_score
        
        # 创建研究并优化
        study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=42))
        study.optimize(objective, n_trials=self.n_trials)
        
        # 更新最佳试验
        if study.best_trial:
            self.best_trial = OptimizationTrial(
                trial_id=len(self.trials) - 1,
                parameters=study.best_params,
                quality_score=study.best_value,
                processing_time=time.time(),
                timestamp=time.time()
            )
        
        return self.best_trial.parameters if self.best_trial else {}
    
    def build_parameter_model(self) -> RandomForestRegressor:
        """构建参数预测模型"""
        if len(self.trials) < 10:
            logger.warning("试验数据不足，无法构建模型")
            return None
        
        # 准备训练数据
        X = []
        y = []
        
        for trial in self.trials:
            features = [
                trial.parameters['color_clusters'],
                trial.parameters['edge_smoothness'],
                trial.parameters['texture_detail_level'],
                trial.parameters['pattern_continuity_weight'],
                trial.parameters['noise_reduction'],
                trial.parameters['contrast_enhancement']
            ]
            X.append(features)
            y.append(trial.quality_score)
        
        X = np.array(X)
        y = np.array(y)
        
        # 分割训练和测试数据
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # 训练模型
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # 评估模型
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        logger.info(f"参数预测模型MSE: {mse:.4f}")
        
        return model
    
    def predict_optimal_parameters(self, 
                                 image_features: Dict[str, float],
                                 model: RandomForestRegressor = None) -> Dict[str, Any]:
        """预测最优参数"""
        if model is None:
            model = self.build_parameter_model()
        
        if model is None:
            logger.warning("无法构建预测模型，返回默认参数")
            return {
                'color_clusters': 16,
                'edge_smoothness': 1.5,
                'texture_detail_level': 0.8,
                'pattern_continuity_weight': 0.6,
                'noise_reduction': 0.3,
                'contrast_enhancement': 1.2
            }
        
        # 提取图像特征
        features = [
            image_features.get('color_complexity', 0.5),
            image_features.get('edge_density', 0.5),
            image_features.get('texture_complexity', 0.5),
            image_features.get('pattern_complexity', 0.5),
            image_features.get('noise_level', 0.5),
            image_features.get('contrast_level', 0.5)
        ]
        
        # 预测参数
        predicted_params = model.predict([features])[0]
        
        # 将预测结果映射到参数空间
        params = self._map_prediction_to_parameters(predicted_params)
        
        return params
    
    def _map_prediction_to_parameters(self, prediction: float) -> Dict[str, Any]:
        """将预测结果映射到参数空间"""
        # 简化的映射策略
        params = {}
        
        # 根据预测评分调整参数
        if prediction > 0.8:
            # 高质量预测，使用精细参数
            params['color_clusters'] = 20
            params['edge_smoothness'] = 1.8
            params['texture_detail_level'] = 1.0
            params['pattern_continuity_weight'] = 0.8
            params['noise_reduction'] = 0.4
            params['contrast_enhancement'] = 1.3
        elif prediction > 0.6:
            # 中等质量预测，使用平衡参数
            params['color_clusters'] = 16
            params['edge_smoothness'] = 1.5
            params['texture_detail_level'] = 0.8
            params['pattern_continuity_weight'] = 0.6
            params['noise_reduction'] = 0.3
            params['contrast_enhancement'] = 1.2
        else:
            # 低质量预测，使用保守参数
            params['color_clusters'] = 12
            params['edge_smoothness'] = 1.2
            params['texture_detail_level'] = 0.6
            params['pattern_continuity_weight'] = 0.4
            params['noise_reduction'] = 0.2
            params['contrast_enhancement'] = 1.1
        
        return params
    
    def analyze_parameter_importance(self) -> Dict[str, float]:
        """分析参数重要性"""
        if len(self.trials) < 10:
            return {}
        
        # 构建模型
        model = self.build_parameter_model()
        if model is None:
            return {}
        
        # 获取特征重要性
        feature_names = [
            'color_clusters',
            'edge_smoothness', 
            'texture_detail_level',
            'pattern_continuity_weight',
            'noise_reduction',
            'contrast_enhancement'
        ]
        
        importance = dict(zip(feature_names, model.feature_importances_))
        
        # 按重要性排序
        sorted_importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
        
        return sorted_importance
    
    def visualize_optimization_history(self, output_path: str = None):
        """可视化优化历史"""
        if len(self.trials) < 2:
            logger.warning("试验数据不足，无法可视化")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 质量评分历史
        scores = [trial.quality_score for trial in self.trials]
        axes[0, 0].plot(scores)
        axes[0, 0].set_title('质量评分历史')
        axes[0, 0].set_xlabel('试验次数')
        axes[0, 0].set_ylabel('质量评分')
        axes[0, 0].grid(True)
        
        # 处理时间历史
        times = [trial.processing_time for trial in self.trials]
        axes[0, 1].plot(times)
        axes[0, 1].set_title('处理时间历史')
        axes[0, 1].set_xlabel('试验次数')
        axes[0, 1].set_ylabel('处理时间(秒)')
        axes[0, 1].grid(True)
        
        # 参数重要性
        importance = self.analyze_parameter_importance()
        if importance:
            params = list(importance.keys())
            values = list(importance.values())
            axes[1, 0].bar(params, values)
            axes[1, 0].set_title('参数重要性')
            axes[1, 0].set_ylabel('重要性')
            axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 最佳参数分布
        if self.best_trial:
            best_params = self.best_trial.parameters
            param_names = list(best_params.keys())
            param_values = list(best_params.values())
            axes[1, 1].bar(param_names, param_values)
            axes[1, 1].set_title('最佳参数值')
            axes[1, 1].set_ylabel('参数值')
            axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def save_optimization_results(self, output_path: str):
        """保存优化结果"""
        results = {
            'best_trial': {
                'trial_id': self.best_trial.trial_id if self.best_trial else None,
                'parameters': self.best_trial.parameters if self.best_trial else {},
                'quality_score': self.best_trial.quality_score if self.best_trial else 0.0,
                'processing_time': self.best_trial.processing_time if self.best_trial else 0.0
            },
            'all_trials': [
                {
                    'trial_id': trial.trial_id,
                    'parameters': trial.parameters,
                    'quality_score': trial.quality_score,
                    'processing_time': trial.processing_time,
                    'timestamp': trial.timestamp
                }
                for trial in self.trials
            ],
            'optimization_history': self.optimization_history,
            'parameter_importance': self.analyze_parameter_importance()
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"优化结果已保存: {output_path}")


if __name__ == "__main__":
    # 测试参数优化器
    print("参数自动调优系统测试")
    print("=" * 50)
    
    # 创建测试优化函数
    def test_optimization_function(params):
        """测试优化函数"""
        # 模拟优化过程
        time.sleep(0.1)
        
        # 计算模拟质量评分
        score = (
            params['color_clusters'] / 32.0 * 0.3 +
            params['edge_smoothness'] / 3.0 * 0.2 +
            params['texture_detail_level'] / 1.2 * 0.2 +
            params['pattern_continuity_weight'] * 0.15 +
            (1.0 - params['noise_reduction']) * 0.1 +
            params['contrast_enhancement'] / 1.8 * 0.05
        )
        
        # 添加一些随机性
        score += random.uniform(-0.1, 0.1)
        score = max(0.0, min(1.0, score))
        
        return score
    
    # 创建参数优化器
    optimizer = ParameterOptimizer(
        optimization_function=test_optimization_function,
        n_trials=20,
        n_jobs=2
    )
    
    # 执行优化
    best_params = optimizer.optimize(target_score=0.8)
    
    print(f"最佳参数: {best_params}")
    print(f"最佳评分: {optimizer.best_trial.quality_score:.3f}")
    
    # 分析参数重要性
    importance = optimizer.analyze_parameter_importance()
    print(f"参数重要性: {importance}")
    
    print("参数优化器测试完成") 