"""
综合优化演示系统
展示刺绣质量分析、优化和参数调优的完整流程
"""

import cv2
import numpy as np
import logging
import time
import json
import os
from typing import Dict, Any, List, Tuple
import matplotlib.pyplot as plt
from pathlib import Path

# 导入我们的优化组件
from embroidery_quality_analyzer import EmbroideryQualityAnalyzer
from embroidery_optimizer import EmbroideryOptimizer, OptimizationParams
from parameter_optimizer import ParameterOptimizer, ParameterSpace

logger = logging.getLogger(__name__)


class ComprehensiveOptimizationDemo:
    """综合优化演示系统"""
    
    def __init__(self, output_dir: str = "optimization_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # 初始化组件
        self.quality_analyzer = EmbroideryQualityAnalyzer()
        self.embroidery_optimizer = EmbroideryOptimizer()
        self.parameter_optimizer = None
        
        # 演示结果
        self.demo_results = {
            'quality_analysis': {},
            'optimization_results': {},
            'parameter_optimization': {},
            'comparison_results': {}
        }
    
    def create_test_images(self) -> Tuple[np.ndarray, np.ndarray]:
        """创建测试图像"""
        logger.info("创建测试图像...")
        
        # 创建生成图像（模拟当前系统输出）
        generated_image = self._create_generated_image()
        
        # 创建参考图像（模拟真实刺绣）
        reference_image = self._create_reference_image()
        
        # 保存测试图像
        cv2.imwrite(str(self.output_dir / "generated_test.png"), generated_image)
        cv2.imwrite(str(self.output_dir / "reference_test.png"), reference_image)
        
        return generated_image, reference_image
    
    def _create_generated_image(self) -> np.ndarray:
        """创建模拟的生成图像"""
        # 创建一个简单的熊猫图案（模拟当前系统的输出）
        image = np.ones((512, 512, 3), dtype=np.uint8) * 240  # 浅灰色背景
        
        # 熊猫头部（简化的像素化效果）
        # 白色区域
        cv2.rectangle(image, (200, 150), (312, 250), (240, 240, 200), -1)
        
        # 黑色眼睛区域
        cv2.rectangle(image, (220, 180), (240, 200), (50, 50, 50), -1)
        cv2.rectangle(image, (272, 180), (292, 200), (50, 50, 50), -1)
        
        # 黑色耳朵
        cv2.ellipse(image, (230, 160), (20, 15), 0, 0, 360, (50, 50, 50), -1)
        cv2.ellipse(image, (282, 160), (20, 15), 0, 0, 360, (50, 50, 50), -1)
        
        # 添加一些像素化效果
        for i in range(0, 512, 8):
            for j in range(0, 512, 8):
                block_color = np.mean(image[i:i+8, j:j+8], axis=(0, 1))
                image[i:i+8, j:j+8] = block_color
        
        return image
    
    def _create_reference_image(self) -> np.ndarray:
        """创建模拟的参考图像（真实刺绣效果）"""
        # 创建一个更自然的熊猫图案
        image = np.ones((512, 512, 3), dtype=np.uint8) * 245  # 更亮的背景
        
        # 熊猫头部（更自然的形状）
        # 白色区域
        cv2.ellipse(image, (256, 200), (80, 60), 0, 0, 360, (250, 250, 220), -1)
        
        # 黑色眼睛区域（更自然）
        cv2.ellipse(image, (230, 185), (12, 15), 0, 0, 360, (30, 30, 30), -1)
        cv2.ellipse(image, (282, 185), (12, 15), 0, 0, 360, (30, 30, 30), -1)
        
        # 眼睛高光
        cv2.circle(image, (228, 182), 3, (255, 255, 255), -1)
        cv2.circle(image, (280, 182), 3, (255, 255, 255), -1)
        
        # 黑色耳朵（更自然）
        cv2.ellipse(image, (220, 150), (25, 20), 30, 0, 360, (20, 20, 20), -1)
        cv2.ellipse(image, (292, 150), (25, 20), -30, 0, 360, (20, 20, 20), -1)
        
        # 添加纹理效果
        noise = np.random.normal(0, 10, image.shape).astype(np.int16)
        image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        # 添加轻微的模糊效果
        image = cv2.GaussianBlur(image, (3, 3), 0.5)
        
        return image
    
    def run_quality_analysis(self, generated_image: np.ndarray, reference_image: np.ndarray):
        """运行质量分析"""
        logger.info("运行质量分析...")
        
        # 保存临时图像文件
        generated_path = str(self.output_dir / "temp_generated.png")
        reference_path = str(self.output_dir / "temp_reference.png")
        cv2.imwrite(generated_path, generated_image)
        cv2.imwrite(reference_path, reference_image)
        
        # 执行质量分析
        analysis = self.quality_analyzer.analyze_embroidery_quality(generated_path, reference_path)
        
        # 生成分析报告
        report_path = str(self.output_dir / "quality_analysis_report.json")
        self.quality_analyzer.generate_analysis_report(analysis, report_path)
        
        # 生成可视化
        viz_path = str(self.output_dir / "quality_analysis_visualization.png")
        self.quality_analyzer.visualize_analysis(analysis, viz_path)
        
        # 记录结果
        self.demo_results['quality_analysis'] = {
            'overall_score': analysis.quality_metrics.overall_score,
            'detailed_metrics': {
                'color_accuracy': analysis.quality_metrics.color_accuracy,
                'edge_quality': analysis.quality_metrics.edge_quality,
                'texture_detail': analysis.quality_metrics.texture_detail,
                'pattern_continuity': analysis.quality_metrics.pattern_continuity
            },
            'recommendations': analysis.quality_metrics.recommendations
        }
        
        # 清理临时文件
        os.remove(generated_path)
        os.remove(reference_path)
        
        return analysis
    
    def run_optimization_demo(self, generated_image: np.ndarray):
        """运行优化演示"""
        logger.info("运行优化演示...")
        
        # 测试不同的优化参数
        optimization_configs = [
            ("保守优化", OptimizationParams(12, 1.0, 0.6, 0.4, 0.2, 1.0)),
            ("平衡优化", OptimizationParams(16, 1.5, 0.8, 0.6, 0.3, 1.2)),
            ("激进优化", OptimizationParams(20, 2.0, 1.0, 0.8, 0.5, 1.4)),
            ("艺术风格", OptimizationParams(18, 1.8, 0.9, 0.7, 0.4, 1.3))
        ]
        
        optimization_results = {}
        
        for config_name, params in optimization_configs:
            logger.info(f"执行 {config_name}...")
            
            # 执行优化
            result = self.embroidery_optimizer.optimize_embroidery(
                generated_image, params, target_style="realistic"
            )
            
            # 保存优化结果
            output_path = str(self.output_dir / f"optimized_{config_name}.png")
            self.embroidery_optimizer.save_optimization_result(result, output_path)
            
            # 记录结果
            optimization_results[config_name] = {
                'parameters': {
                    'color_clusters': params.color_clusters,
                    'edge_smoothness': params.edge_smoothness,
                    'texture_detail_level': params.texture_detail_level,
                    'pattern_continuity_weight': params.pattern_continuity_weight,
                    'noise_reduction': params.noise_reduction,
                    'contrast_enhancement': params.contrast_enhancement
                },
                'quality_improvements': result.quality_improvements,
                'processing_time': result.processing_time,
                'output_path': output_path
            }
        
        self.demo_results['optimization_results'] = optimization_results
        
        return optimization_results
    
    def run_parameter_optimization(self, generated_image: np.ndarray, reference_image: np.ndarray):
        """运行参数自动优化"""
        logger.info("运行参数自动优化...")
        
        # 定义优化函数
        def optimization_function(params):
            """优化函数"""
            try:
                # 创建优化参数
                opt_params = OptimizationParams(**params)
                
                # 执行优化
                result = self.embroidery_optimizer.optimize_embroidery(
                    generated_image, opt_params, target_style="realistic"
                )
                
                # 计算质量评分
                quality_score = sum(result.quality_improvements.values()) / len(result.quality_improvements)
                
                return quality_score
                
            except Exception as e:
                logger.warning(f"优化函数执行失败: {e}")
                return 0.0
        
        # 创建参数优化器
        parameter_space = ParameterSpace(
            color_clusters=(8, 24),
            edge_smoothness=(0.5, 2.5),
            texture_detail_level=(0.4, 1.1),
            pattern_continuity_weight=(0.3, 0.9),
            noise_reduction=(0.1, 0.7),
            contrast_enhancement=(0.9, 1.6)
        )
        
        self.parameter_optimizer = ParameterOptimizer(
            optimization_function=optimization_function,
            parameter_space=parameter_space,
            n_trials=30,  # 减少试验次数以加快演示
            n_jobs=2
        )
        
        # 执行优化
        best_params = self.parameter_optimizer.optimize(target_score=0.8, timeout=300)
        
        # 使用最佳参数进行最终优化
        if best_params:
            final_result = self.embroidery_optimizer.optimize_embroidery(
                generated_image, 
                OptimizationParams(**best_params),
                target_style="realistic"
            )
            
            # 保存最终结果
            final_output_path = str(self.output_dir / "auto_optimized_final.png")
            self.embroidery_optimizer.save_optimization_result(final_result, final_output_path)
            
            # 保存优化历史
            history_path = str(self.output_dir / "parameter_optimization_history.json")
            self.parameter_optimizer.save_optimization_results(history_path)
            
            # 生成可视化
            viz_path = str(self.output_dir / "parameter_optimization_visualization.png")
            self.parameter_optimizer.visualize_optimization_history(viz_path)
            
            # 记录结果
            self.demo_results['parameter_optimization'] = {
                'best_parameters': best_params,
                'best_score': self.parameter_optimizer.best_trial.quality_score if self.parameter_optimizer.best_trial else 0.0,
                'final_result': {
                    'quality_improvements': final_result.quality_improvements,
                    'processing_time': final_result.processing_time,
                    'output_path': final_output_path
                },
                'parameter_importance': self.parameter_optimizer.analyze_parameter_importance()
            }
        
        return best_params
    
    def run_comparison_analysis(self, 
                              original_image: np.ndarray,
                              reference_image: np.ndarray,
                              optimization_results: Dict[str, Any]):
        """运行对比分析"""
        logger.info("运行对比分析...")
        
        comparison_results = {}
        
        # 分析原始图像与参考图像的差异
        original_analysis = self.quality_analyzer.analyze_embroidery_quality(
            str(self.output_dir / "temp_original.png"),
            str(self.output_dir / "temp_reference.png")
        )
        
        # 为每个优化结果创建临时文件并分析
        for config_name, result_data in optimization_results.items():
            # 加载优化后的图像
            optimized_image = cv2.imread(result_data['output_path'])
            
            # 保存临时文件
            temp_optimized_path = str(self.output_dir / f"temp_optimized_{config_name}.png")
            cv2.imwrite(temp_optimized_path, optimized_image)
            
            # 分析优化后的图像
            optimized_analysis = self.quality_analyzer.analyze_embroidery_quality(
                temp_optimized_path,
                str(self.output_dir / "temp_reference.png")
            )
            
            # 计算改进程度
            improvement = {
                'overall_score_improvement': optimized_analysis.quality_metrics.overall_score - original_analysis.quality_metrics.overall_score,
                'color_accuracy_improvement': optimized_analysis.quality_metrics.color_accuracy - original_analysis.quality_metrics.color_accuracy,
                'edge_quality_improvement': optimized_analysis.quality_metrics.edge_quality - original_analysis.quality_metrics.edge_quality,
                'texture_detail_improvement': optimized_analysis.quality_metrics.texture_detail - original_analysis.quality_metrics.texture_detail,
                'pattern_continuity_improvement': optimized_analysis.quality_metrics.pattern_continuity - original_analysis.quality_metrics.pattern_continuity
            }
            
            comparison_results[config_name] = {
                'original_score': original_analysis.quality_metrics.overall_score,
                'optimized_score': optimized_analysis.quality_metrics.overall_score,
                'improvements': improvement,
                'recommendations': optimized_analysis.quality_metrics.recommendations
            }
            
            # 清理临时文件
            os.remove(temp_optimized_path)
        
        self.demo_results['comparison_results'] = comparison_results
        
        return comparison_results
    
    def generate_comprehensive_report(self):
        """生成综合报告"""
        logger.info("生成综合报告...")
        
        report = {
            'demo_summary': {
                'timestamp': time.time(),
                'output_directory': str(self.output_dir),
                'components_tested': [
                    'Quality Analysis',
                    'Embroidery Optimization', 
                    'Parameter Optimization',
                    'Comparison Analysis'
                ]
            },
            'results': self.demo_results
        }
        
        # 保存综合报告
        report_path = str(self.output_dir / "comprehensive_demo_report.json")
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        # 生成可视化总结
        self._generate_summary_visualization()
        
        logger.info(f"综合报告已保存: {report_path}")
        
        return report
    
    def _generate_summary_visualization(self):
        """生成总结可视化"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. 质量分析结果
        if self.demo_results['quality_analysis']:
            metrics = self.demo_results['quality_analysis']['detailed_metrics']
            metric_names = list(metrics.keys())
            metric_values = list(metrics.values())
            
            axes[0, 0].bar(metric_names, metric_values, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
            axes[0, 0].set_title('质量分析结果')
            axes[0, 0].set_ylabel('评分')
            axes[0, 0].set_ylim(0, 1)
            axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. 优化结果对比
        if self.demo_results['optimization_results']:
            config_names = list(self.demo_results['optimization_results'].keys())
            avg_improvements = []
            
            for config_name in config_names:
                improvements = self.demo_results['optimization_results'][config_name]['quality_improvements']
                avg_improvement = sum(improvements.values()) / len(improvements)
                avg_improvements.append(avg_improvement)
            
            axes[0, 1].bar(config_names, avg_improvements, color=['#FFEAA7', '#DDA0DD', '#98D8C8', '#F7DC6F'])
            axes[0, 1].set_title('各配置优化效果对比')
            axes[0, 1].set_ylabel('平均改进')
            axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 3. 参数重要性
        if self.demo_results['parameter_optimization'] and 'parameter_importance' in self.demo_results['parameter_optimization']:
            importance = self.demo_results['parameter_optimization']['parameter_importance']
            if importance:
                param_names = list(importance.keys())
                importance_values = list(importance.values())
                
                axes[1, 0].bar(param_names, importance_values, color='#FF9999')
                axes[1, 0].set_title('参数重要性分析')
                axes[1, 0].set_ylabel('重要性')
                axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 4. 总体改进对比
        if self.demo_results['comparison_results']:
            config_names = list(self.demo_results['comparison_results'].keys())
            overall_improvements = []
            
            for config_name in config_names:
                improvement = self.demo_results['comparison_results'][config_name]['overall_score_improvement']
                overall_improvements.append(improvement)
            
            colors = ['green' if x > 0 else 'red' for x in overall_improvements]
            axes[1, 1].bar(config_names, overall_improvements, color=colors)
            axes[1, 1].set_title('总体质量改进对比')
            axes[1, 1].set_ylabel('改进幅度')
            axes[1, 1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
            axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        # 保存可视化
        viz_path = str(self.output_dir / "comprehensive_summary_visualization.png")
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        logger.info(f"总结可视化已保存: {viz_path}")
    
    def run_full_demo(self):
        """运行完整演示"""
        logger.info("开始综合优化演示...")
        
        try:
            # 1. 创建测试图像
            generated_image, reference_image = self.create_test_images()
            
            # 2. 运行质量分析
            quality_analysis = self.run_quality_analysis(generated_image, reference_image)
            
            # 3. 运行优化演示
            optimization_results = self.run_optimization_demo(generated_image)
            
            # 4. 运行参数自动优化
            best_params = self.run_parameter_optimization(generated_image, reference_image)
            
            # 5. 运行对比分析
            comparison_results = self.run_comparison_analysis(
                generated_image, reference_image, optimization_results
            )
            
            # 6. 生成综合报告
            comprehensive_report = self.generate_comprehensive_report()
            
            logger.info("综合优化演示完成！")
            
            return {
                'success': True,
                'quality_analysis': quality_analysis,
                'optimization_results': optimization_results,
                'best_params': best_params,
                'comparison_results': comparison_results,
                'comprehensive_report': comprehensive_report
            }
            
        except Exception as e:
            logger.error(f"演示过程中出现错误: {e}")
            return {
                'success': False,
                'error': str(e)
            }


if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("综合优化演示系统")
    print("=" * 60)
    
    # 创建演示系统
    demo = ComprehensiveOptimizationDemo()
    
    # 运行完整演示
    results = demo.run_full_demo()
    
    if results['success']:
        print("\n✅ 演示成功完成！")
        print(f"📁 结果保存在: {demo.output_dir}")
        print(f"📊 质量分析评分: {results['quality_analysis'].quality_metrics.overall_score:.3f}")
        print(f"🔧 最佳参数: {results['best_params']}")
        
        # 显示优化结果摘要
        if results['optimization_results']:
            print("\n📈 优化效果摘要:")
            for config_name, result_data in results['optimization_results'].items():
                avg_improvement = sum(result_data['quality_improvements'].values()) / len(result_data['quality_improvements'])
                print(f"  {config_name}: {avg_improvement:+.3f}")
        
        print("\n🎉 所有优化组件已成功集成并测试！")
    else:
        print(f"\n❌ 演示失败: {results['error']}")
    
    print("\n演示完成！") 