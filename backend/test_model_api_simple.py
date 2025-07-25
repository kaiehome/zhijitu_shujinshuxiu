"""
简化模型API测试
测试可用的模型API组件，避免深度学习依赖问题
"""

import asyncio
import cv2
import numpy as np
import time
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple

# 只导入可用的组件
try:
    from simple_model_api import SimpleModelAPIManager, SimpleModelAPIConfig, SimpleGenerationRequest
    SIMPLE_API_AVAILABLE = True
except ImportError as e:
    print(f"简化模型API不可用: {e}")
    SIMPLE_API_AVAILABLE = False

try:
    from ai_enhanced_processor import AIEnhancedProcessor
    AI_ENHANCED_AVAILABLE = True
except ImportError as e:
    print(f"AI增强处理器不可用: {e}")
    AI_ENHANCED_AVAILABLE = False

try:
    from ai_image_generator import AIImageGenerator
    AI_GENERATOR_AVAILABLE = True
except ImportError as e:
    print(f"AI图像生成器不可用: {e}")
    AI_GENERATOR_AVAILABLE = False

try:
    from ai_segmentation import AISegmenter
    AI_SEGMENTATION_AVAILABLE = True
except ImportError as e:
    print(f"AI分割器不可用: {e}")
    AI_SEGMENTATION_AVAILABLE = False

try:
    from local_ai_generator import LocalAIGenerator
    LOCAL_AI_AVAILABLE = True
except ImportError as e:
    print(f"本地AI生成器不可用: {e}")
    LOCAL_AI_AVAILABLE = False

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimpleModelAPITest:
    """简化模型API测试类"""

    def __init__(self):
        self.test_results = {}
        self.test_images = {}
        self.output_dir = Path("test_outputs")
        self.output_dir.mkdir(exist_ok=True)

    def create_test_images(self) -> Dict[str, np.ndarray]:
        """创建测试图像"""
        logger.info("创建测试图像...")
        
        images = {}
        
        # 1. 简单测试图像
        simple_image = np.ones((512, 512, 3), dtype=np.uint8) * 240
        cv2.rectangle(simple_image, (100, 100), (412, 412), (100, 100, 100), 2)
        cv2.circle(simple_image, (256, 256), 80, (150, 150, 150), -1)
        images["simple"] = simple_image
        
        # 2. 熊猫图案（模拟真实刺绣）
        panda_image = np.ones((512, 512, 3), dtype=np.uint8) * 240
        # 熊猫头部
        cv2.ellipse(panda_image, (256, 256), (120, 100), 0, 0, 360, (240, 240, 240), -1)
        # 眼睛
        cv2.circle(panda_image, (220, 230), 15, (50, 50, 50), -1)
        cv2.circle(panda_image, (292, 230), 15, (50, 50, 50), -1)
        # 鼻子
        cv2.circle(panda_image, (256, 270), 8, (50, 50, 50), -1)
        images["panda"] = panda_image
        
        # 保存测试图像
        for name, image in images.items():
            cv2.imwrite(str(self.output_dir / f"test_{name}.png"), image)
        
        self.test_images = images
        logger.info(f"创建了 {len(images)} 个测试图像")
        return images

    async def test_simple_model_api(self) -> Dict[str, Any]:
        """测试简化模型API"""
        if not SIMPLE_API_AVAILABLE:
            return {"status": "skipped", "reason": "组件不可用"}
        
        logger.info("🧪 测试简化模型API...")
        
        results = {
            "status": "passed",
            "tests": {},
            "performance": {},
            "errors": []
        }
        
        try:
            # 初始化API管理器
            config = SimpleModelAPIConfig(
                enable_optimization=True,
                enable_quality_analysis=True,
                default_style="basic"
            )
            api_manager = SimpleModelAPIManager(config)
            
            # 测试1: 检查可用模型
            available_models = api_manager.get_available_models()
            results["tests"]["available_models"] = {
                "status": "passed",
                "data": available_models
            }
            
            # 测试2: 基础图像处理
            test_image_path = str(self.output_dir / "test_simple.png")
            request = SimpleGenerationRequest(
                image_path=test_image_path,
                style="basic",
                color_count=12,
                optimization_level="balanced"
            )
            
            start_time = time.time()
            job_id = await api_manager.generate_embroidery(request)
            
            # 等待任务完成
            max_wait = 30
            while max_wait > 0:
                status = api_manager.get_job_status(job_id)
                if status and status['status'] in ['completed', 'failed']:
                    break
                await asyncio.sleep(1)
                max_wait -= 1
            
            if status and status['status'] == 'completed':
                results["tests"]["basic_processing"] = {
                    "status": "passed",
                    "processing_time": status['processing_time'],
                    "output_files": status['output_files']
                }
            else:
                results["tests"]["basic_processing"] = {
                    "status": "failed",
                    "error": status.get('error_message', 'Unknown error') if status else 'Timeout'
                }
                results["status"] = "failed"
            
            # 测试3: 不同风格处理
            styles = ["basic", "embroidery", "traditional", "modern"]
            style_results = {}
            
            for style in styles:
                request.style = style
                job_id = await api_manager.generate_embroidery(request)
                
                # 等待完成
                max_wait = 20
                while max_wait > 0:
                    status = api_manager.get_job_status(job_id)
                    if status and status['status'] in ['completed', 'failed']:
                        break
                    await asyncio.sleep(1)
                    max_wait -= 1
                
                style_results[style] = {
                    "status": status['status'] if status else "timeout",
                    "processing_time": status.get('processing_time', 0) if status else 0
                }
            
            results["tests"]["style_processing"] = {
                "status": "passed" if all(r["status"] == "completed" for r in style_results.values()) else "failed",
                "data": style_results
            }
            
            # 测试4: 系统统计
            stats = api_manager.get_system_stats()
            results["tests"]["system_stats"] = {
                "status": "passed",
                "data": stats
            }
            
            # 性能统计
            results["performance"] = {
                "total_tests": len(results["tests"]),
                "passed_tests": sum(1 for t in results["tests"].values() if t["status"] == "passed"),
                "failed_tests": sum(1 for t in results["tests"].values() if t["status"] == "failed")
            }
            
        except Exception as e:
            logger.error(f"简化模型API测试失败: {e}")
            results["status"] = "failed"
            results["errors"].append(str(e))
        
        self.test_results["simple_api"] = results
        return results

    def test_ai_enhanced_processor(self) -> Dict[str, Any]:
        """测试AI增强处理器"""
        if not AI_ENHANCED_AVAILABLE:
            return {"status": "skipped", "reason": "组件不可用"}
        
        logger.info("🧪 测试AI增强处理器...")
        
        results = {
            "status": "passed",
            "tests": {},
            "performance": {},
            "errors": []
        }
        
        try:
            # 初始化AI增强处理器
            processor = AIEnhancedProcessor()
            
            # 测试1: 处理器初始化
            results["tests"]["initialization"] = {
                "status": "passed",
                "ai_enabled": processor.ai_enabled
            }
            
            # 测试2: 图像内容分析
            test_image = self.test_images["panda"]
            analysis = processor.analyze_image_content(test_image)
            
            results["tests"]["image_analysis"] = {
                "status": "passed",
                "analysis": analysis
            }
            
            # 测试3: 回退分析
            fallback_analysis = processor._fallback_analysis(test_image)
            results["tests"]["fallback_analysis"] = {
                "status": "passed",
                "analysis": fallback_analysis
            }
            
            # 性能统计
            results["performance"] = {
                "total_tests": len(results["tests"]),
                "passed_tests": len(results["tests"])
            }
            
        except Exception as e:
            logger.error(f"AI增强处理器测试失败: {e}")
            results["status"] = "failed"
            results["errors"].append(str(e))
        
        self.test_results["ai_enhanced"] = results
        return results

    def test_ai_image_generator(self) -> Dict[str, Any]:
        """测试AI图像生成器"""
        if not AI_GENERATOR_AVAILABLE:
            return {"status": "skipped", "reason": "组件不可用"}
        
        logger.info("🧪 测试AI图像生成器...")
        
        results = {
            "status": "passed",
            "tests": {},
            "performance": {},
            "errors": []
        }
        
        try:
            # 初始化AI图像生成器
            generator = AIImageGenerator()
            
            # 测试1: 生成器初始化
            results["tests"]["initialization"] = {
                "status": "passed",
                "ai_enabled": generator.ai_enabled
            }
            
            # 测试2: 专业背景生成
            background = generator.generate_professional_background(512, 512, "熊猫")
            
            results["tests"]["background_generation"] = {
                "status": "passed",
                "background_shape": background.shape
            }
            
            # 保存生成的背景
            cv2.imwrite(str(self.output_dir / "generated_background.png"), background)
            
            # 测试3: 基础背景生成
            basic_background = generator._generate_basic_background(256, 256)
            results["tests"]["basic_background"] = {
                "status": "passed",
                "background_shape": basic_background.shape
            }
            
            # 性能统计
            results["performance"] = {
                "total_tests": len(results["tests"]),
                "passed_tests": len(results["tests"])
            }
            
        except Exception as e:
            logger.error(f"AI图像生成器测试失败: {e}")
            results["status"] = "failed"
            results["errors"].append(str(e))
        
        self.test_results["ai_generation"] = results
        return results

    def test_ai_segmentation(self) -> Dict[str, Any]:
        """测试AI分割器"""
        if not AI_SEGMENTATION_AVAILABLE:
            return {"status": "skipped", "reason": "组件不可用"}
        
        logger.info("🧪 测试AI分割器...")
        
        results = {
            "status": "passed",
            "tests": {},
            "performance": {},
            "errors": []
        }
        
        try:
            # 测试不同的分割方法
            segmentation_methods = ['grabcut', 'watershed', 'slic', 'contour']
            
            for method in segmentation_methods:
                try:
                    # 初始化分割器
                    segmenter = AISegmenter(model_name=method)
                    
                    # 执行分割
                    test_image = self.test_images["panda"]
                    segmentation_result = segmenter.segment(test_image)
                    
                    results["tests"][f"{method}_segmentation"] = {
                        "status": "passed",
                        "result_shape": segmentation_result.shape,
                        "unique_values": len(np.unique(segmentation_result))
                    }
                    
                    # 保存分割结果
                    cv2.imwrite(str(self.output_dir / f"segmentation_{method}.png"), segmentation_result)
                    
                except Exception as e:
                    results["tests"][f"{method}_segmentation"] = {
                        "status": "failed",
                        "error": str(e)
                    }
            
            # 性能统计
            passed_tests = sum(1 for t in results["tests"].values() if t["status"] == "passed")
            results["performance"] = {
                "total_tests": len(results["tests"]),
                "passed_tests": passed_tests,
                "failed_tests": len(results["tests"]) - passed_tests
            }
            
            if passed_tests < len(results["tests"]):
                results["status"] = "partial"
            
        except Exception as e:
            logger.error(f"AI分割器测试失败: {e}")
            results["status"] = "failed"
            results["errors"].append(str(e))
        
        self.test_results["ai_segmentation"] = results
        return results

    def test_local_ai_generator(self) -> Dict[str, Any]:
        """测试本地AI生成器"""
        if not LOCAL_AI_AVAILABLE:
            return {"status": "skipped", "reason": "组件不可用"}
        
        logger.info("🧪 测试本地AI生成器...")
        
        results = {
            "status": "passed",
            "tests": {},
            "performance": {},
            "errors": []
        }
        
        try:
            # 初始化本地AI生成器
            generator = LocalAIGenerator()
            
            # 测试1: 生成器初始化
            results["tests"]["initialization"] = {
                "status": "passed",
                "models_loaded": generator.models_loaded,
                "device": generator.device
            }
            
            # 测试2: 图像理解
            test_image = self.test_images["panda"]
            understanding = generator.understand_image(test_image)
            
            results["tests"]["image_understanding"] = {
                "status": "passed",
                "understanding": understanding
            }
            
            # 测试3: 风格转换
            style_result = generator.apply_style(test_image, "embroidery")
            
            results["tests"]["style_transfer"] = {
                "status": "passed",
                "result_shape": style_result.shape
            }
            
            # 保存结果
            cv2.imwrite(str(self.output_dir / "local_ai_style.png"), style_result)
            
            # 性能统计
            results["performance"] = {
                "total_tests": len(results["tests"]),
                "passed_tests": len(results["tests"])
            }
            
        except Exception as e:
            logger.error(f"本地AI生成器测试失败: {e}")
            results["status"] = "failed"
            results["errors"].append(str(e))
        
        self.test_results["local_ai"] = results
        return results

    def generate_test_report(self) -> str:
        """生成测试报告"""
        logger.info("📊 生成测试报告...")
        
        report = []
        report.append("# 🤖 简化模型API测试报告")
        report.append("")
        report.append(f"**测试时间**: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # 组件可用性
        report.append("## 📋 组件可用性")
        report.append("")
        components = [
            ("简化模型API", SIMPLE_API_AVAILABLE),
            ("AI增强处理器", AI_ENHANCED_AVAILABLE),
            ("AI图像生成器", AI_GENERATOR_AVAILABLE),
            ("AI分割器", AI_SEGMENTATION_AVAILABLE),
            ("本地AI生成器", LOCAL_AI_AVAILABLE)
        ]
        
        for name, available in components:
            status = "✅ 可用" if available else "❌ 不可用"
            report.append(f"- **{name}**: {status}")
        report.append("")
        
        # 总体统计
        total_tests = 0
        total_passed = 0
        total_failed = 0
        total_skipped = 0
        
        for component, results in self.test_results.items():
            if not results:
                continue
                
            total_tests += 1
            if results["status"] == "passed":
                total_passed += 1
            elif results["status"] == "failed":
                total_failed += 1
            elif results["status"] == "skipped":
                total_skipped += 1
        
        report.append("## 📈 总体测试结果")
        report.append("")
        report.append(f"- **总组件数**: {total_tests}")
        report.append(f"- **通过**: {total_passed} ✅")
        report.append(f"- **失败**: {total_failed} ❌")
        report.append(f"- **跳过**: {total_skipped} ⏭️")
        report.append("")
        
        # 详细结果
        report.append("## 🔍 详细测试结果")
        report.append("")
        
        for component, results in self.test_results.items():
            if not results:
                continue
                
            status_emoji = "✅" if results["status"] == "passed" else "❌" if results["status"] == "failed" else "⏭️"
            report.append(f"### {status_emoji} {component.replace('_', ' ').title()}")
            report.append("")
            
            if "reason" in results:
                report.append(f"**跳过原因**: {results['reason']}")
                report.append("")
                continue
            
            if "tests" in results:
                for test_name, test_result in results["tests"].items():
                    test_status = "✅" if test_result["status"] == "passed" else "❌" if test_result["status"] == "failed" else "⏭️"
                    report.append(f"- {test_status} **{test_name}**: {test_result['status']}")
                    
                    if "error" in test_result:
                        report.append(f"  - 错误: {test_result['error']}")
            
            if "performance" in results:
                perf = results["performance"]
                report.append(f"- 📊 **性能**: {perf.get('passed_tests', 0)}/{perf.get('total_tests', 0)} 通过")
            
            if "errors" in results and results["errors"]:
                report.append("**错误列表**:")
                for error in results["errors"]:
                    report.append(f"- {error}")
            
            report.append("")
        
        # 建议
        report.append("## 💡 建议")
        report.append("")
        
        if total_failed > 0:
            report.append("### 🔧 需要修复的问题")
            report.append("")
            for component, results in self.test_results.items():
                if results and results["status"] == "failed":
                    report.append(f"- **{component}**: 检查错误日志并修复相关问题")
            report.append("")
        
        if total_passed > 0:
            report.append("### ✅ 运行良好的组件")
            report.append("")
            for component, results in self.test_results.items():
                if results and results["status"] == "passed":
                    report.append(f"- **{component}**: 功能正常，可以投入使用")
            report.append("")
        
        report.append("### 🚀 下一步行动")
        report.append("")
        report.append("1. **修复失败组件**: 根据错误信息修复相关问题")
        report.append("2. **安装缺失依赖**: 安装PyTorch、TensorFlow等深度学习库")
        report.append("3. **性能优化**: 对通过测试的组件进行性能优化")
        report.append("4. **生产部署**: 将测试通过的组件部署到生产环境")
        
        report_content = "\n".join(report)
        
        # 保存报告
        report_path = self.output_dir / "simple_model_api_test_report.md"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        logger.info(f"测试报告已保存: {report_path}")
        return report_content

    async def run_all_tests(self) -> Dict[str, Any]:
        """运行所有测试"""
        logger.info("🚀 开始运行简化模型API测试...")
        
        # 创建测试图像
        self.create_test_images()
        
        # 运行各个组件测试
        await self.test_simple_model_api()
        self.test_ai_enhanced_processor()
        self.test_ai_image_generator()
        self.test_ai_segmentation()
        self.test_local_ai_generator()
        
        # 生成测试报告
        report = self.generate_test_report()
        
        logger.info("🎉 简化模型API测试完成！")
        return self.test_results


async def main():
    """主函数"""
    print("🤖 简化模型API测试")
    print("=" * 50)
    
    # 创建测试实例
    tester = SimpleModelAPITest()
    
    # 运行所有测试
    results = await tester.run_all_tests()
    
    # 打印简要结果
    print("\n📊 测试结果摘要:")
    for component, result in results.items():
        if result:
            status = result["status"]
            emoji = "✅" if status == "passed" else "❌" if status == "failed" else "⏭️"
            print(f"{emoji} {component}: {status}")
    
    print(f"\n📄 详细报告已保存到: {tester.output_dir}/simple_model_api_test_report.md")


if __name__ == "__main__":
    asyncio.run(main()) 