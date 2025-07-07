#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
蜀锦蜀绣AI打样图生成工具 - 系统测试脚本
提供完整的API接口测试、图像处理测试和性能测试功能
"""

import os
import sys
import time
import json
import asyncio
import aiohttp
import aiofiles
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import logging
from datetime import datetime
import argparse
from dataclasses import dataclass
import tempfile
import hashlib

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('test_results.log')
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class TestConfig:
    """测试配置"""
    backend_url: str = "http://localhost:8000"
    frontend_url: str = "http://localhost:3000"
    test_images_dir: str = "test_images"
    timeout: int = 300
    max_concurrent: int = 5
    
@dataclass
class TestResult:
    """测试结果"""
    name: str
    success: bool
    duration: float
    message: str
    details: Optional[Dict[str, Any]] = None

class SystemTester:
    """系统测试器"""
    
    def __init__(self, config: TestConfig):
        self.config = config
        self.session: Optional[aiohttp.ClientSession] = None
        self.test_results: List[TestResult] = []
        
    async def __aenter__(self):
        """异步上下文管理器入口"""
        connector = aiohttp.TCPConnector(limit=self.config.max_concurrent)
        timeout = aiohttp.ClientTimeout(total=self.config.timeout)
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers={'User-Agent': 'SichuanBrocade-SystemTester/1.0'}
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器出口"""
        if self.session:
            await self.session.close()
    
    def log_test_result(self, result: TestResult):
        """记录测试结果"""
        self.test_results.append(result)
        status = "✅ PASS" if result.success else "❌ FAIL"
        logger.info(f"{status} {result.name} ({result.duration:.2f}s): {result.message}")
        
        if result.details:
            for key, value in result.details.items():
                logger.info(f"  {key}: {value}")
    
    async def test_backend_health(self) -> TestResult:
        """测试后端健康检查"""
        start_time = time.time()
        
        try:
            async with self.session.get(f"{self.config.backend_url}/api/health") as response:
                if response.status == 200:
                    data = await response.json()
                    duration = time.time() - start_time
                    
                    return TestResult(
                        name="Backend Health Check",
                        success=True,
                        duration=duration,
                        message="Backend is healthy",
                        details=data
                    )
                else:
                    duration = time.time() - start_time
                    return TestResult(
                        name="Backend Health Check",
                        success=False,
                        duration=duration,
                        message=f"Health check failed with status {response.status}"
                    )
                    
        except Exception as e:
            duration = time.time() - start_time
            return TestResult(
                name="Backend Health Check",
                success=False,
                duration=duration,
                message=f"Health check error: {str(e)}"
            )
    
    async def test_frontend_availability(self) -> TestResult:
        """测试前端可用性"""
        start_time = time.time()
        
        try:
            async with self.session.get(self.config.frontend_url) as response:
                if response.status == 200:
                    content = await response.text()
                    duration = time.time() - start_time
                    
                    # 检查关键内容
                    if "蜀锦蜀绣" in content and "AI" in content:
                        return TestResult(
                            name="Frontend Availability",
                            success=True,
                            duration=duration,
                            message="Frontend is accessible",
                            details={"content_length": len(content)}
                        )
                    else:
                        return TestResult(
                            name="Frontend Availability",
                            success=False,
                            duration=duration,
                            message="Frontend content validation failed"
                        )
                else:
                    duration = time.time() - start_time
                    return TestResult(
                        name="Frontend Availability",
                        success=False,
                        duration=duration,
                        message=f"Frontend returned status {response.status}"
                    )
                    
        except Exception as e:
            duration = time.time() - start_time
            return TestResult(
                name="Frontend Availability",
                success=False,
                duration=duration,
                message=f"Frontend access error: {str(e)}"
            )
    
    async def create_test_image(self, size: Tuple[int, int] = (512, 512)) -> Path:
        """创建测试图像"""
        try:
            from PIL import Image, ImageDraw
            import numpy as np
            
            # 创建测试图像
            image = Image.new('RGB', size, color='white')
            draw = ImageDraw.Draw(image)
            
            # 绘制简单的测试图案
            width, height = size
            colors = ['red', 'blue', 'green', 'yellow', 'purple']
            
            for i, color in enumerate(colors):
                x = (i * width // len(colors))
                y = (i * height // len(colors))
                draw.rectangle([x, y, x + width//len(colors), y + height//len(colors)], fill=color)
            
            # 保存到临时文件
            temp_dir = Path(tempfile.gettempdir()) / "sichuan_brocade_test"
            temp_dir.mkdir(exist_ok=True)
            
            test_image_path = temp_dir / f"test_image_{int(time.time())}.png"
            image.save(test_image_path, 'PNG')
            
            return test_image_path
            
        except ImportError:
            # 如果PIL不可用，创建简单的文本文件作为替代
            logger.warning("PIL not available, creating dummy test file")
            temp_dir = Path(tempfile.gettempdir()) / "sichuan_brocade_test"
            temp_dir.mkdir(exist_ok=True)
            
            test_file_path = temp_dir / f"test_dummy_{int(time.time())}.txt"
            with open(test_file_path, 'w') as f:
                f.write("This is a test file for system testing")
            
            return test_file_path
    
    async def test_image_upload(self) -> TestResult:
        """测试图像上传功能"""
        start_time = time.time()
        
        try:
            # 创建测试图像
            test_image_path = await self.create_test_image()
            
            # 准备上传数据
            data = aiohttp.FormData()
            data.add_field('file', 
                          open(test_image_path, 'rb'),
                          filename=test_image_path.name,
                          content_type='image/png')
            
            # 上传文件
            async with self.session.post(
                f"{self.config.backend_url}/api/upload",
                data=data
            ) as response:
                
                duration = time.time() - start_time
                
                if response.status == 200:
                    result_data = await response.json()
                    
                    # 清理测试文件
                    test_image_path.unlink(missing_ok=True)
                    
                    return TestResult(
                        name="Image Upload",
                        success=True,
                        duration=duration,
                        message="Image uploaded successfully",
                        details=result_data
                    )
                else:
                    error_text = await response.text()
                    
                    # 清理测试文件
                    test_image_path.unlink(missing_ok=True)
                    
                    return TestResult(
                        name="Image Upload",
                        success=False,
                        duration=duration,
                        message=f"Upload failed with status {response.status}: {error_text}"
                    )
                    
        except Exception as e:
            duration = time.time() - start_time
            return TestResult(
                name="Image Upload",
                success=False,
                duration=duration,
                message=f"Upload error: {str(e)}"
            )
    
    async def test_image_processing(self) -> TestResult:
        """测试图像处理功能"""
        start_time = time.time()
        
        try:
            # 首先上传图像
            test_image_path = await self.create_test_image()
            
            # 上传文件
            data = aiohttp.FormData()
            data.add_field('file', 
                          open(test_image_path, 'rb'),
                          filename=test_image_path.name,
                          content_type='image/png')
            
            async with self.session.post(
                f"{self.config.backend_url}/api/upload",
                data=data
            ) as upload_response:
                
                if upload_response.status != 200:
                    duration = time.time() - start_time
                    return TestResult(
                        name="Image Processing",
                        success=False,
                        duration=duration,
                        message="Failed to upload test image for processing"
                    )
                
                upload_result = await upload_response.json()
                filename = upload_result.get('filename')
                
                if not filename:
                    duration = time.time() - start_time
                    return TestResult(
                        name="Image Processing",
                        success=False,
                        duration=duration,
                        message="No filename returned from upload"
                    )
                
                # 处理图像
                process_data = {
                    "filename": filename,
                                         "color_count": 16,
                    "edge_enhancement": True,
                    "noise_reduction": True,
                    "style": "sichuan_brocade"
                }
                
                async with self.session.post(
                    f"{self.config.backend_url}/api/process",
                    json=process_data
                ) as process_response:
                    
                    if process_response.status == 200:
                        process_result = await process_response.json()
                        job_id = process_result.get('job_id')
                        
                        if not job_id:
                            duration = time.time() - start_time
                            return TestResult(
                                name="Image Processing",
                                success=False,
                                duration=duration,
                                message="No job_id returned from process request"
                            )
                        
                        # 轮询处理状态
                        max_wait_time = 60  # 最多等待60秒
                        poll_interval = 2   # 每2秒查询一次
                        
                        for _ in range(max_wait_time // poll_interval):
                            async with self.session.get(
                                f"{self.config.backend_url}/api/status/{job_id}"
                            ) as status_response:
                                
                                if status_response.status == 200:
                                    status_data = await status_response.json()
                                    status = status_data.get('status')
                                    
                                    if status == 'completed':
                                        duration = time.time() - start_time
                                        
                                        # 清理测试文件
                                        test_image_path.unlink(missing_ok=True)
                                        
                                        return TestResult(
                                            name="Image Processing",
                                            success=True,
                                            duration=duration,
                                            message="Image processing completed successfully",
                                            details=status_data
                                        )
                                    elif status == 'failed':
                                        duration = time.time() - start_time
                                        
                                        # 清理测试文件
                                        test_image_path.unlink(missing_ok=True)
                                        
                                        return TestResult(
                                            name="Image Processing",
                                            success=False,
                                            duration=duration,
                                            message=f"Image processing failed: {status_data.get('message', 'Unknown error')}"
                                        )
                                    
                                    # 继续等待
                                    await asyncio.sleep(poll_interval)
                                else:
                                    break
                        
                        # 超时
                        duration = time.time() - start_time
                        
                        # 清理测试文件
                        test_image_path.unlink(missing_ok=True)
                        
                        return TestResult(
                            name="Image Processing",
                            success=False,
                            duration=duration,
                            message="Image processing timeout"
                        )
                    else:
                        error_text = await process_response.text()
                        duration = time.time() - start_time
                        
                        # 清理测试文件
                        test_image_path.unlink(missing_ok=True)
                        
                        return TestResult(
                            name="Image Processing",
                            success=False,
                            duration=duration,
                            message=f"Process request failed with status {process_response.status}: {error_text}"
                        )
                        
        except Exception as e:
            duration = time.time() - start_time
            return TestResult(
                name="Image Processing",
                success=False,
                duration=duration,
                message=f"Processing error: {str(e)}"
            )
    
    async def test_api_endpoints(self) -> List[TestResult]:
        """测试所有API端点"""
        results = []
        
        # 测试健康检查
        result = await self.test_backend_health()
        results.append(result)
        self.log_test_result(result)
        
        # 测试上传功能
        result = await self.test_image_upload()
        results.append(result)
        self.log_test_result(result)
        
        # 测试处理功能
        result = await self.test_image_processing()
        results.append(result)
        self.log_test_result(result)
        
        return results
    
    async def test_frontend_endpoints(self) -> List[TestResult]:
        """测试前端端点"""
        results = []
        
        # 测试前端可用性
        result = await self.test_frontend_availability()
        results.append(result)
        self.log_test_result(result)
        
        return results
    
    async def run_performance_test(self, concurrent_requests: int = 3) -> TestResult:
        """运行性能测试"""
        start_time = time.time()
        
        try:
            logger.info(f"开始性能测试，并发请求数: {concurrent_requests}")
            
            # 创建多个并发上传任务
            tasks = []
            for i in range(concurrent_requests):
                task = self.test_image_upload()
                tasks.append(task)
            
            # 执行并发测试
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 统计结果
            successful_requests = 0
            failed_requests = 0
            total_duration = 0
            
            for result in results:
                if isinstance(result, TestResult):
                    if result.success:
                        successful_requests += 1
                    else:
                        failed_requests += 1
                    total_duration += result.duration
                else:
                    failed_requests += 1
            
            duration = time.time() - start_time
            avg_response_time = total_duration / len(results) if results else 0
            
            success_rate = successful_requests / len(results) * 100 if results else 0
            
            return TestResult(
                name="Performance Test",
                success=success_rate >= 80,  # 80%成功率认为通过
                duration=duration,
                message=f"Performance test completed",
                details={
                    "concurrent_requests": concurrent_requests,
                    "successful_requests": successful_requests,
                    "failed_requests": failed_requests,
                    "success_rate": f"{success_rate:.1f}%",
                    "avg_response_time": f"{avg_response_time:.2f}s",
                    "total_duration": f"{duration:.2f}s"
                }
            )
            
        except Exception as e:
            duration = time.time() - start_time
            return TestResult(
                name="Performance Test",
                success=False,
                duration=duration,
                message=f"Performance test error: {str(e)}"
            )
    
    async def run_all_tests(self, include_performance: bool = True) -> Dict[str, Any]:
        """运行所有测试"""
        logger.info("🧵 开始系统测试...")
        
        all_results = []
        
        # API测试
        logger.info("📡 测试后端API...")
        api_results = await self.test_api_endpoints()
        all_results.extend(api_results)
        
        # 前端测试
        logger.info("🌐 测试前端...")
        frontend_results = await self.test_frontend_endpoints()
        all_results.extend(frontend_results)
        
        # 性能测试
        if include_performance:
            logger.info("⚡ 运行性能测试...")
            perf_result = await self.run_performance_test()
            all_results.append(perf_result)
            self.log_test_result(perf_result)
        
        # 统计结果
        total_tests = len(all_results)
        passed_tests = sum(1 for r in all_results if r.success)
        failed_tests = total_tests - passed_tests
        
        success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        
        summary = {
            "timestamp": datetime.now().isoformat(),
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": failed_tests,
            "success_rate": success_rate,
            "results": [
                {
                    "name": r.name,
                    "success": r.success,
                    "duration": r.duration,
                    "message": r.message,
                    "details": r.details
                }
                for r in all_results
            ]
        }
        
        return summary

def print_test_summary(summary: Dict[str, Any]):
    """打印测试摘要"""
    print("\n" + "="*60)
    print("🧵 蜀锦蜀绣AI打样图生成工具 - 系统测试报告")
    print("="*60)
    
    print(f"📊 测试统计:")
    print(f"   总测试数: {summary['total_tests']}")
    print(f"   通过测试: {summary['passed_tests']}")
    print(f"   失败测试: {summary['failed_tests']}")
    print(f"   成功率: {summary['success_rate']:.1f}%")
    
    print(f"\n📝 详细结果:")
    for result in summary['results']:
        status = "✅" if result['success'] else "❌"
        print(f"   {status} {result['name']} ({result['duration']:.2f}s)")
        if not result['success']:
            print(f"      错误: {result['message']}")
    
    print(f"\n🕐 测试时间: {summary['timestamp']}")
    
    if summary['success_rate'] >= 80:
        print("🎉 系统测试通过！")
    else:
        print("⚠️  系统测试存在问题，请检查失败的测试项。")
    
    print("="*60)

async def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="蜀锦蜀绣AI打样图生成工具系统测试")
    parser.add_argument("--backend-url", default="http://localhost:8000", help="后端服务地址")
    parser.add_argument("--frontend-url", default="http://localhost:3000", help="前端服务地址")
    parser.add_argument("--no-performance", action="store_true", help="跳过性能测试")
    parser.add_argument("--timeout", type=int, default=300, help="请求超时时间（秒）")
    parser.add_argument("--output", help="输出结果到JSON文件")
    
    args = parser.parse_args()
    
    # 创建测试配置
    config = TestConfig(
        backend_url=args.backend_url,
        frontend_url=args.frontend_url,
        timeout=args.timeout
    )
    
    # 运行测试
    async with SystemTester(config) as tester:
        summary = await tester.run_all_tests(include_performance=not args.no_performance)
    
    # 打印结果
    print_test_summary(summary)
    
    # 保存结果
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        logger.info(f"测试结果已保存到: {args.output}")
    
    # 退出码
    sys.exit(0 if summary['success_rate'] >= 80 else 1)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("测试被用户中断")
        sys.exit(1)
    except Exception as e:
        logger.error(f"测试执行失败: {str(e)}", exc_info=True)
        sys.exit(1) 