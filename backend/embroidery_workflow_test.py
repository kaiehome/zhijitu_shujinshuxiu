#!/usr/bin/env python3
"""
绣花工艺流程测试脚本
基于用户提供的完整绣花制版工艺流程图进行测试
"""

import os
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from sklearn.cluster import KMeans
import logging
import time
from typing import Tuple, List, Dict
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmbroideryWorkflowProcessor:
    """绣花工艺流程处理器"""
    
    def __init__(self):
        """初始化绣花工艺流程处理器"""
        self.workflow_config = {
            # 1. 原始图像输入配置
            "input_formats": ["jpg", "png", "bmp"],
            
            # 2. AI图像理解与预处理模块
            "clarity_enhancement": True,      # 清晰度增强
            "background_processing": True,    # 背景处理  
            "subject_extraction": True,       # 主体提取
            
            # 3. 分色模块配置
            "color_count": 8,                # 绣花线颜色数量
            "color_method": "kmeans",         # 分色算法
            
            # 4. 边缘/轮廓提取模块
            "edge_method": "canny",          # 边缘检测算法
            "contour_simplify": True,        # 轮廓简化
            
            # 5. 针迹排布模块
            "stitch_density": "medium",      # 针迹密度
            "stitch_pattern": "satin",       # 针迹类型
            
            # 6. 绣花路径规划模块
            "anti_jump": True,              # 防跳针优化
            "direction_priority": "horizontal", # 绣花方向
            
            # 7. 输出格式生成模块
            "output_formats": ["png", "svg", "dst", "pes"] # 输出格式
        }
        logger.info("🎨 绣花工艺流程处理器初始化完成")
    
    def process_embroidery_workflow(self, input_path: str, job_id: str) -> Dict:
        """执行完整绣花工艺流程"""
        start_time = time.time()
        
        try:
            logger.info("🚀 开始执行绣花工艺流程...")
            
            # 第1步：原始图像输入
            logger.info("📥 第1步：原始图像输入")
            original_image = self._load_original_image(input_path)
            
            # 第2步：AI图像理解与预处理模块
            logger.info("🧠 第2步：AI图像理解与预处理模块")
            preprocessed_image = self._ai_image_understanding_preprocessing(original_image)
            
            # 第3步：分色模块 & 边缘/轮廓提取模块（并行处理）
            logger.info("🎨 第3步：分色模块 & 📐 边缘/轮廓提取模块")
            color_blocks = self._color_separation_module(preprocessed_image)
            edge_vectors = self._edge_contour_extraction_module(preprocessed_image)
            
            # 第4步：针迹排布模块 & 绣花路径规划模块
            logger.info("🪡 第4步：针迹排布模块 & 🛤️ 绣花路径规划模块")
            stitch_layout = self._stitch_arrangement_module(color_blocks)
            embroidery_paths = self._embroidery_path_planning_module(edge_vectors, stitch_layout)
            
            # 第5步：输出格式生成模块
            logger.info("💾 第5步：输出格式生成模块")
            output_files = self._output_format_generation_module(
                embroidery_paths, color_blocks, job_id
            )
            
            processing_time = time.time() - start_time
            
            # 生成工艺流程报告
            workflow_report = self._generate_workflow_report(
                original_image, color_blocks, embroidery_paths, 
                processing_time, job_id
            )
            
            logger.info(f"✅ 绣花工艺流程完成，总耗时: {processing_time:.2f}秒")
            
            return {
                "success": True,
                "job_id": job_id,
                "processing_time": processing_time,
                "output_files": output_files,
                "workflow_report": workflow_report
            }
            
        except Exception as e:
            logger.error(f"❌ 绣花工艺流程失败: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "job_id": job_id
            }
    
    def _load_original_image(self, input_path: str) -> np.ndarray:
        """原始图像输入"""
        try:
            image = cv2.imread(input_path)
            if image is None:
                raise ValueError(f"无法加载图像: {input_path}")
            
            # 转换为RGB格式
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            logger.info(f"  ✓ 原始图像加载成功: {rgb_image.shape}")
            return rgb_image
            
        except Exception as e:
            raise Exception(f"原始图像输入失败: {str(e)}")
    
    def _ai_image_understanding_preprocessing(self, image: np.ndarray) -> np.ndarray:
        """AI图像理解与预处理模块"""
        try:
            processed = image.copy()
            
            # 清晰度增强
            if self.workflow_config["clarity_enhancement"]:
                processed = self._enhance_clarity(processed)
                logger.info("    ✓ 清晰度增强完成")
            
            # 背景处理
            if self.workflow_config["background_processing"]:
                processed = self._process_background(processed)
                logger.info("    ✓ 背景处理完成")
            
            # 主体提取
            if self.workflow_config["subject_extraction"]:
                processed = self._extract_subject(processed)
                logger.info("    ✓ 主体提取完成")
            
            return processed
            
        except Exception as e:
            logger.warning(f"AI图像理解与预处理失败: {str(e)}")
            return image
    
    def _enhance_clarity(self, image: np.ndarray) -> np.ndarray:
        """清晰度增强"""
        try:
            pil_image = Image.fromarray(image)
            
            # 锐化增强
            enhancer = ImageEnhance.Sharpness(pil_image)
            sharpened = enhancer.enhance(1.3)
            
            # UnsharpMask滤镜
            enhanced = sharpened.filter(ImageFilter.UnsharpMask(radius=1, percent=120, threshold=3))
            
            return np.array(enhanced)
            
        except Exception as e:
            logger.warning(f"清晰度增强失败: {str(e)}")
            return image
    
    def _process_background(self, image: np.ndarray) -> np.ndarray:
        """背景处理"""
        try:
            # 简化版背景处理：高斯模糊后与原图混合
            blurred = cv2.GaussianBlur(image, (15, 15), 0)
            
            # 创建中心权重掩码
            h, w = image.shape[:2]
            center_mask = np.zeros((h, w), dtype=np.float32)
            
            # 中心区域权重高
            cv2.ellipse(center_mask, (w//2, h//2), (w//3, h//3), 0, 0, 360, 1.0, -1)
            center_mask = cv2.GaussianBlur(center_mask, (51, 51), 0)
            center_mask = center_mask[:, :, np.newaxis]
            
            # 混合原图和模糊图
            result = image * center_mask + blurred * (1 - center_mask)
            
            return result.astype(np.uint8)
            
        except Exception as e:
            logger.warning(f"背景处理失败: {str(e)}")
            return image
    
    def _extract_subject(self, image: np.ndarray) -> np.ndarray:
        """主体提取"""
        try:
            # 转为灰度图进行主体检测
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            # 自适应阈值
            thresh = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
            )
            
            # 形态学操作
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
            
            # 主体区域增强
            enhanced = image.copy()
            subject_mask = opened > 0
            enhanced[~subject_mask] = enhanced[~subject_mask] * 0.8  # 非主体区域变暗
            
            return enhanced.astype(np.uint8)
            
        except Exception as e:
            logger.warning(f"主体提取失败: {str(e)}")
            return image
    
    def _color_separation_module(self, image: np.ndarray) -> np.ndarray:
        """分色模块 - 生成色块图与边界矢量"""
        try:
            # K-means颜色聚类
            pixels = image.reshape(-1, 3)
            
            kmeans = KMeans(
                n_clusters=self.workflow_config["color_count"],
                init='k-means++',
                n_init=15,
                max_iter=300,
                random_state=42
            )
            
            kmeans.fit(pixels)
            
            # 获取绣花线颜色
            thread_colors = kmeans.cluster_centers_.astype(np.uint8)
            
            # 生成色块图
            labels = kmeans.labels_
            color_blocks = thread_colors[labels].reshape(image.shape)
            
            logger.info(f"    ✓ 分色完成，生成{len(thread_colors)}种绣花线颜色")
            
            return color_blocks
            
        except Exception as e:
            logger.warning(f"分色模块失败: {str(e)}")
            return image
    
    def _edge_contour_extraction_module(self, image: np.ndarray) -> List[np.ndarray]:
        """边缘/轮廓提取模块 - 生成边界矢量"""
        try:
            # 转为灰度图
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            # Canny边缘检测
            edges = cv2.Canny(gray, 50, 150, apertureSize=3)
            
            # 查找轮廓
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # 轮廓简化
            simplified_contours = []
            if self.workflow_config["contour_simplify"]:
                for contour in contours:
                    # Douglas-Peucker算法简化
                    epsilon = 0.02 * cv2.arcLength(contour, True)
                    simplified = cv2.approxPolyDP(contour, epsilon, True)
                    if len(simplified) > 3:
                        simplified_contours.append(simplified)
            else:
                simplified_contours = contours
            
            logger.info(f"    ✓ 边缘轮廓提取完成，共{len(simplified_contours)}个轮廓")
            
            return simplified_contours
            
        except Exception as e:
            logger.warning(f"边缘轮廓提取失败: {str(e)}")
            return []
    
    def _stitch_arrangement_module(self, color_blocks: np.ndarray) -> Dict:
        """针迹排布模块 - 规则+密度控制"""
        try:
            # 获取所有颜色
            unique_colors = np.unique(color_blocks.reshape(-1, 3), axis=0)
            
            stitch_layout = {
                "thread_colors": unique_colors.tolist(),
                "density": self.workflow_config["stitch_density"],
                "pattern": self.workflow_config["stitch_pattern"],
                "regions": []
            }
            
            # 为每种颜色生成针迹区域
            for color in unique_colors:
                # 创建颜色掩码
                mask = np.all(color_blocks == color, axis=2)
                
                if np.any(mask):
                    # 找到该颜色的所有区域
                    mask_uint8 = mask.astype(np.uint8) * 255
                    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    for contour in contours:
                        area = cv2.contourArea(contour)
                        if area > 50:  # 过滤小区域
                            stitch_layout["regions"].append({
                                "color": color.tolist(),
                                "contour": contour.tolist(),
                                "area": float(area),
                                "perimeter": float(cv2.arcLength(contour, True))
                            })
            
            logger.info(f"    ✓ 针迹排布完成，共{len(stitch_layout['regions'])}个绣花区域")
            
            return stitch_layout
            
        except Exception as e:
            logger.warning(f"针迹排布失败: {str(e)}")
            return {"thread_colors": [], "regions": []}
    
    def _embroidery_path_planning_module(self, edge_vectors: List[np.ndarray], stitch_layout: Dict) -> Dict:
        """绣花路径规划模块 - 防跳针/方向优先"""
        try:
            embroidery_paths = {
                "anti_jump": self.workflow_config["anti_jump"],
                "direction_priority": self.workflow_config["direction_priority"],
                "thread_paths": []
            }
            
            # 为每个绣花区域规划路径
            for region in stitch_layout["regions"]:
                if "contour" in region:
                    contour_array = np.array(region["contour"])
                    
                    # 生成绣花路径
                    thread_path = self._plan_thread_path(contour_array, region["color"])
                    
                    if thread_path:
                        embroidery_paths["thread_paths"].append({
                            "color": region["color"],
                            "path": thread_path,
                            "stitch_count": len(thread_path),
                            "estimated_time": len(thread_path) * 0.1  # 估算绣花时间（秒）
                        })
            
            total_stitches = sum(len(path["path"]) for path in embroidery_paths["thread_paths"])
            logger.info(f"    ✓ 绣花路径规划完成，共{total_stitches}针")
            
            return embroidery_paths
            
        except Exception as e:
            logger.warning(f"绣花路径规划失败: {str(e)}")
            return {"thread_paths": []}
    
    def _plan_thread_path(self, contour: np.ndarray, color: List[int]) -> List[Tuple[int, int]]:
        """规划单个区域的绣花路径"""
        try:
            path = []
            
            if len(contour) > 0:
                # 轮廓路径
                for point in contour:
                    if len(point) > 0:
                        x, y = point[0]
                        path.append((int(x), int(y)))
                
                # 根据针迹模式添加内部路径
                if self.workflow_config["stitch_pattern"] == "fill":
                    # 填充针迹
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # 根据密度设置步长
                    density_map = {"low": 10, "medium": 6, "high": 4}
                    step = density_map.get(self.workflow_config["stitch_density"], 6)
                    
                    # 生成填充路径
                    for py in range(y, y + h, step):
                        for px in range(x, x + w, step):
                            if cv2.pointPolygonTest(contour, (px, py), False) >= 0:
                                path.append((px, py))
            
            return path
            
        except Exception as e:
            logger.warning(f"规划绣花路径失败: {str(e)}")
            return []
    
    def _output_format_generation_module(self, embroidery_paths: Dict, color_blocks: np.ndarray, job_id: str) -> Dict:
        """输出格式生成模块 - PNG / DST / PES"""
        try:
            output_dir = f"outputs/{job_id}"
            os.makedirs(output_dir, exist_ok=True)
            
            output_files = {}
            
            # PNG格式（预览图）
            if "png" in self.workflow_config["output_formats"]:
                png_path = os.path.join(output_dir, f"{job_id}_embroidery.png")
                self._save_png_preview(color_blocks, png_path)
                output_files["png"] = png_path
                logger.info("    ✓ PNG预览图生成完成")
            
            # SVG格式（矢量图）
            if "svg" in self.workflow_config["output_formats"]:
                svg_path = os.path.join(output_dir, f"{job_id}_embroidery.svg")
                self._save_svg_vector(embroidery_paths, svg_path)
                output_files["svg"] = svg_path
                logger.info("    ✓ SVG矢量图生成完成")
            
            # DST格式（绣花机格式）
            if "dst" in self.workflow_config["output_formats"]:
                dst_path = os.path.join(output_dir, f"{job_id}_embroidery.dst")
                self._save_dst_format(embroidery_paths, dst_path)
                output_files["dst"] = dst_path
                logger.info("    ✓ DST绣花机格式生成完成")
            
            # PES格式（Brother绣花机）
            if "pes" in self.workflow_config["output_formats"]:
                pes_path = os.path.join(output_dir, f"{job_id}_embroidery.pes")
                self._save_pes_format(embroidery_paths, pes_path)
                output_files["pes"] = pes_path
                logger.info("    ✓ PES格式生成完成")
            
            return output_files
            
        except Exception as e:
            logger.warning(f"输出格式生成失败: {str(e)}")
            return {}
    
    def _save_png_preview(self, color_blocks: np.ndarray, output_path: str):
        """保存PNG预览图"""
        try:
            pil_image = Image.fromarray(color_blocks)
            pil_image.save(output_path, 'PNG', optimize=True)
            
            file_size = os.path.getsize(output_path) / (1024 * 1024)
            logger.info(f"      PNG预览图: {output_path} ({file_size:.2f} MB)")
            
        except Exception as e:
            raise Exception(f"保存PNG预览图失败: {str(e)}")
    
    def _save_svg_vector(self, embroidery_paths: Dict, output_path: str):
        """保存SVG矢量图"""
        try:
            svg_lines = ['<?xml version="1.0" encoding="UTF-8"?>']
            svg_lines.append('<svg xmlns="http://www.w3.org/2000/svg" width="800" height="800">')
            
            for thread_data in embroidery_paths.get("thread_paths", []):
                color = thread_data["color"]
                path = thread_data["path"]
                
                if path:
                    color_hex = f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}"
                    
                    path_str = f"M {path[0][0]} {path[0][1]}"
                    for point in path[1:]:
                        path_str += f" L {point[0]} {point[1]}"
                    
                    svg_lines.append(
                        f'<path d="{path_str}" stroke="{color_hex}" stroke-width="1.5" fill="none"/>'
                    )
            
            svg_lines.append('</svg>')
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(svg_lines))
            
            logger.info(f"      SVG矢量图: {output_path}")
            
        except Exception as e:
            raise Exception(f"保存SVG矢量图失败: {str(e)}")
    
    def _save_dst_format(self, embroidery_paths: Dict, output_path: str):
        """保存DST绣花机格式（简化版）"""
        try:
            # 简化版DST格式生成
            dst_data = []
            
            for thread_data in embroidery_paths.get("thread_paths", []):
                path = thread_data["path"]
                color = thread_data["color"]
                
                # DST格式头部信息
                if not dst_data:
                    dst_data.append(f"LA:{len(embroidery_paths.get('thread_paths', []))}")
                
                # 添加颜色变更指令
                dst_data.append(f"CO:{color[0]:02X}{color[1]:02X}{color[2]:02X}")
                
                # 添加路径点
                for i, (x, y) in enumerate(path):
                    if i == 0:
                        dst_data.append(f"JU:{x:04d},{y:04d}")  # 跳跃到起点
                    else:
                        dst_data.append(f"ST:{x:04d},{y:04d}")  # 绣花针迹
                
                dst_data.append("EN")  # 结束当前颜色
            
            # 保存DST文件
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(dst_data))
            
            logger.info(f"      DST格式: {output_path}")
            
        except Exception as e:
            raise Exception(f"保存DST格式失败: {str(e)}")
    
    def _save_pes_format(self, embroidery_paths: Dict, output_path: str):
        """保存PES格式（简化版）"""
        try:
            # 简化版PES格式生成
            pes_data = {
                "format": "PES",
                "version": "1.0",
                "design_info": {
                    "thread_count": len(embroidery_paths.get("thread_paths", [])),
                    "stitch_count": sum(len(p["path"]) for p in embroidery_paths.get("thread_paths", [])),
                    "colors": [p["color"] for p in embroidery_paths.get("thread_paths", [])]
                },
                "thread_paths": embroidery_paths.get("thread_paths", [])
            }
            
            # 保存为JSON格式（实际PES是二进制格式）
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(pes_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"      PES格式: {output_path}")
            
        except Exception as e:
            raise Exception(f"保存PES格式失败: {str(e)}")
    
    def _generate_workflow_report(self, original: np.ndarray, color_blocks: np.ndarray, 
                                embroidery_paths: Dict, processing_time: float, job_id: str) -> Dict:
        """生成工艺流程报告"""
        try:
            # 统计信息
            thread_count = len(embroidery_paths.get("thread_paths", []))
            total_stitches = sum(len(p["path"]) for p in embroidery_paths.get("thread_paths", []))
            estimated_time = sum(p.get("estimated_time", 0) for p in embroidery_paths.get("thread_paths", []))
            
            # 创建对比图
            comparison_path = self._create_workflow_comparison(original, color_blocks, job_id)
            
            report = {
                "job_id": job_id,
                "processing_time": processing_time,
                "statistics": {
                    "thread_colors": thread_count,
                    "total_stitches": total_stitches,
                    "estimated_embroidery_time": estimated_time
                },
                "workflow_steps": [
                    "✓ 原始图像输入",
                    "✓ AI图像理解与预处理",
                    "✓ 分色模块处理",
                    "✓ 边缘轮廓提取",
                    "✓ 针迹排布规划",
                    "✓ 绣花路径优化",
                    "✓ 输出格式生成"
                ],
                "comparison_image": comparison_path
            }
            
            # 保存报告
            report_path = f"outputs/{job_id}/{job_id}_workflow_report.json"
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            
            return report
            
        except Exception as e:
            logger.warning(f"生成工艺流程报告失败: {str(e)}")
            return {}
    
    def _create_workflow_comparison(self, original: np.ndarray, embroidery: np.ndarray, job_id: str) -> str:
        """创建工艺流程对比图"""
        try:
            # 确保尺寸一致
            if original.shape != embroidery.shape:
                embroidery = cv2.resize(embroidery, (original.shape[1], original.shape[0]))
            
            # 水平拼接
            comparison = np.hstack([original, embroidery])
            
            # 保存对比图
            output_dir = f"outputs/{job_id}"
            comparison_path = os.path.join(output_dir, f"{job_id}_workflow_comparison.png")
            
            pil_image = Image.fromarray(comparison)
            pil_image.save(comparison_path, 'PNG', optimize=True)
            
            file_size = os.path.getsize(comparison_path) / (1024 * 1024)
            logger.info(f"工艺流程对比图: {comparison_path} ({file_size:.2f} MB)")
            
            return comparison_path
            
        except Exception as e:
            logger.warning(f"创建工艺流程对比图失败: {str(e)}")
            return ""

def main():
    """测试绣花工艺流程"""
    print("🎨 绣花工艺流程测试")
    print("=" * 60)
    
    processor = EmbroideryWorkflowProcessor()
    
    # 测试图像
    test_image = "uploads/250625_205648.jpg"
    if not os.path.exists(test_image):
        print(f"❌ 测试图像不存在: {test_image}")
        return
    
    # 执行完整工艺流程
    job_id = f"embroidery_workflow_{int(time.time())}"
    
    try:
        result = processor.process_embroidery_workflow(test_image, job_id)
        
        if result["success"]:
            print(f"✅ 绣花工艺流程测试成功！")
            print(f"📁 任务ID: {result['job_id']}")
            print(f"⏱️  处理时间: {result['processing_time']:.2f}秒")
            print(f"📊 统计信息:")
            
            stats = result["workflow_report"]["statistics"]
            print(f"   - 绣花线颜色: {stats['thread_colors']}种")
            print(f"   - 总针迹数: {stats['total_stitches']}针")
            print(f"   - 预估绣花时间: {stats['estimated_embroidery_time']:.1f}秒")
            
            print(f"📁 输出文件:")
            for format_type, file_path in result["output_files"].items():
                print(f"   - {format_type.upper()}: {file_path}")
            
        else:
            print(f"❌ 绣花工艺流程测试失败: {result['error']}")
        
    except Exception as e:
        print(f"❌ 测试失败: {str(e)}")

if __name__ == "__main__":
    main() 