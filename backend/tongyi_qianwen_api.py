#!/usr/bin/env python3
"""
通义千问API集成模块
用于调用通义千问大模型进行织机识别图生成
"""

import os
import sys
import logging
import time
import json
import base64
import requests
from pathlib import Path
from typing import Optional, Dict, Any, List
from PIL import Image
import io
import numpy as np

class TongyiQianwenAPI:
    """
    通义千问API封装
    用于调用通义千问大模型进行图像生成和增强
    """
    
    def __init__(self, 
                 api_key: str = None,
                 base_url: str = "https://dashscope.aliyuncs.com",
                 model: str = "qwen-vl-plus"):
        
        self.api_key = api_key or os.getenv("TONGYI_API_KEY")
        self.base_url = base_url
        self.model = model
        self.logger = logging.getLogger(__name__)
        
        if not self.api_key:
            self.logger.warning("通义千问API密钥未设置，请设置TONGYI_API_KEY环境变量")
    
    def _encode_image(self, image_path: str) -> str:
        """将图片编码为base64"""
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            self.logger.error(f"图片编码失败: {e}")
            return None
    
    def _decode_image(self, base64_string: str, output_path: str) -> bool:
        """将base64解码为图片"""
        try:
            image_data = base64.b64decode(base64_string)
            with open(output_path, "wb") as f:
                f.write(image_data)
            return True
        except Exception as e:
            self.logger.error(f"图片解码失败: {e}")
            return False
    
    def generate_loom_recognition_image(self, 
                                      source_image_path: str, 
                                      output_path: str = None,
                                      style_prompt: str = None) -> Optional[str]:
        """
        使用通义千问生成织机识别图
        """
        if not self.api_key:
            self.logger.error("通义千问API密钥未设置")
            return None
        try:
            # 编码图片
            base64_image = self._encode_image(source_image_path)
            if not base64_image:
                return None
            # 生成输出路径
            if output_path is None:
                timestamp = int(time.time() * 1000)
                source_name = Path(source_image_path).stem
                output_path = f"uploads/tongyi_{timestamp}_{source_name}.jpg"
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            # 风格描述
            style_prompt = style_prompt or "请将这张图片转换为织机识别图，风格要求：像素化，色彩丰富，边缘清晰，强对比度"
            # 构建请求数据
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            data = {
                "model": self.model,
                "input": {
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"image": base64_image},
                                {"text": style_prompt}
                            ]
                        }
                    ]
                }
            }
            # 发送请求
            url = f"{self.base_url}/api/v1/services/aigc/multimodal-generation/generation"
            response = requests.post(
                url,
                headers=headers,
                json=data,
                timeout=60
            )
            if response.status_code == 200:
                result = response.json()
                # 解析响应
                if "output" in result and "images" in result["output"]:
                    image_data = result["output"]["images"][0]
                    # 保存图片
                    if self._decode_image(image_data, output_path):
                        self.logger.info(f"通义千问生成成功: {output_path}")
                        return output_path
                    else:
                        self.logger.error("图片保存失败")
                        return None
                else:
                    self.logger.error(f"响应格式错误: {result}")
                    return None
            else:
                self.logger.error(f"API请求失败: {response.status_code} - {response.text}")
                return None
        except Exception as e:
            self.logger.error(f"通义千问生成失败: {e}")
            return None
    
    def enhance_image_quality(self, 
                            image_path: str, 
                            output_path: str = None) -> Optional[str]:
        """
        使用通义千问增强图片质量
        
        Args:
            image_path: 输入图片路径
            output_path: 输出图片路径
            
        Returns:
            增强后的图片路径，失败返回None
        """
        if not self.api_key:
            self.logger.error("通义千问API密钥未设置")
            return None
        
        try:
            # 编码图片
            base64_image = self._encode_image(image_path)
            if not base64_image:
                return None
            
            # 生成输出路径
            if output_path is None:
                timestamp = int(time.time() * 1000)
                source_name = Path(image_path).stem
                output_path = f"uploads/tongyi_enhanced_{timestamp}_{source_name}.jpg"
            
            # 确保输出目录存在
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # 构建增强提示词
            enhance_prompt = """
            请增强这张图片的质量，要求：
            1. 提高分辨率和清晰度
            2. 增强颜色饱和度和对比度
            3. 锐化边缘和细节
            4. 优化色彩平衡
            5. 保持原始内容不变
            请输出高质量的增强版本。
            """
            
            # 构建请求数据
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": self.model,
                "input": {
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "image": base64_image
                                },
                                {
                                    "text": enhance_prompt
                                }
                            ]
                        }
                    ]
                },
                "parameters": {
                    "result_format": "message",
                    "max_tokens": 1500,
                    "temperature": 0.5,
                    "top_p": 0.9
                }
            }
            
            # 发送请求
            response = requests.post(
                f"{self.base_url}/services/aigc/text-generation/generation",
                headers=headers,
                json=data,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                
                # 解析响应
                if "output" in result and "images" in result["output"]:
                    image_data = result["output"]["images"][0]
                    
                    # 保存图片
                    if self._decode_image(image_data, output_path):
                        self.logger.info(f"通义千问增强成功: {output_path}")
                        return output_path
                    else:
                        self.logger.error("图片保存失败")
                        return None
                else:
                    self.logger.error(f"响应格式错误: {result}")
                    return None
            else:
                self.logger.error(f"API请求失败: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            self.logger.error(f"通义千问增强失败: {e}")
            return None
    
    def analyze_image_style(self, image_path: str) -> Optional[Dict[str, Any]]:
        """
        使用通义千问分析图片风格特征
        
        Args:
            image_path: 图片路径
            
        Returns:
            风格分析结果，失败返回None
        """
        if not self.api_key:
            self.logger.error("通义千问API密钥未设置")
            return None
        
        try:
            # 编码图片
            base64_image = self._encode_image(image_path)
            if not base64_image:
                return None
            
            # 构建分析提示词
            analysis_prompt = """
            请分析这张图片的风格特征，包括：
            1. 颜色特征：饱和度、对比度、色相分布
            2. 纹理特征：边缘清晰度、纹理复杂度
            3. 结构特征：构图、层次感
            4. 风格类型：艺术风格、技术特征
            5. 质量评估：清晰度、噪声水平
            
            请以JSON格式返回分析结果。
            """
            
            # 构建请求数据
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": self.model,
                "input": {
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "image": base64_image
                                },
                                {
                                    "text": analysis_prompt
                                }
                            ]
                        }
                    ]
                },
                "parameters": {
                    "result_format": "message",
                    "max_tokens": 1000,
                    "temperature": 0.3,
                    "top_p": 0.8
                }
            }
            
            # 发送请求
            response = requests.post(
                f"{self.base_url}/services/aigc/text-generation/generation",
                headers=headers,
                json=data,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                
                # 解析响应
                if "output" in result and "text" in result["output"]:
                    analysis_text = result["output"]["text"]
                    
                    # 尝试解析JSON
                    try:
                        analysis_result = json.loads(analysis_text)
                        return analysis_result
                    except json.JSONDecodeError:
                        # 如果不是JSON格式，返回文本
                        return {"analysis": analysis_text}
                else:
                    self.logger.error(f"响应格式错误: {result}")
                    return None
            else:
                self.logger.error(f"API请求失败: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            self.logger.error(f"通义千问分析失败: {e}")
            return None
    
    def get_api_status(self) -> Dict[str, Any]:
        """获取API状态信息"""
        return {
            "api_available": bool(self.api_key),
            "model": self.model,
            "base_url": self.base_url,
            "api_key_set": bool(self.api_key)
        }

# 单例模式
_tongyi_api = None

def get_tongyi_api() -> TongyiQianwenAPI:
    """获取通义千问API实例（单例模式）"""
    global _tongyi_api
    if _tongyi_api is None:
        _tongyi_api = TongyiQianwenAPI()
    return _tongyi_api

def init_tongyi_api():
    """初始化通义千问API"""
    global _tongyi_api
    if _tongyi_api is None:
        _tongyi_api = TongyiQianwenAPI()
    return _tongyi_api 