# tongyi_composer_api.py
# 通义千问Composer图生图API封装
# 基于Qwen-VL + Composer模型实现图生图功能

import os
import base64
import requests
import logging
import json
import time
from typing import Optional, Dict, Any, List
from pathlib import Path

logger = logging.getLogger(__name__)

class TongyiComposerAPI:
    """
    通义千问Composer图生图API封装
    基于Qwen-VL + Composer模型实现图生图功能
    """
    
    def __init__(self, 
                 api_key: str = None,
                 base_url: str = "https://dashscope.aliyuncs.com",
                 model: str = "wanx2.1-imageedit"):
        
        self.api_key = api_key or os.getenv("TONGYI_API_KEY")
        self.base_url = base_url
        self.model = model
        # 修正为正确的图生图端点
        self.endpoint = "/api/v1/services/aigc/image2image/image-synthesis"
        self.logger = logging.getLogger(__name__)
        
        if not self.api_key:
            self.logger.warning("通义千问API密钥未设置，请设置TONGYI_API_KEY环境变量")
    
    def get_api_status(self) -> Dict[str, Any]:
        """获取API状态"""
        return {
            "api_available": bool(self.api_key),
            "model": self.model,
            "base_url": self.base_url
        }
    
    def _encode_image(self, image_path: str) -> str:
        """编码图片为base64"""
        try:
            with open(image_path, "rb") as f:
                image_data = f.read()
            return base64.b64encode(image_data).decode('utf-8')
        except Exception as e:
            self.logger.error(f"图片编码失败: {e}")
            raise
    
    def _get_headers(self) -> Dict[str, str]:
        """获取请求头"""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "X-DashScope-SSE": "disable",
            "X-DashScope-Async": "enable"
        }
    
    def _submit_image_to_image_task(self, 
                                   image_path: str, 
                                   prompt: str = "",
                                   style_preset: str = "weaving_machine") -> str:
        """提交图生图异步任务"""
        
        # 编码图片
        base64_image = self._encode_image(image_path)
        
        # 构造请求头
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "X-DashScope-Async": "enable"
        }
        
        # 构造请求体
        data = {
            "model": self.model,
            "input": {
                "image": base64_image
            },
            "parameters": {
                "prompt": prompt,
                "style": style_preset,
                "size": "1024*1024",
                "n": 1,
                "seed": 42
            }
        }
        
        # 发送请求
        try:
            response = requests.post(
                f"{self.base_url}{self.endpoint}",
                headers=headers,
                json=data,
                timeout=60
            )
            
            self.logger.info(f"API响应状态码: {response.status_code}")
            self.logger.info(f"API响应头: {dict(response.headers)}")
            
            if response.status_code == 200:
                result = response.json()
                self.logger.info(f"API响应成功: {result}")
                
                # 检查是否是异步任务
                if "output" in result and "task_id" in result["output"]:
                    task_id = result["output"]["task_id"]
                    task_status = result["output"].get("task_status", "PENDING")
                    self.logger.info(f"异步任务ID: {task_id}, 状态: {task_status}")
                    
                    # 轮询任务状态
                    if task_status in ["PENDING", "RUNNING"]:
                        return self._poll_task_status(task_id, headers)
                    else:
                        raise Exception(f"任务状态异常: {task_status}")
                else:
                    raise Exception(f"响应格式错误: {result}")
            else:
                error_text = response.text
                self.logger.error(f"API请求失败: {response.status_code} - {error_text}")
                raise Exception(f"提交任务失败: {response.status_code} - {error_text}")
                
        except requests.exceptions.Timeout:
            self.logger.error("API请求超时")
            raise Exception("API请求超时")
        except requests.exceptions.RequestException as e:
            self.logger.error(f"API请求异常: {str(e)}")
            raise Exception(f"API请求异常: {str(e)}")
        except Exception as e:
            self.logger.error(f"未知错误: {str(e)}")
            raise Exception(f"未知错误: {str(e)}")
    
    def _poll_task_status(self, task_id: str, max_wait: int = 300) -> Dict[str, Any]:
        """轮询任务状态"""
        
        start_time = time.time()
        while time.time() - start_time < max_wait:
            try:
                response = requests.get(
                    f"{self.base_url}/api/v1/tasks/{task_id}",
                    headers=self._get_headers(),
                    timeout=30
                )
                
                if response.status_code != 200:
                    raise Exception(f"查询任务状态失败: {response.status_code}")
                
                result = response.json()
                task_status = result.get("output", {}).get("task_status")
                
                if task_status is None:
                    self.logger.warning(f"任务状态为空，响应: {result}")
                    time.sleep(5)
                    continue
                
                if task_status == "SUCCEEDED":
                    return result
                elif task_status == "FAILED":
                    raise Exception(f"任务执行失败: {result}")
                elif task_status in ["PENDING", "RUNNING"]:
                    self.logger.info(f"任务状态: {task_status}, 继续等待...")
                    time.sleep(5)  # 等待5秒后重试
                else:
                    raise Exception(f"未知任务状态: {task_status}")
                    
            except Exception as e:
                self.logger.error(f"轮询任务状态异常: {e}")
                time.sleep(5)
        
        raise Exception(f"任务超时，等待时间: {max_wait}秒")
    
    def _download_image(self, image_url: str, output_path: str) -> str:
        """下载图片"""
        try:
            response = requests.get(image_url, timeout=60)
            if response.status_code != 200:
                raise Exception(f"下载图片失败: {response.status_code}")
            
            with open(output_path, "wb") as f:
                f.write(response.content)
            
            return output_path
        except Exception as e:
            self.logger.error(f"下载图片失败: {e}")
            raise
    
    def generate_image_to_image(self, 
                              source_image_path: str, 
                              output_path: str = None,
                              prompt: str = "",
                              style_preset: str = "weaving_machine") -> str:
        """
        使用通义千问Composer模型进行图生图
        
        Args:
            source_image_path: 源图片路径
            output_path: 输出图片路径
            prompt: 提示词
            style_preset: 风格预设
            
        Returns:
            生成的图片路径
        """
        
        if not self.api_key:
            raise Exception("通义千问API密钥未设置")
        
        if not os.path.exists(source_image_path):
            raise Exception(f"源图片不存在: {source_image_path}")
        
        # 生成输出路径
        if output_path is None:
            timestamp = int(time.time() * 1000)
            output_path = f"outputs/composer_{timestamp}.jpg"
        
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        try:
            # 1. 提交异步任务
            self.logger.info("提交图生图异步任务...")
            task_id = self._submit_image_to_image_task(
                source_image_path, prompt, style_preset
            )
            self.logger.info(f"任务已提交，ID: {task_id}")
            
            # 2. 轮询任务状态
            self.logger.info("轮询任务状态...")
            result = self._poll_task_status(task_id)
            
            # 3. 获取生成的图片URL
            if "output" in result and "images" in result["output"]:
                images = result["output"]["images"]
                if images and "url" in images[0]:
                    image_url = images[0]["url"]
                    self.logger.info(f"图片生成成功，URL: {image_url}")
                    
                    # 4. 下载图片
                    self.logger.info("下载生成的图片...")
                    return self._download_image(image_url, output_path)
                else:
                    raise Exception("响应中没有图片URL")
            else:
                raise Exception(f"响应格式错误: {result}")
                
        except Exception as e:
            self.logger.error(f"图生图处理失败: {e}")
            raise

# 全局API实例
_composer_api = None

def get_composer_api() -> TongyiComposerAPI:
    """获取Composer API实例"""
    global _composer_api
    if _composer_api is None:
        _composer_api = TongyiComposerAPI()
    return _composer_api

def init_composer_api(api_key: str = None) -> TongyiComposerAPI:
    """初始化Composer API实例"""
    global _composer_api
    _composer_api = TongyiComposerAPI(api_key=api_key)
    return _composer_api 