import os
import json
import requests
import numpy as np
import cv2
from PIL import Image
import logging
from typing import Tuple, Dict, Any, Optional
import base64
from io import BytesIO

logger = logging.getLogger(__name__)

class AIEnhancedProcessor:
    """基于国内大模型的AI增强图像处理器"""
    
    def __init__(self):
        self.deepseek_api_key = os.getenv('DEEPSEEK_API_KEY')
        self.qwen_api_key = os.getenv('QWEN_API_KEY')
        
        # API配置
        self.api_timeout = int(os.getenv('API_TIMEOUT', '30'))
        self.ai_enabled = os.getenv('AI_ENHANCED_MODE', 'true').lower() == 'true'
        
        if not self.deepseek_api_key:
            logger.warning("未找到DEEPSEEK_API_KEY环境变量")
        if not self.qwen_api_key:
            logger.warning("未找到QWEN_API_KEY环境变量")
        
        if not self.deepseek_api_key and not self.qwen_api_key:
            logger.warning("未找到任何AI API密钥，将使用传统分析方法")
            self.ai_enabled = False
    
    def analyze_image_content(self, image: np.ndarray) -> Dict[str, Any]:
        """使用真实AI API分析图像内容，识别主体对象"""
        try:
            if not self.ai_enabled:
                return self._fallback_analysis(image)
            
            # 优先使用通义千问VL分析
            if self.qwen_api_key:
                try:
                    # 将图像转换为base64
                    image_base64 = self._image_to_base64(image)
                    
                    # 调用通义千问VL API
                    analysis_result = self._call_qwen_vl(image_base64)
                    
                    if analysis_result:
                        # 解析结果获取主体信息
                        subject_info = self._parse_subject_info(analysis_result)
                        logger.info(f"通义千问VL分析成功: {subject_info['main_subject']}")
                        return subject_info
                except Exception as e:
                    logger.warning(f"通义千问VL分析失败: {str(e)}")
            
            # 备用：DeepSeek API分析
            if self.deepseek_api_key:
                try:
                    subject_info = self._call_deepseek_analysis(image)
                    if subject_info:
                        logger.info(f"DeepSeek分析成功: {subject_info['main_subject']}")
                        return subject_info
                except Exception as e:
                    logger.warning(f"DeepSeek分析失败: {str(e)}")
            
            # 最后回退到传统分析
            logger.warning("所有AI API不可用，使用传统分析方法")
            return self._fallback_analysis(image)
            
        except Exception as e:
            logger.warning(f"AI图像分析失败: {str(e)}")
            return self._fallback_analysis(image)
    
    def generate_smart_mask(self, image: np.ndarray, subject_info: Dict[str, Any]) -> np.ndarray:
        """基于AI分析结果生成智能主体掩码"""
        try:
            height, width = image.shape[:2]
            
            # 基于主体类型调整检测策略
            subject_type = subject_info.get("main_subject", "unknown")
            
            if subject_type in ["动物", "人物", "熊猫", "猫", "狗"]:
                # 生物体检测策略
                mask = self._detect_living_subject(image, subject_info)
            elif subject_type in ["物体", "建筑", "风景"]:
                # 静物检测策略
                mask = self._detect_static_subject(image, subject_info)
            else:
                # 通用检测策略
                mask = self._detect_general_subject(image)
            
            return mask
            
        except Exception as e:
            logger.warning(f"智能掩码生成失败: {str(e)}")
            return self._fallback_mask(image)
    
    def enhance_colors_with_ai(self, image: np.ndarray, subject_info: Dict[str, Any]) -> np.ndarray:
        """基于AI理解的智能颜色增强"""
        try:
            subject_type = subject_info.get("main_subject", "unknown")
            
            # 根据主体类型调整颜色策略
            if "熊猫" in subject_type:
                # 熊猫：强化黑白对比
                enhanced = self._enhance_panda_colors(image)
            elif "动物" in subject_type:
                # 动物：保持自然色彩
                enhanced = self._enhance_animal_colors(image)
            else:
                # 通用增强
                enhanced = self._enhance_general_colors(image)
            
            return enhanced
            
        except Exception as e:
            logger.warning(f"AI颜色增强失败: {str(e)}")
            return image
    
    def _image_to_base64(self, image: np.ndarray) -> str:
        """将numpy图像转换为base64字符串"""
        try:
            # 转换为PIL Image
            pil_image = Image.fromarray(image)
            
            # 压缩图像以减少API调用大小
            pil_image.thumbnail((512, 512), Image.Resampling.LANCZOS)
            
            # 转换为base64
            buffer = BytesIO()
            pil_image.save(buffer, format='JPEG', quality=85)
            img_str = base64.b64encode(buffer.getvalue()).decode()
            
            return img_str
            
        except Exception as e:
            raise Exception(f"图像转换失败: {str(e)}")
    
    def _call_qwen_vl(self, image_base64: str) -> str:
        """调用通义千问VL模型分析图像"""
        try:
            import dashscope
            from dashscope import MultiModalConversation
            
            # 配置API
            dashscope.api_key = self.qwen_api_key
            
            logger.info("正在调用通义千问VL进行图像内容分析...")
            
            # 构建多模态消息
            messages = [
                {
                    'role': 'user',
                    'content': [
                        {
                            'image': f'data:image/jpeg;base64,{image_base64}'
                        },
                        {
                            'text': '请分析这张图片中的主要对象。识别主体是什么（动物/人物/物体），位置在哪里，有什么特征。简洁回答格式：主体：[名称]，位置：[位置]，特征：[特征]'
                        }
                    ]
                }
            ]
            
            # 调用通义千问VL API
            response = MultiModalConversation.call(
                model='qwen-vl-plus',
                messages=messages
            )
            
            if response.status_code == 200:
                analysis_result = response.output.choices[0].message.content
                logger.info(f"通义千问VL分析完成: {analysis_result[:100]}...")
                return analysis_result
            else:
                logger.warning(f"通义千问VL API调用失败: {response.message}")
                return ""
            
        except ImportError:
            logger.warning("dashscope库未安装，请运行: pip install dashscope")
            return ""
        except Exception as e:
            logger.warning(f"通义千问VL调用异常: {str(e)}")
            return ""
    
    def _parse_subject_info(self, analysis_text: str) -> Dict[str, Any]:
        """解析AI分析结果"""
        try:
            subject_info = {
                "main_subject": "unknown",
                "position": "center",
                "features": [],
                "confidence": 0.8
            }
            
            # 关键词匹配
            if "熊猫" in analysis_text:
                subject_info["main_subject"] = "熊猫"
                subject_info["features"] = ["黑白色", "圆形耳朵", "黑眼圈"]
            elif any(word in analysis_text for word in ["猫", "狗", "动物"]):
                subject_info["main_subject"] = "动物"
            elif any(word in analysis_text for word in ["人", "人物", "人像"]):
                subject_info["main_subject"] = "人物"
            
            return subject_info
            
        except Exception as e:
            logger.warning(f"分析结果解析失败: {str(e)}")
            return {"main_subject": "unknown", "confidence": 0.0}
    
    def _call_deepseek_analysis(self, image: np.ndarray) -> Optional[Dict[str, Any]]:
        """调用DeepSeek进行图像分析"""
        try:
            import requests
            
            # 将图像转换为base64
            image_base64 = self._image_to_base64(image)
            
            logger.info("正在调用DeepSeek进行图像内容分析...")
            
            # DeepSeek API endpoint
            url = "https://api.deepseek.com/v1/chat/completions"
            
            headers = {
                "Authorization": f"Bearer {self.deepseek_api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": "deepseek-vl-chat",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_base64}"
                                }
                            },
                            {
                                "type": "text",
                                "text": "请分析这张图片中的主要对象。识别主体是什么（动物/人物/物体），位置在哪里，有什么特征。简洁回答格式：主体：[名称]，位置：[位置]，特征：[特征]"
                            }
                        ]
                    }
                ]
            }
            
            response = requests.post(url, headers=headers, json=data, timeout=self.api_timeout)
            
            if response.status_code == 200:
                result = response.json()
                analysis_text = result['choices'][0]['message']['content']
                logger.info(f"DeepSeek分析完成: {analysis_text[:100]}...")
                
                # 解析结果
                subject_info = self._parse_subject_info(analysis_text)
                return subject_info
            else:
                logger.warning(f"DeepSeek API调用失败: HTTP {response.status_code}")
                return None
                
        except Exception as e:
            logger.warning(f"DeepSeek分析失败: {str(e)}")
            return None
    
    def _fallback_analysis(self, image: np.ndarray) -> Dict[str, Any]:
        """传统回退分析方法"""
        try:
            # 基于颜色特征的智能检测
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            
            # 黑白区域检测（熊猫特征）
            black_mask = cv2.inRange(hsv, (0, 0, 0), (180, 255, 50))
            white_mask = cv2.inRange(hsv, (0, 0, 200), (180, 30, 255))
            
            black_ratio = np.sum(black_mask > 0) / (image.shape[0] * image.shape[1])
            white_ratio = np.sum(white_mask > 0) / (image.shape[0] * image.shape[1])
            
            if black_ratio > 0.15 and white_ratio > 0.3:
                main_subject = "熊猫"
                confidence = 0.8
            elif black_ratio > 0.2:
                main_subject = "动物"
                confidence = 0.6
            else:
                main_subject = "其他"
                confidence = 0.4
                
            logger.info(f"传统分析结果: {main_subject} (置信度: {confidence})")
            
            return {
                "main_subject": main_subject,
                "position": "center",
                "features": ["检测到的主体"],
                "confidence": confidence
            }
            
        except Exception as e:
            logger.warning(f"传统分析失败: {str(e)}")
            return {"main_subject": "unknown", "confidence": 0.0}
    
    def _detect_living_subject(self, image: np.ndarray, subject_info: Dict[str, Any]) -> np.ndarray:
        """检测生物体主体"""
        height, width = image.shape[:2]
        
        if subject_info["main_subject"] == "熊猫":
            # 熊猫特化检测：查找黑白区域
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            # 检测白色区域（熊猫脸部）
            white_mask = gray > 180
            # 检测黑色区域（熊猫眼部、耳朵）
            black_mask = gray < 80
            
            # 合并熊猫特征区域
            panda_mask = np.logical_or(white_mask, black_mask)
            
            # 形态学处理连接区域
            kernel = np.ones((15, 15), np.uint8)
            panda_mask = cv2.morphologyEx(panda_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
            
            return panda_mask > 0
        
        else:
            # 通用生物体检测
            return self._detect_general_subject(image)
    
    def _detect_static_subject(self, image: np.ndarray, subject_info: Dict[str, Any]) -> np.ndarray:
        """检测静物主体"""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            mask = np.zeros(gray.shape, dtype=np.uint8)
            cv2.fillPoly(mask, [largest_contour], 255)
            return mask > 0
        
        return self._fallback_mask(image)
    
    def _detect_general_subject(self, image: np.ndarray) -> np.ndarray:
        """通用主体检测"""
        height, width = image.shape[:2]
        
        center_x, center_y = width // 2, height // 2
        center_width = int(width * 0.7)
        center_height = int(height * 0.7)
        
        mask = np.zeros((height, width), dtype=bool)
        start_x = max(0, center_x - center_width // 2)
        end_x = min(width, center_x + center_width // 2)
        start_y = max(0, center_y - center_height // 2)
        end_y = min(height, center_y + center_height // 2)
        
        mask[start_y:end_y, start_x:end_x] = True
        return mask
    
    def _fallback_mask(self, image: np.ndarray) -> np.ndarray:
        """回退掩码策略"""
        return self._detect_general_subject(image)
    
    def _enhance_panda_colors(self, image: np.ndarray) -> np.ndarray:
        """熊猫专用颜色增强"""
        # 强化黑白对比
        enhanced = image.copy()
        
        # 转换到LAB色彩空间进行处理
        lab = cv2.cvtColor(enhanced, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        
        # 增强亮度对比
        l = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8)).apply(l)
        
        # 重新组合
        enhanced_lab = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)
        
        return enhanced
    
    def _enhance_animal_colors(self, image: np.ndarray) -> np.ndarray:
        """动物专用颜色增强"""
        # 保持自然色彩，适度增强
        enhanced = cv2.convertScaleAbs(image, alpha=1.1, beta=10)
        return enhanced
    
    def _enhance_general_colors(self, image: np.ndarray) -> np.ndarray:
        """通用颜色增强"""
        # 适度增强饱和度和对比度
        enhanced = cv2.convertScaleAbs(image, alpha=1.05, beta=5)
        return enhanced 