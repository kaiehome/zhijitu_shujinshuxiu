import os
import json
import requests
import numpy as np
import cv2
from PIL import Image
import logging
from typing import Tuple, Dict, Any, Optional, List
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
        """检测生物体主体，特别优化熊猫检测"""
        try:
            height, width = image.shape[:2]
            
            # 检查是否为熊猫
            if "熊猫" in subject_info.get("main_subject", ""):
                return self._detect_panda_specific(image, subject_info)
            
            # 其他生物体检测
            return self._detect_general_living_subject(image, subject_info)
            
        except Exception as e:
            logger.warning(f"生物体检测失败: {str(e)}")
            return self._fallback_mask(image)
    
    def _detect_panda_specific(self, image: np.ndarray, subject_info: Dict[str, Any]) -> np.ndarray:
        """专门针对熊猫的检测算法，增强面部特征检测"""
        try:
            height, width = image.shape[:2]
            
            # 转换为灰度图
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # 创建掩码
            mask = np.zeros((height, width), dtype=np.uint8)
            
            # 1. 检测熊猫的黑色区域（眼睛、耳朵、身体）
            # 使用自适应阈值检测黑色区域
            black_mask = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
            )
            
            # 2. 检测熊猫的白色区域（面部、身体）
            # 使用高阈值检测白色区域
            _, white_mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
            
            # 3. 结合黑白区域
            panda_mask = cv2.bitwise_or(black_mask, white_mask)
            
            # 4. 形态学操作清理噪声
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            panda_mask = cv2.morphologyEx(panda_mask, cv2.MORPH_CLOSE, kernel)
            panda_mask = cv2.morphologyEx(panda_mask, cv2.MORPH_OPEN, kernel)
            
            # 5. 查找轮廓
            contours, _ = cv2.findContours(panda_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # 6. 选择最大的轮廓作为熊猫主体
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                cv2.fillPoly(mask, [largest_contour], 255)
            
            # 7. 增强面部特征检测
            face_features_mask = self._detect_panda_face_features(image, mask)
            if face_features_mask is not None:
                mask = cv2.bitwise_or(mask, face_features_mask)
            
            # 8. 特别检测下巴/颈部区域（基于用户标记）
            chin_mask = self._detect_chin_neck_area(image, mask)
            if chin_mask is not None:
                mask = cv2.bitwise_or(mask, chin_mask)
            
            # 9. 优化区域边界
            mask = self._optimize_panda_boundaries(mask)
            
            return mask
            
        except Exception as e:
            logger.warning(f"熊猫特定检测失败: {str(e)}")
            return self._fallback_mask(image)
    
    def _detect_panda_face_features(self, image: np.ndarray, base_mask: np.ndarray) -> Optional[np.ndarray]:
        """检测熊猫面部特征（眼睛、鼻子、耳朵）"""
        try:
            height, width = image.shape[:2]
            
            # 创建面部特征掩码
            features_mask = np.zeros((height, width), dtype=np.uint8)
            
            # 转换为灰度图
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # 1. 检测眼睛区域（熊猫的黑色眼圈）
            # 使用圆形检测器查找眼睛
            circles = cv2.HoughCircles(
                gray, cv2.HOUGH_GRADIENT, dp=1, minDist=50,
                param1=50, param2=30, minRadius=10, maxRadius=50
            )
            
            if circles is not None:
                circles = np.uint16(np.around(circles))
                for circle in circles[0, :]:
                    # 检查是否在基础掩码内
                    if base_mask[circle[1], circle[0]] > 0:
                        cv2.circle(features_mask, (circle[0], circle[1]), circle[2], 255, -1)
            
            # 2. 检测鼻子区域（熊猫的黑色鼻子）
            # 使用模板匹配或轮廓检测
            nose_mask = self._detect_panda_nose(gray, base_mask)
            if nose_mask is not None:
                features_mask = cv2.bitwise_or(features_mask, nose_mask)
            
            # 3. 检测耳朵区域（熊猫的圆形耳朵）
            ears_mask = self._detect_panda_ears(gray, base_mask)
            if ears_mask is not None:
                features_mask = cv2.bitwise_or(features_mask, ears_mask)
            
            return features_mask if np.sum(features_mask) > 0 else None
            
        except Exception as e:
            logger.warning(f"熊猫面部特征检测失败: {str(e)}")
            return None
    
    def _detect_panda_nose(self, gray: np.ndarray, base_mask: np.ndarray) -> Optional[np.ndarray]:
        """检测熊猫鼻子"""
        try:
            height, width = gray.shape
            
            # 创建鼻子掩码
            nose_mask = np.zeros((height, width), dtype=np.uint8)
            
            # 在图像下半部分寻找鼻子（通常鼻子在面部下半部分）
            lower_half = gray[height//2:, :]
            lower_mask = base_mask[height//2:, :]
            
            # 使用形态学操作找到小的黑色区域
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            black_regions = cv2.morphologyEx(lower_half, cv2.MORPH_OPEN, kernel)
            
            # 找到轮廓
            contours, _ = cv2.findContours(black_regions, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                # 检查轮廓大小（鼻子应该比较小）
                area = cv2.contourArea(contour)
                if 50 < area < 500:  # 鼻子的合理大小范围
                    # 调整坐标
                    contour[:, :, 1] += height//2
                    cv2.fillPoly(nose_mask, [contour], 255)
            
            return nose_mask if np.sum(nose_mask) > 0 else None
            
        except Exception as e:
            logger.warning(f"熊猫鼻子检测失败: {str(e)}")
            return None
    
    def _detect_panda_ears(self, gray: np.ndarray, base_mask: np.ndarray) -> Optional[np.ndarray]:
        """检测熊猫耳朵"""
        try:
            height, width = gray.shape
            
            # 创建耳朵掩码
            ears_mask = np.zeros((height, width), dtype=np.uint8)
            
            # 在图像上半部分寻找耳朵
            upper_half = gray[:height//2, :]
            upper_mask = base_mask[:height//2, :]
            
            # 使用圆形检测器查找耳朵
            circles = cv2.HoughCircles(
                upper_half, cv2.HOUGH_GRADIENT, dp=1, minDist=30,
                param1=50, param2=25, minRadius=15, maxRadius=40
            )
            
            if circles is not None:
                circles = np.uint16(np.around(circles))
                for circle in circles[0, :]:
                    # 检查是否在基础掩码内
                    if upper_mask[circle[1], circle[0]] > 0:
                        # 调整坐标
                        circle_y = circle[1]  # 已经在upper_half中
                        cv2.circle(ears_mask, (circle[0], circle_y), circle[2], 255, -1)
            
            return ears_mask if np.sum(ears_mask) > 0 else None
            
        except Exception as e:
            logger.warning(f"熊猫耳朵检测失败: {str(e)}")
            return None
    
    def _detect_chin_neck_area(self, image: np.ndarray, base_mask: np.ndarray) -> Optional[np.ndarray]:
        """检测熊猫下巴/颈部区域，基于用户标记优化"""
        try:
            height, width = image.shape[:2]
            
            # 创建下巴/颈部检测掩码
            chin_mask = np.zeros((height, width), dtype=np.uint8)
            
            # 基于用户标记的红色椭圆区域，优化检测策略
            # 1. 在基础掩码的下半部分寻找下巴/颈部区域
            lower_region = base_mask[height//2:, :]
            
            if np.sum(lower_region) > 0:
                # 2. 使用更精确的轮廓检测
                contours, _ = cv2.findContours(lower_region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                if contours:
                    # 3. 分析轮廓形状，找到符合下巴/颈部特征的轮廓
                    chin_contours = self._filter_chin_neck_contours(contours, width, height//2)
                    
                    for contour in chin_contours:
                        # 调整轮廓坐标（因为是从下半部分提取的）
                        contour[:, :, 1] += height//2
                        
                        # 4. 使用椭圆拟合优化区域形状
                        if len(contour) >= 5:  # 需要至少5个点才能拟合椭圆
                            ellipse = cv2.fitEllipse(contour)
                            # 创建椭圆掩码
                            cv2.ellipse(chin_mask, ellipse, 255, -1)
                        else:
                            # 如果点太少，直接填充轮廓
                            cv2.fillPoly(chin_mask, [contour], 255)
                    
                    # 5. 应用形态学操作使区域更平滑
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                    chin_mask = cv2.morphologyEx(chin_mask, cv2.MORPH_CLOSE, kernel)
                    chin_mask = cv2.morphologyEx(chin_mask, cv2.MORPH_OPEN, kernel)
            
            # 6. 如果检测到的区域太小，扩大搜索范围
            if np.sum(chin_mask) < 1000:  # 如果区域太小
                chin_mask = self._expand_chin_search_area(image, base_mask)
            
            return chin_mask if np.sum(chin_mask) > 0 else None
            
        except Exception as e:
            logger.warning(f"下巴/颈部区域检测失败: {str(e)}")
            return None
    
    def _filter_chin_neck_contours(self, contours: List, width: int, height: int) -> List:
        """过滤出符合下巴/颈部特征的轮廓"""
        try:
            filtered_contours = []
            
            for contour in contours:
                # 计算轮廓特征
                area = cv2.contourArea(contour)
                perimeter = cv2.arcLength(contour, True)
                
                if perimeter > 0:
                    # 计算圆形度（下巴/颈部通常比较圆润）
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    
                    # 计算边界框
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = w / h if h > 0 else 0
                    
                    # 下巴/颈部的特征：
                    # 1. 面积适中（不太小也不太大）
                    # 2. 圆形度较高（比较圆润）
                    # 3. 宽高比适中（不太扁也不太瘦）
                    # 4. 位置在图像下半部分
                    
                    if (500 < area < 10000 and  # 面积适中
                        circularity > 0.3 and    # 比较圆润
                        0.5 < aspect_ratio < 2.0 and  # 宽高比适中
                        y > height * 0.3):        # 在下半部分
                        
                        filtered_contours.append(contour)
            
            return filtered_contours
            
        except Exception as e:
            logger.warning(f"轮廓过滤失败: {str(e)}")
            return contours
    
    def _expand_chin_search_area(self, image: np.ndarray, base_mask: np.ndarray) -> np.ndarray:
        """扩大下巴/颈部搜索区域"""
        try:
            height, width = image.shape[:2]
            
            # 创建扩展搜索掩码
            expanded_mask = np.zeros((height, width), dtype=np.uint8)
            
            # 在更大的区域内搜索
            search_region = base_mask[height//3:, :]  # 从1/3处开始搜索
            
            if np.sum(search_region) > 0:
                # 使用更宽松的参数进行检测
                contours, _ = cv2.findContours(search_region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                if contours:
                    # 选择最大的轮廓
                    largest_contour = max(contours, key=cv2.contourArea)
                    
                    # 调整坐标
                    largest_contour[:, :, 1] += height//3
                    
                    # 填充轮廓
                    cv2.fillPoly(expanded_mask, [largest_contour], 255)
                    
                    # 应用形态学操作
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
                    expanded_mask = cv2.morphologyEx(expanded_mask, cv2.MORPH_CLOSE, kernel)
            
            return expanded_mask
            
        except Exception as e:
            logger.warning(f"扩展搜索区域失败: {str(e)}")
            return np.zeros((height, width), dtype=np.uint8)
    
    def _detect_general_living_subject(self, image: np.ndarray, subject_info: Dict[str, Any]) -> np.ndarray:
        """通用生物体检测"""
        try:
            height, width = image.shape[:2]
            
            # 转换为灰度图
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # 使用GrabCut算法进行前景分割
            mask = np.zeros((height, width), np.uint8)
            bgd_model = np.zeros((1, 65), np.float64)
            fgd_model = np.zeros((1, 65), np.float64)
            
            # 定义矩形区域（假设主体在中心）
            rect = (width//8, height//8, width*3//4, height*3//4)
            
            cv2.grabCut(image, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
            
            # 创建掩码
            mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
            
            return mask2 * 255
            
        except Exception as e:
            logger.warning(f"通用生物体检测失败: {str(e)}")
            return self._fallback_mask(image)
    
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
        """专门针对熊猫的颜色增强算法"""
        try:
            # 转换为HSV色彩空间
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            # 增强对比度
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # 使用CLAHE增强亮度通道
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            
            # 重新合并LAB通道
            lab = cv2.merge([l, a, b])
            enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            
            # 特别优化熊猫的黑白区域
            enhanced = self._optimize_panda_black_white_areas(enhanced)
            
            # 优化下巴/颈部区域的颜色
            enhanced = self._enhance_chin_neck_colors(enhanced)
            
            return enhanced
            
        except Exception as e:
            logger.warning(f"熊猫颜色增强失败: {str(e)}")
            return image
    
    def _optimize_panda_black_white_areas(self, image: np.ndarray) -> np.ndarray:
        """优化熊猫黑白区域的颜色"""
        try:
            # 转换为灰度图
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # 检测黑色区域（熊猫的眼睛、耳朵、身体）
            black_mask = gray < 80
            
            # 检测白色区域（熊猫的面部、身体）
            white_mask = gray > 180
            
            # 增强黑色区域的深度
            if np.sum(black_mask) > 0:
                # 在黑色区域应用更深的黑色
                image[black_mask] = [0, 0, 0]
            
            # 增强白色区域的亮度
            if np.sum(white_mask) > 0:
                # 在白色区域应用更亮的白色
                image[white_mask] = [255, 255, 255]
            
            return image
            
        except Exception as e:
            logger.warning(f"熊猫黑白区域优化失败: {str(e)}")
            return image
    
    def _enhance_chin_neck_colors(self, image: np.ndarray) -> np.ndarray:
        """增强下巴/颈部区域的颜色"""
        try:
            height, width = image.shape[:2]
            
            # 定义下巴/颈部区域（图像下半部分）
            chin_neck_region = image[height//2:, :]
            
            # 转换为LAB色彩空间进行颜色增强
            lab = cv2.cvtColor(chin_neck_region, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # 增强下巴/颈部区域的对比度
            clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(4, 4))
            l = clahe.apply(l)
            
            # 轻微增强饱和度
            a = cv2.add(a, 5)
            b = cv2.add(b, 5)
            
            # 重新合并通道
            lab = cv2.merge([l, a, b])
            enhanced_chin_neck = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            
            # 将增强后的区域放回原图
            image[height//2:, :] = enhanced_chin_neck
            
            return image
            
        except Exception as e:
            logger.warning(f"下巴/颈部颜色增强失败: {str(e)}")
            return image
    
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

    def _optimize_panda_boundaries(self, mask: np.ndarray) -> np.ndarray:
        """优化熊猫区域边界"""
        try:
            # 使用形态学操作优化边界
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            
            # 先进行开运算去除小噪声
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            
            # 再进行闭运算填充小孔洞
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            
            # 使用高斯模糊平滑边界
            mask = cv2.GaussianBlur(mask, (3, 3), 0)
            
            # 重新二值化
            _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
            
            return mask
            
        except Exception as e:
            logger.warning(f"熊猫边界优化失败: {str(e)}")
            return mask 