import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from sklearn.cluster import KMeans
import requests
import json
import base64
import io
import logging
import time
from typing import Dict, List, Tuple, Optional

class AIEnhancedProcessor:
    """
    AI增强的专业识别图处理器
    结合大模型API和本地图像处理算法
    """
    
    def __init__(self, api_key: str = None, model_endpoint: str = None):
        self.logger = logging.getLogger(__name__)
        self.api_key = api_key
        self.model_endpoint = model_endpoint or "https://api.openai.com/v1/chat/completions"
        
    def encode_image_to_base64(self, image_path: str) -> str:
        """将图像编码为base64字符串"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    def analyze_image_with_ai(self, image_path: str) -> Dict:
        """
        使用AI分析图像内容，获取智能处理建议
        """
        try:
            # 编码图像
            base64_image = self.encode_image_to_base64(image_path)
            
            # 构建AI分析请求
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            
            payload = {
                "model": "gpt-4-vision-preview",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": """请分析这张图像，并提供专业识别图转换建议：
                                
1. 主要对象识别（如熊猫、花朵等）
2. 当前颜色分析（饱和度、对比度、色调）
3. 需要增强的区域（如毛发、眼睛、背景）
4. 目标风格特征（高饱和度、强对比度、像素化效果）
5. 具体的处理参数建议

请以JSON格式返回分析结果。"""
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                "max_tokens": 1000
            }
            
            # 发送AI分析请求
            response = requests.post(self.model_endpoint, headers=headers, json=payload)
            
            if response.status_code == 200:
                result = response.json()
                content = result['choices'][0]['message']['content']
                
                # 尝试解析JSON响应
                try:
                    analysis = json.loads(content)
                    return analysis
                except json.JSONDecodeError:
                    # 如果不是JSON格式，返回文本分析
                    return {"analysis": content, "type": "text"}
            else:
                self.logger.warning(f"AI分析失败: {response.status_code}")
                return self._get_default_analysis()
                
        except Exception as e:
            self.logger.error(f"AI分析出错: {str(e)}")
            return self._get_default_analysis()
    
    def _get_default_analysis(self) -> Dict:
        """获取默认分析结果"""
        return {
            "objects": ["panda", "flowers", "leaves"],
            "current_colors": {
                "saturation": "medium",
                "contrast": "medium", 
                "brightness": "normal"
            },
            "enhancement_areas": [
                "panda_fur",
                "panda_eyes", 
                "background_flowers"
            ],
            "target_style": {
                "saturation_boost": 4.0,
                "contrast_boost": 4.0,
                "color_clusters": 32,
                "edge_sharpness": "extreme"
            },
            "processing_params": {
                "hsv_saturation_factor": 4.0,
                "contrast_factor": 4.0,
                "brightness_factor": 1.2,
                "sharpness_factor": 3.0,
                "color_quantization": 32,
                "edge_enhancement": True,
                "morphological_operations": True
            }
        }
    
    def generate_ai_enhanced_image(self, image_path: str, output_path: str = None) -> str:
        """
        生成AI增强的专业识别图
        """
        try:
            self.logger.info(f"开始AI增强图像处理: {image_path}")
            
            # 1. AI分析图像
            analysis = self.analyze_image_with_ai(image_path)
            self.logger.info(f"AI分析完成: {analysis}")
            
            # 2. 根据AI分析结果进行智能处理
            processed_image = self._apply_ai_enhanced_processing(image_path, analysis)
            
            # 3. 保存结果
            if output_path is None:
                timestamp = int(time.time() * 1000)
                filename = f"{timestamp}_ai_enhanced.jpg"
                output_path = f"uploads/{filename}"
            
            cv2.imwrite(output_path, processed_image)
            self.logger.info(f"AI增强图像生成完成: {output_path}")
            
            return output_path
            
        except Exception as e:
            self.logger.error(f"AI增强处理失败: {str(e)}")
            raise
    
    def _apply_ai_enhanced_processing(self, image_path: str, analysis: Dict) -> np.ndarray:
        """
        根据AI分析结果应用智能处理
        """
        # 读取图像
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"无法读取图像: {image_path}")
        
        # 获取处理参数
        params = analysis.get("processing_params", self._get_default_analysis()["processing_params"])
        
        # 1. 智能颜色增强
        image = self._ai_enhanced_color_processing(image, params)
        
        # 2. 智能对比度增强
        image = self._ai_enhanced_contrast_processing(image, params)
        
        # 3. 智能边缘增强
        image = self._ai_enhanced_edge_processing(image, params)
        
        # 4. 智能颜色量化
        image = self._ai_enhanced_quantization(image, params)
        
        # 5. 智能形态学处理
        image = self._ai_enhanced_morphological_processing(image, params)
        
        return image
    
    def _ai_enhanced_color_processing(self, image: np.ndarray, params: Dict) -> np.ndarray:
        """AI增强的颜色处理"""
        # 转换到HSV色彩空间
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # 根据AI建议调整饱和度
        saturation_factor = params.get("hsv_saturation_factor", 4.0)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation_factor, 0, 255)
        
        # 智能色调调整
        hue_shift = params.get("hue_shift", 0)
        hsv[:, :, 0] = (hsv[:, :, 0] + hue_shift) % 180
        
        # 转换回BGR
        enhanced = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        return enhanced
    
    def _ai_enhanced_contrast_processing(self, image: np.ndarray, params: Dict) -> np.ndarray:
        """AI增强的对比度处理"""
        # 转换到LAB色彩空间
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        
        # 分离通道
        l, a, b = cv2.split(lab)
        
        # 应用CLAHE（对比度受限的自适应直方图均衡）
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        
        # 增强对比度
        contrast_factor = params.get("contrast_factor", 4.0)
        l = np.clip(l * contrast_factor, 0, 255)
        
        # 合并通道
        enhanced_lab = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        
        return enhanced
    
    def _ai_enhanced_edge_processing(self, image: np.ndarray, params: Dict) -> np.ndarray:
        """AI增强的边缘处理"""
        # 转换为PIL图像
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # 多重锐化
        sharpness_factor = params.get("sharpness_factor", 3.0)
        for i in range(int(sharpness_factor)):
            pil_image = pil_image.filter(ImageFilter.SHARPEN)
            pil_image = pil_image.filter(ImageFilter.EDGE_ENHANCE)
        
        # 转换回OpenCV格式
        enhanced = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        
        return enhanced
    
    def _ai_enhanced_quantization(self, image: np.ndarray, params: Dict) -> np.ndarray:
        """AI增强的颜色量化"""
        # 重塑图像
        data = image.reshape((-1, 3))
        data = np.float32(data)
        
        # K-means聚类
        n_colors = params.get("color_quantization", 32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        _, labels, centers = cv2.kmeans(data, n_colors, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        
        # 重建图像
        centers = np.uint8(centers)
        quantized = centers[labels.flatten()]
        quantized = quantized.reshape(image.shape)
        
        return quantized
    
    def _ai_enhanced_morphological_processing(self, image: np.ndarray, params: Dict) -> np.ndarray:
        """AI增强的形态学处理"""
        # 创建核
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        
        # 应用形态学操作
        enhanced = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
        enhanced = cv2.morphologyEx(enhanced, cv2.MORPH_OPEN, kernel)
        
        # 边缘保持滤波
        enhanced = cv2.bilateralFilter(enhanced, 9, 75, 75)
        
        return enhanced

def test_ai_enhanced_processor():
    """测试AI增强处理器"""
    # 注意：需要设置API密钥才能使用AI功能
    processor = AIEnhancedProcessor()
    
    # 测试图像路径
    test_image = "uploads/250625_162043.jpg"
    
    try:
        # 使用默认分析进行测试
        output_path = processor.generate_ai_enhanced_image(test_image)
        print(f"AI增强图像生成成功: {output_path}")
    except Exception as e:
        print(f"生成失败: {str(e)}")

if __name__ == "__main__":
    test_ai_enhanced_processor() 