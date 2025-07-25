import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from sklearn.cluster import KMeans
import logging
import time
from typing import Dict, List, Tuple, Optional

class LocalAIGenerator:
    """
    本地AI增强专业识别图生成器
    使用本地图像处理算法模拟AI分析能力，无需外部API
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def analyze_image_intelligently(self, image_path: str) -> Dict:
        """
        智能分析图像内容，模拟AI分析能力
        """
        try:
            # 读取图像
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"无法读取图像: {image_path}")
            
            # 1. 对象识别分析
            objects = self._detect_objects(image)
            
            # 2. 颜色分析
            color_analysis = self._analyze_colors(image)
            
            # 3. 区域分析
            enhancement_areas = self._identify_enhancement_areas(image)
            
            # 4. 风格特征分析
            style_features = self._analyze_style_features(image)
            
            # 5. 生成处理参数
            processing_params = self._generate_processing_params(
                objects, color_analysis, enhancement_areas, style_features
            )
            
            analysis = {
                "objects": objects,
                "current_colors": color_analysis,
                "enhancement_areas": enhancement_areas,
                "target_style": style_features,
                "processing_params": processing_params
            }
            
            self.logger.info(f"智能分析完成: {analysis}")
            return analysis
            
        except Exception as e:
            self.logger.error(f"智能分析失败: {str(e)}")
            return self._get_default_analysis()
    
    def _detect_objects(self, image: np.ndarray) -> List[str]:
        """检测图像中的主要对象"""
        objects = []
        
        # 转换为灰度图
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 检测熊猫特征
        if self._detect_panda_features(image):
            objects.append("panda")
        
        # 检测花朵特征
        if self._detect_flower_features(image):
            objects.append("flowers")
        
        # 检测叶子特征
        if self._detect_leaf_features(image):
            objects.append("leaves")
        
        # 如果没有检测到特定对象，添加通用对象
        if not objects:
            objects.append("general_object")
        
        return objects
    
    def _detect_panda_features(self, image: np.ndarray) -> bool:
        """检测熊猫特征"""
        try:
            # 转换为HSV色彩空间
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            # 检测黑色区域（熊猫的眼睛、耳朵、身体）
            black_mask = cv2.inRange(hsv, (0, 0, 0), (180, 255, 50))
            
            # 检测白色区域（熊猫的面部、身体）
            white_mask = cv2.inRange(hsv, (0, 0, 200), (180, 30, 255))
            
            # 计算黑白区域比例
            total_pixels = image.shape[0] * image.shape[1]
            black_ratio = np.sum(black_mask > 0) / total_pixels
            white_ratio = np.sum(white_mask > 0) / total_pixels
            
            # 熊猫特征：黑白区域比例适中
            return 0.1 < black_ratio < 0.4 and 0.2 < white_ratio < 0.6
            
        except Exception as e:
            self.logger.warning(f"熊猫特征检测失败: {str(e)}")
            return False
    
    def _detect_flower_features(self, image: np.ndarray) -> bool:
        """检测花朵特征"""
        try:
            # 转换为HSV色彩空间
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            # 检测红色/粉色区域（花朵）
            red_mask1 = cv2.inRange(hsv, (0, 100, 100), (10, 255, 255))
            red_mask2 = cv2.inRange(hsv, (170, 100, 100), (180, 255, 255))
            red_mask = cv2.bitwise_or(red_mask1, red_mask2)
            
            # 检测黄色区域（花朵中心）
            yellow_mask = cv2.inRange(hsv, (20, 100, 100), (30, 255, 255))
            
            # 计算花朵区域比例
            total_pixels = image.shape[0] * image.shape[1]
            flower_ratio = (np.sum(red_mask > 0) + np.sum(yellow_mask > 0)) / total_pixels
            
            return flower_ratio > 0.05
            
        except Exception as e:
            self.logger.warning(f"花朵特征检测失败: {str(e)}")
            return False
    
    def _detect_leaf_features(self, image: np.ndarray) -> bool:
        """检测叶子特征"""
        try:
            # 转换为HSV色彩空间
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            # 检测绿色区域（叶子）
            green_mask = cv2.inRange(hsv, (35, 50, 50), (85, 255, 255))
            
            # 计算叶子区域比例
            total_pixels = image.shape[0] * image.shape[1]
            leaf_ratio = np.sum(green_mask > 0) / total_pixels
            
            return leaf_ratio > 0.1
            
        except Exception as e:
            self.logger.warning(f"叶子特征检测失败: {str(e)}")
            return False
    
    def _analyze_colors(self, image: np.ndarray) -> Dict:
        """分析当前颜色特征"""
        try:
            # 转换为HSV色彩空间
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            # 计算饱和度统计
            saturation = hsv[:, :, 1]
            avg_saturation = np.mean(saturation)
            
            # 计算对比度
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            contrast = np.std(gray)
            
            # 计算亮度
            brightness = np.mean(gray)
            
            # 判断颜色特征
            if avg_saturation < 80:
                saturation_level = "low"
            elif avg_saturation < 150:
                saturation_level = "medium"
            else:
                saturation_level = "high"
            
            if contrast < 30:
                contrast_level = "low"
            elif contrast < 60:
                contrast_level = "medium"
            else:
                contrast_level = "high"
            
            if brightness < 80:
                brightness_level = "dark"
            elif brightness < 150:
                brightness_level = "normal"
            else:
                brightness_level = "bright"
            
            return {
                "saturation": saturation_level,
                "contrast": contrast_level,
                "brightness": brightness_level,
                "avg_saturation": float(avg_saturation),
                "contrast_value": float(contrast),
                "brightness_value": float(brightness)
            }
            
        except Exception as e:
            self.logger.warning(f"颜色分析失败: {str(e)}")
            return {"saturation": "medium", "contrast": "medium", "brightness": "normal"}
    
    def _identify_enhancement_areas(self, image: np.ndarray) -> List[str]:
        """识别需要增强的区域"""
        areas = []
        
        try:
            # 检测熊猫相关区域
            if self._detect_panda_features(image):
                areas.extend(["panda_fur", "panda_eyes", "panda_ears"])
            
            # 检测背景区域
            if self._detect_flower_features(image):
                areas.append("background_flowers")
            
            if self._detect_leaf_features(image):
                areas.append("background_leaves")
            
            # 添加通用增强区域
            areas.extend(["overall_contrast", "edge_sharpness"])
            
        except Exception as e:
            self.logger.warning(f"区域识别失败: {str(e)}")
            areas = ["general_enhancement"]
        
        return areas
    
    def _analyze_style_features(self, image: np.ndarray) -> Dict:
        """分析目标风格特征"""
        try:
            # 分析当前图像的特征
            color_analysis = self._analyze_colors(image)
            
            # 根据当前特征确定目标风格参数
            current_saturation = color_analysis.get("avg_saturation", 100)
            current_contrast = color_analysis.get("contrast_value", 40)
            
            # 计算需要的增强倍数
            saturation_boost = max(3.0, 200 / max(current_saturation, 1))
            contrast_boost = max(3.0, 100 / max(current_contrast, 1))
            
            return {
                "saturation_boost": min(saturation_boost, 5.0),
                "contrast_boost": min(contrast_boost, 5.0),
                "color_clusters": 32,
                "edge_sharpness": "extreme",
                "pixelation_level": "high"
            }
            
        except Exception as e:
            self.logger.warning(f"风格特征分析失败: {str(e)}")
            return {
                "saturation_boost": 4.0,
                "contrast_boost": 4.0,
                "color_clusters": 32,
                "edge_sharpness": "extreme"
            }
    
    def _generate_processing_params(self, objects: List[str], color_analysis: Dict, 
                                  enhancement_areas: List[str], style_features: Dict) -> Dict:
        """根据分析结果生成处理参数"""
        try:
            # 基础参数
            params = {
                "hsv_saturation_factor": style_features.get("saturation_boost", 4.0),
                "contrast_factor": style_features.get("contrast_boost", 4.0),
                "brightness_factor": 1.2,
                "sharpness_factor": 3.0,
                "color_quantization": style_features.get("color_clusters", 32),
                "edge_enhancement": True,
                "morphological_operations": True
            }
            
            # 根据对象类型调整参数
            if "panda" in objects:
                params.update({
                    "hsv_saturation_factor": min(params["hsv_saturation_factor"] * 1.2, 5.0),
                    "contrast_factor": min(params["contrast_factor"] * 1.3, 5.0),
                    "black_white_enhancement": True
                })
            
            if "flowers" in objects:
                params.update({
                    "hsv_saturation_factor": min(params["hsv_saturation_factor"] * 1.1, 5.0),
                    "color_quantization": max(params["color_quantization"] - 8, 16)
                })
            
            # 根据当前颜色特征调整参数
            current_saturation = color_analysis.get("saturation", "medium")
            if current_saturation == "low":
                params["hsv_saturation_factor"] *= 1.5
            elif current_saturation == "high":
                params["hsv_saturation_factor"] *= 0.8
            
            current_contrast = color_analysis.get("contrast", "medium")
            if current_contrast == "low":
                params["contrast_factor"] *= 1.5
            elif current_contrast == "high":
                params["contrast_factor"] *= 0.8
            
            return params
            
        except Exception as e:
            self.logger.warning(f"参数生成失败: {str(e)}")
            return self._get_default_analysis()["processing_params"]
    
    def _get_default_analysis(self) -> Dict:
        """获取默认分析结果"""
        return {
            "objects": ["general_object"],
            "current_colors": {
                "saturation": "medium",
                "contrast": "medium", 
                "brightness": "normal"
            },
            "enhancement_areas": ["general_enhancement"],
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
    
    def generate_local_ai_enhanced_image(self, image_path: str, output_path: str = None) -> str:
        """
        生成本地AI增强的专业识别图
        """
        try:
            self.logger.info(f"开始本地AI增强图像处理: {image_path}")
            
            # 1. 智能分析图像
            analysis = self.analyze_image_intelligently(image_path)
            self.logger.info(f"智能分析完成: {analysis}")
            
            # 2. 根据分析结果进行智能处理
            processed_image = self._apply_local_ai_enhanced_processing(image_path, analysis)
            
            # 3. 保存结果
            if output_path is None:
                timestamp = int(time.time() * 1000)
                filename = f"{timestamp}_local_ai_enhanced.jpg"
                output_path = f"uploads/{filename}"
            
            cv2.imwrite(output_path, processed_image)
            self.logger.info(f"本地AI增强图像生成完成: {output_path}")
            
            return output_path
            
        except Exception as e:
            self.logger.error(f"本地AI增强处理失败: {str(e)}")
            raise
    
    def _apply_local_ai_enhanced_processing(self, image_path: str, analysis: Dict) -> np.ndarray:
        """
        根据智能分析结果应用本地AI增强处理
        """
        # 读取图像
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"无法读取图像: {image_path}")
        
        # 获取处理参数
        params = analysis.get("processing_params", self._get_default_analysis()["processing_params"])
        objects = analysis.get("objects", [])
        
        # 1. 智能颜色增强
        image = self._local_ai_color_processing(image, params, objects)
        
        # 2. 智能对比度增强
        image = self._local_ai_contrast_processing(image, params)
        
        # 3. 智能边缘增强
        image = self._local_ai_edge_processing(image, params)
        
        # 4. 智能颜色量化
        image = self._local_ai_quantization(image, params)
        
        # 5. 智能形态学处理
        image = self._local_ai_morphological_processing(image, params)
        
        # 6. 特殊对象处理
        if "panda" in objects:
            image = self._enhance_panda_specifically(image, params)
        
        return image
    
    def _local_ai_color_processing(self, image: np.ndarray, params: Dict, objects: List[str]) -> np.ndarray:
        """本地AI增强的颜色处理"""
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
    
    def _local_ai_contrast_processing(self, image: np.ndarray, params: Dict) -> np.ndarray:
        """本地AI增强的对比度处理"""
        # 转换到LAB色彩空间
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        
        # 分离通道
        l, a, b = cv2.split(lab)
        
        # 应用CLAHE（对比度受限的自适应直方图均衡）
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        
        # 增强对比度
        contrast_factor = params.get("contrast_factor", 4.0)
        l = np.clip(l * contrast_factor, 0, 255).astype(np.uint8)
        
        # 确保所有通道具有相同的数据类型
        a = a.astype(np.uint8)
        b = b.astype(np.uint8)
        
        # 合并通道
        enhanced_lab = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        
        return enhanced
    
    def _local_ai_edge_processing(self, image: np.ndarray, params: Dict) -> np.ndarray:
        """本地AI增强的边缘处理"""
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
    
    def _local_ai_quantization(self, image: np.ndarray, params: Dict) -> np.ndarray:
        """本地AI增强的颜色量化"""
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
    
    def _local_ai_morphological_processing(self, image: np.ndarray, params: Dict) -> np.ndarray:
        """本地AI增强的形态学处理"""
        # 创建核
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        
        # 应用形态学操作
        enhanced = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
        enhanced = cv2.morphologyEx(enhanced, cv2.MORPH_OPEN, kernel)
        
        # 边缘保持滤波
        enhanced = cv2.bilateralFilter(enhanced, 9, 75, 75)
        
        return enhanced
    
    def _enhance_panda_specifically(self, image: np.ndarray, params: Dict) -> np.ndarray:
        """专门针对熊猫的增强处理"""
        try:
            # 转换为HSV色彩空间
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            # 检测熊猫的黑色区域
            black_mask = cv2.inRange(hsv, (0, 0, 0), (180, 255, 50))
            
            # 检测熊猫的白色区域
            white_mask = cv2.inRange(hsv, (0, 0, 200), (180, 30, 255))
            
            # 增强黑色区域的深度
            if np.sum(black_mask) > 0:
                # 在黑色区域应用更深的黑色
                image[black_mask > 0] = [0, 0, 0]
            
            # 增强白色区域的亮度
            if np.sum(white_mask) > 0:
                # 在白色区域应用更亮的白色
                image[white_mask > 0] = [255, 255, 255]
            
            return image
            
        except Exception as e:
            self.logger.warning(f"熊猫专门增强失败: {str(e)}")
            return image

def test_local_ai_generator():
    """测试本地AI生成器"""
    generator = LocalAIGenerator()
    
    # 测试图像路径
    test_image = "uploads/250625_162043.jpg"
    
    try:
        output_path = generator.generate_local_ai_enhanced_image(test_image)
        print(f"本地AI增强图像生成成功: {output_path}")
    except Exception as e:
        print(f"生成失败: {str(e)}")

if __name__ == "__main__":
    test_local_ai_generator() 