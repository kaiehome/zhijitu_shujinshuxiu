import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from sklearn.cluster import KMeans
import logging

class ProfessionalRecognitionGenerator:
    """
    专业真实识别图生成器
    专门用于生成高饱和度、强对比度、清晰边缘的专业识别图效果
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def generate_professional_recognition(self, image_path: str, output_path: str = None) -> str:
        """
        生成专业真实识别图
        
        Args:
            image_path: 输入图像路径
            output_path: 输出图像路径，如果为None则自动生成
            
        Returns:
            输出图像路径
        """
        try:
            # 读取图像
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"无法读取图像: {image_path}")
            
            self.logger.info(f"开始处理图像: {image_path}")
            
            # 转换为RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # 应用专业识别图处理流程
            processed_image = self._apply_professional_recognition_style(image_rgb)
            
            # 保存结果
            if output_path is None:
                output_path = image_path.replace('.', '_professional_recognition.')
            
            processed_image_bgr = cv2.cvtColor(processed_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(output_path, processed_image_bgr)
            
            self.logger.info(f"专业识别图已保存: {output_path}")
            return output_path
            
        except Exception as e:
            self.logger.error(f"生成专业识别图时出错: {str(e)}")
            raise
    
    def _apply_professional_recognition_style(self, image: np.ndarray) -> np.ndarray:
        """
        应用专业识别图风格处理
        
        Args:
            image: RGB图像数组
            
        Returns:
            处理后的图像
        """
        # 1. 图像预处理 - 增强基础质量
        image = self._preprocess_image(image)
        
        # 2. 颜色增强 - 高饱和度处理
        image = self._enhance_colors_extreme(image)
        
        # 3. 对比度增强 - 强对比度处理
        image = self._enhance_contrast_extreme(image)
        
        # 4. 边缘增强 - 清晰边缘处理
        image = self._enhance_edges_sharp(image)
        
        # 5. 色彩层次优化 - 丰富色彩层次
        image = self._optimize_color_layers(image)
        
        # 6. 最终优化 - 专业质感处理
        image = self._final_professional_touch(image)
        
        return image
    
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """图像预处理"""
        # 调整图像大小以确保处理质量
        height, width = image.shape[:2]
        if width > 1024:
            scale = 1024 / width
            new_width = int(width * scale)
            new_height = int(height * scale)
            image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
        
        # 轻微锐化
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        image = cv2.filter2D(image, -1, kernel)
        
        return image
    
    def _enhance_colors_extreme(self, image: np.ndarray) -> np.ndarray:
        """极强颜色增强"""
        # 转换为HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        # 极强饱和度增强
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 2.5, 0, 255)
        
        # 增强亮度对比
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 1.3, 0, 255)
        
        # 转回RGB
        enhanced = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        
        # 使用PIL进行额外颜色增强
        pil_image = Image.fromarray(enhanced)
        enhancer = ImageEnhance.Color(pil_image)
        enhanced_pil = enhancer.enhance(2.0)  # 极强颜色增强
        
        return np.array(enhanced_pil)
    
    def _enhance_contrast_extreme(self, image: np.ndarray) -> np.ndarray:
        """极强对比度增强"""
        # 使用PIL进行对比度增强
        pil_image = Image.fromarray(image)
        enhancer = ImageEnhance.Contrast(pil_image)
        enhanced = enhancer.enhance(2.5)  # 极强对比度
        
        # 亮度调整
        brightness_enhancer = ImageEnhance.Brightness(enhanced)
        enhanced = brightness_enhancer.enhance(1.2)
        
        return np.array(enhanced)
    
    def _enhance_edges_sharp(self, image: np.ndarray) -> np.ndarray:
        """清晰边缘增强"""
        # 使用Canny边缘检测
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        # 膨胀边缘
        kernel = np.ones((2, 2), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
        
        # 将边缘叠加到原图
        edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        enhanced = cv2.addWeighted(image, 0.8, edges_rgb, 0.2, 0)
        
        # 使用PIL进行锐化
        pil_image = Image.fromarray(enhanced)
        sharpened = pil_image.filter(ImageFilter.SHARPEN)
        
        return np.array(sharpened)
    
    def _optimize_color_layers(self, image: np.ndarray) -> np.ndarray:
        """优化色彩层次"""
        # 使用K-means进行颜色聚类，增加色彩层次
        pixels = image.reshape(-1, 3)
        kmeans = KMeans(n_clusters=32, random_state=42, n_init=10)
        labels = kmeans.fit_predict(pixels)
        clustered = kmeans.cluster_centers_[labels].reshape(image.shape).astype(np.uint8)
        
        # 混合原图和聚类结果
        enhanced = cv2.addWeighted(image, 0.6, clustered, 0.4, 0)
        
        return enhanced
    
    def _final_professional_touch(self, image: np.ndarray) -> np.ndarray:
        """最终专业质感处理"""
        # 最终锐化
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(image, -1, kernel)
        
        # 轻微降噪
        denoised = cv2.bilateralFilter(sharpened, 9, 75, 75)
        
        # 最终颜色微调
        hsv = cv2.cvtColor(denoised, cv2.COLOR_RGB2HSV)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.1, 0, 255)  # 轻微增加饱和度
        final = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        
        return final

def test_professional_recognition():
    """测试专业识别图生成"""
    generator = ProfessionalRecognitionGenerator()
    
    # 测试图像路径
    test_image = "uploads/250625_162043.jpg"
    
    try:
        output_path = generator.generate_professional_recognition(test_image)
        print(f"专业识别图生成成功: {output_path}")
    except Exception as e:
        print(f"生成失败: {str(e)}")

if __name__ == "__main__":
    test_professional_recognition() 