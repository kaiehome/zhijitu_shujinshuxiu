import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from sklearn.cluster import KMeans
import logging

class UltimateProfessionalGenerator:
    """
    终极专业识别图生成器
    专门用于生成高饱和度、强对比度、像素化效果和锐利边缘的专业识别图
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def generate_ultimate_professional(self, image_path: str, output_path: str = None) -> str:
        """
        生成终极专业识别图
        
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
            
            # 应用终极专业识别图处理流程
            processed_image = self._apply_ultimate_professional_style(image_rgb)
            
            # 保存结果
            if output_path is None:
                output_path = image_path.replace('.', '_ultimate_professional.')
            
            processed_image_bgr = cv2.cvtColor(processed_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(output_path, processed_image_bgr)
            
            self.logger.info(f"终极专业识别图已保存: {output_path}")
            return output_path
            
        except Exception as e:
            self.logger.error(f"生成终极专业识别图时出错: {str(e)}")
            raise
    
    def _apply_ultimate_professional_style(self, image: np.ndarray) -> np.ndarray:
        """
        应用终极专业识别图风格处理
        
        Args:
            image: RGB图像数组
            
        Returns:
            处理后的图像
        """
        # 1. 图像预处理和尺寸优化
        image = self._preprocess_and_resize(image)
        
        # 2. 极强颜色增强 - 高饱和度处理
        image = self._extreme_color_enhancement(image)
        
        # 3. 极强对比度增强 - 强对比度处理
        image = self._extreme_contrast_enhancement(image)
        
        # 4. 像素化效果处理 - 块状效果
        image = self._pixelation_effect(image)
        
        # 5. 智能色彩层次优化 - 丰富色彩层次
        image = self._intelligent_color_layering(image)
        
        # 6. 锐利边缘增强 - 清晰边缘处理
        image = self._sharp_edge_enhancement(image)
        
        # 7. 最终优化和锐化
        image = self._final_optimization_and_sharpening(image)
        
        return image
    
    def _preprocess_and_resize(self, image: np.ndarray) -> np.ndarray:
        """图像预处理和尺寸优化"""
        # 调整图像大小以确保处理质量
        height, width = image.shape[:2]
        if width > 1000:
            scale = 1000 / width
            new_width = int(width * scale)
            new_height = int(height * scale)
            image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
        
        # 轻微降噪
        image = cv2.bilateralFilter(image, 9, 75, 75)
        
        return image
    
    def _extreme_color_enhancement(self, image: np.ndarray) -> np.ndarray:
        """极强颜色增强"""
        # 转换为HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        # 极强饱和度增强
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 4.0, 0, 255)  # 4倍饱和度
        
        # 增强亮度对比
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 1.6, 0, 255)
        
        # 转回RGB
        enhanced = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        
        # 使用PIL进行额外颜色增强
        pil_image = Image.fromarray(enhanced)
        enhancer = ImageEnhance.Color(pil_image)
        enhanced_pil = enhancer.enhance(3.0)  # 极强颜色增强
        
        # 应用自动对比度
        enhanced_pil = ImageOps.autocontrast(enhanced_pil, cutoff=0)
        
        return np.array(enhanced_pil)
    
    def _extreme_contrast_enhancement(self, image: np.ndarray) -> np.ndarray:
        """极强对比度增强"""
        # 使用PIL进行对比度增强
        pil_image = Image.fromarray(image)
        enhancer = ImageEnhance.Contrast(pil_image)
        enhanced = enhancer.enhance(4.0)  # 极强对比度
        
        # 亮度调整
        brightness_enhancer = ImageEnhance.Brightness(enhanced)
        enhanced = brightness_enhancer.enhance(1.4)
        
        # 应用自动对比度
        enhanced = ImageOps.autocontrast(enhanced, cutoff=3)
        
        return np.array(enhanced)
    
    def _pixelation_effect(self, image: np.ndarray) -> np.ndarray:
        """像素化效果处理"""
        # 缩小图像
        height, width = image.shape[:2]
        small_height = height // 8
        small_width = width // 8
        
        # 缩小
        small = cv2.resize(image, (small_width, small_height), interpolation=cv2.INTER_LINEAR)
        
        # 放大回原尺寸，产生像素化效果
        pixelated = cv2.resize(small, (width, height), interpolation=cv2.INTER_NEAREST)
        
        # 混合原图和像素化效果
        enhanced = cv2.addWeighted(image, 0.3, pixelated, 0.7, 0)
        
        return enhanced
    
    def _intelligent_color_layering(self, image: np.ndarray) -> np.ndarray:
        """智能色彩层次优化"""
        # 使用K-means进行颜色聚类，增加色彩层次
        pixels = image.reshape(-1, 3)
        kmeans = KMeans(n_clusters=32, random_state=42, n_init=10)  # 减少颜色数量
        labels = kmeans.fit_predict(pixels)
        clustered = kmeans.cluster_centers_[labels].reshape(image.shape).astype(np.uint8)
        
        # 混合原图和聚类结果
        enhanced = cv2.addWeighted(image, 0.4, clustered, 0.6, 0)
        
        # 应用颜色映射增强
        enhanced = self._apply_color_mapping(enhanced)
        
        return enhanced
    
    def _apply_color_mapping(self, image: np.ndarray) -> np.ndarray:
        """应用颜色映射增强"""
        # 转换为LAB色彩空间
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        
        # 增强A和B通道（色彩通道）
        lab[:, :, 1] = np.clip(lab[:, :, 1] * 1.4, 0, 255)
        lab[:, :, 2] = np.clip(lab[:, :, 2] * 1.4, 0, 255)
        
        # 转回RGB
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
        return enhanced
    
    def _sharp_edge_enhancement(self, image: np.ndarray) -> np.ndarray:
        """锐利边缘增强"""
        # 使用Canny边缘检测
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 20, 80)
        
        # 膨胀边缘
        kernel = np.ones((2, 2), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
        
        # 将边缘叠加到原图
        edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        enhanced = cv2.addWeighted(image, 0.6, edges_rgb, 0.4, 0)
        
        # 使用PIL进行锐化
        pil_image = Image.fromarray(enhanced)
        sharpened = pil_image.filter(ImageFilter.SHARPEN)
        sharpened = sharpened.filter(ImageFilter.SHARPEN)  # 双重锐化
        sharpened = sharpened.filter(ImageFilter.SHARPEN)  # 三重锐化
        
        return np.array(sharpened)
    
    def _final_optimization_and_sharpening(self, image: np.ndarray) -> np.ndarray:
        """最终优化和锐化"""
        # 最终锐化
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(image, -1, kernel)
        
        # 使用PIL进行最终锐化
        pil_image = Image.fromarray(sharpened)
        final_sharpened = pil_image.filter(ImageFilter.SHARPEN)
        
        # 最终颜色微调
        final_array = np.array(final_sharpened)
        hsv = cv2.cvtColor(final_array, cv2.COLOR_RGB2HSV)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.2, 0, 255)  # 轻微增加饱和度
        final = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        
        # 最终对比度调整
        pil_final = Image.fromarray(final)
        contrast_enhancer = ImageEnhance.Contrast(pil_final)
        final_enhanced = contrast_enhancer.enhance(1.2)
        
        return np.array(final_enhanced)

def test_ultimate_professional():
    """测试终极专业识别图生成"""
    generator = UltimateProfessionalGenerator()
    
    # 测试图像路径
    test_image = "uploads/250625_162043.jpg"
    
    try:
        output_path = generator.generate_ultimate_professional(test_image)
        print(f"终极专业识别图生成成功: {output_path}")
    except Exception as e:
        print(f"生成失败: {str(e)}")

if __name__ == "__main__":
    test_ultimate_professional() 