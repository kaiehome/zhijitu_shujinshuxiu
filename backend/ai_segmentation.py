# ai_segmentation.py
# 统一AI分割模型推理接口，支持多种分割算法

import numpy as np
import cv2
from typing import Any, Dict, Optional
from skimage.segmentation import slic, watershed
from skimage.filters import sobel
from skimage.color import label2rgb

class AISegmenter:
    def __init__(self, model_name: str = 'grabcut', device: str = 'cpu', model_path: Optional[str] = None):
        self.model_name = model_name.lower()
        self.device = device
        self.model_path = model_path
        self.model = self._load_model()

    def _load_model(self):
        """加载分割模型（目前使用传统CV方法）"""
        if self.model_name in ['grabcut', 'watershed', 'slic', 'contour']:
            return True  # 传统方法不需要预训练模型
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")

    def segment(self, image: np.ndarray, params: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """
        输入原图，输出分割掩码（单通道0/255）
        """
        if params is None:
            params = {}
        
        if self.model_name == 'grabcut':
            return self._grabcut_segment(image, params)
        elif self.model_name == 'watershed':
            return self._watershed_segment(image, params)
        elif self.model_name == 'slic':
            return self._slic_segment(image, params)
        elif self.model_name == 'contour':
            return self._contour_segment(image, params)
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")

    def _grabcut_segment(self, image: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """GrabCut算法分割"""
        # 创建矩形ROI（自动检测或使用默认值）
        height, width = image.shape[:2]
        rect = params.get('rect', (width//8, height//8, width*3//4, height*3//4))
        
        # 创建掩码
        mask = np.zeros(image.shape[:2], np.uint8)
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)
        
        # 执行GrabCut
        cv2.grabCut(image, mask, rect, bgd_model, fgd_model, 
                   params.get('iter_count', 5), cv2.GC_INIT_WITH_RECT)
        
        # 创建掩码
        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
        return mask2 * 255

    def _watershed_segment(self, image: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """分水岭算法分割"""
        # 转换为灰度图
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 使用高斯模糊减少噪声
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # 使用Otsu's方法进行阈值分割
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # 形态学操作
        kernel = np.ones((3, 3), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
        
        # 确定背景区域
        sure_bg = cv2.dilate(opening, kernel, iterations=3)
        
        # 确定前景区域
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
        sure_fg = np.uint8(sure_fg)
        
        # 找到未知区域
        unknown = cv2.subtract(sure_bg, sure_fg)
        
        # 标记
        _, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1
        markers[unknown == 255] = 0
        
        # 应用分水岭算法
        markers = cv2.watershed(image, markers)
        
        # 创建掩码
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        mask[markers > 1] = 255
        
        return mask

    def _slic_segment(self, image: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """SLIC超像素分割"""
        # 转换为RGB格式
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # SLIC参数
        n_segments = params.get('n_segments', 100)
        compactness = params.get('compactness', 10)
        
        # 执行SLIC分割
        segments = slic(image_rgb, n_segments=n_segments, compactness=compactness, start_label=1)
        
        # 创建掩码（选择最大的几个区域）
        unique_labels = np.unique(segments)
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        
        # 选择面积最大的区域作为前景
        max_areas = []
        for label in unique_labels:
            area = np.sum(segments == label)
            max_areas.append((label, area))
        
        # 选择前3个最大的区域
        max_areas.sort(key=lambda x: x[1], reverse=True)
        for label, _ in max_areas[:3]:
            mask[segments == label] = 255
        
        return mask

    def _contour_segment(self, image: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """基于轮廓的分割"""
        # 转换为灰度图
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 边缘检测
        edges = cv2.Canny(gray, 50, 150)
        
        # 查找轮廓
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 创建掩码
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        
        # 填充最大的几个轮廓
        if contours:
            # 按面积排序
            contours = sorted(contours, key=cv2.contourArea, reverse=True)
            
            # 填充前3个最大的轮廓
            for i, contour in enumerate(contours[:3]):
                cv2.fillPoly(mask, [contour], 255)
        
        return mask

# 示例用法
if __name__ == "__main__":
    import cv2
    img = cv2.imread("test.jpg")
    if img is not None:
        segmenter = AISegmenter(model_name='grabcut', device='cpu')
        mask = segmenter.segment(img)
        cv2.imwrite("test_mask.png", mask) 