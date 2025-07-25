# feature_detector.py
# 统一特征点检测接口，支持多种特征检测算法

import numpy as np
import cv2
from typing import Any, Dict, Optional, List, Tuple

class FeatureDetector:
    def __init__(self, model_name: str = 'opencv', device: str = 'cpu', model_path: Optional[str] = None):
        self.model_name = model_name.lower()
        self.device = device
        self.model_path = model_path
        self.model = self._load_model()

    def _load_model(self):
        """加载特征检测模型"""
        if self.model_name == 'opencv':
            # 加载OpenCV的级联分类器
            try:
                face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
                return {'face': face_cascade, 'eye': eye_cascade}
            except:
                return None
        elif self.model_name == 'contour':
            return True  # 轮廓检测不需要预训练模型
        elif self.model_name == 'blob':
            return True  # Blob检测不需要预训练模型
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")

    def detect(self, image: np.ndarray, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        输入原图，输出特征点字典
        """
        if params is None:
            params = {}
        
        if self.model_name == 'opencv':
            return self._opencv_detect(image, params)
        elif self.model_name == 'contour':
            return self._contour_detect(image, params)
        elif self.model_name == 'blob':
            return self._blob_detect(image, params)
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")

    def _opencv_detect(self, image: np.ndarray, params: Dict[str, Any]) -> Dict[str, Any]:
        """OpenCV特征检测"""
        if self.model is None:
            return {}
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        features = {}
        
        # 人脸检测
        face_cascade = self.model.get('face')
        if face_cascade:
            faces = face_cascade.detectMultiScale(
                gray, 
                scaleFactor=params.get('face_scale_factor', 1.1),
                minNeighbors=params.get('face_min_neighbors', 5),
                minSize=params.get('face_min_size', (30, 30))
            )
            
            if len(faces) > 0:
                # 取最大的人脸
                largest_face = max(faces, key=lambda x: x[2] * x[3])
                x, y, w, h = largest_face
                features['face'] = {
                    'bbox': (x, y, w, h),
                    'center': (x + w//2, y + h//2),
                    'corners': [(x, y), (x+w, y), (x+w, y+h), (x, y+h)]
                }
                
                # 在人脸区域内检测眼睛
                face_roi = gray[y:y+h, x:x+w]
                eye_cascade = self.model.get('eye')
                if eye_cascade:
                    eyes = eye_cascade.detectMultiScale(
                        face_roi,
                        scaleFactor=params.get('eye_scale_factor', 1.1),
                        minNeighbors=params.get('eye_min_neighbors', 3)
                    )
                    
                    if len(eyes) >= 2:
                        # 按x坐标排序，区分左右眼
                        eyes = sorted(eyes, key=lambda e: e[0])
                        features['eye_left'] = {
                            'bbox': (x + eyes[0][0], y + eyes[0][1], eyes[0][2], eyes[0][3]),
                            'center': (x + eyes[0][0] + eyes[0][2]//2, y + eyes[0][1] + eyes[0][3]//2)
                        }
                        features['eye_right'] = {
                            'bbox': (x + eyes[1][0], y + eyes[1][1], eyes[1][2], eyes[1][3]),
                            'center': (x + eyes[1][0] + eyes[1][2]//2, y + eyes[1][1] + eyes[1][3]//2)
                        }
        
        return features

    def _contour_detect(self, image: np.ndarray, params: Dict[str, Any]) -> Dict[str, Any]:
        """基于轮廓的特征检测"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        features = {}
        
        # 边缘检测
        edges = cv2.Canny(gray, 50, 150)
        
        # 查找轮廓
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # 按面积排序
            contours = sorted(contours, key=cv2.contourArea, reverse=True)
            
            # 分析前几个最大的轮廓
            for i, contour in enumerate(contours[:5]):
                # 计算轮廓特征
                area = cv2.contourArea(contour)
                perimeter = cv2.arcLength(contour, True)
                
                # 计算边界框
                x, y, w, h = cv2.boundingRect(contour)
                
                # 计算质心
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                else:
                    cx, cy = x + w//2, y + h//2
                
                features[f'contour_{i}'] = {
                    'bbox': (x, y, w, h),
                    'center': (cx, cy),
                    'area': area,
                    'perimeter': perimeter,
                    'contour': contour.tolist()
                }
        
        return features

    def _blob_detect(self, image: np.ndarray, params: Dict[str, Any]) -> Dict[str, Any]:
        """Blob特征检测"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        features = {}
        
        # 设置Blob检测参数
        params_blob = cv2.SimpleBlobDetector_Params()
        params_blob.minThreshold = params.get('min_threshold', 10)
        params_blob.maxThreshold = params.get('max_threshold', 200)
        params_blob.filterByArea = True
        params_blob.minArea = params.get('min_area', 100)
        params_blob.maxArea = params.get('max_area', 10000)
        params_blob.filterByCircularity = True
        params_blob.minCircularity = params.get('min_circularity', 0.1)
        params_blob.filterByConvexity = True
        params_blob.minConvexity = params.get('min_convexity', 0.87)
        params_blob.filterByInertia = True
        params_blob.minInertiaRatio = params.get('min_inertia_ratio', 0.01)
        
        # 创建检测器
        detector = cv2.SimpleBlobDetector_create(params_blob)
        
        # 检测blobs
        keypoints = detector.detect(gray)
        
        # 转换为特征字典
        for i, kp in enumerate(keypoints):
            features[f'blob_{i}'] = {
                'center': (int(kp.pt[0]), int(kp.pt[1])),
                'size': kp.size,
                'angle': kp.angle,
                'response': kp.response,
                'octave': kp.octave
            }
        
        return features

# 示例用法
if __name__ == "__main__":
    import cv2
    img = cv2.imread("test.jpg")
    if img is not None:
        detector = FeatureDetector(model_name='opencv', device='cpu')
        features = detector.detect(img)
        print("检测到的特征:", features) 