import cv2
import numpy as np
from typing import Tuple, Optional, Dict, List

class VineTrunkEnhancer:
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        
        self.default_config = {
            'green_suppression': 0.7,
            'brown_enhancement': 1.5,
            'texture_enhancement': 1.8,
            'clahe_clip_limit': 3.0,
            'clahe_grid_size': (4, 4),
            'shadow_removal': True,
            'shadow_threshold': 60,
            'edge_enhancement': True,
            'edge_strength': 0.8,
            'morphology': True,
            'vertical_kernel_size': (3, 15),
            'bilateral_filter': True,
            'bilateral_diameter': 9,
            'bilateral_sigma_color': 75,
            'bilateral_sigma_space': 75,
            'adaptive_threshold': False,
            'background_suppression': True,
            'background_contrast': 1.3,
        }
        
        self.params = {**self.default_config, **self.config}
        
        self.clahe = cv2.createCLAHE(
            clipLimit=self.params['clahe_clip_limit'],
            tileGridSize=self.params['clahe_grid_size']
        )
        
        self.vertical_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, 
            self.params['vertical_kernel_size']
        )
    
    def enhance_for_trunk_detection(self, frame: np.ndarray) -> np.ndarray:
        if frame is None or frame.size == 0:
            return frame
        
        original = frame.copy()
        
        frame = self.suppress_green(frame)
        frame = self.enhance_brown(frame)
        frame = self.enhance_local_contrast(frame)
        
        if self.params['bilateral_filter']:
            frame = self.bilateral_filter(frame)
        
        frame = self.enhance_texture(frame)
        
        if self.params['morphology']:
            frame = self.enhance_vertical_structures(frame)
        
        if self.params['shadow_removal']:
            frame = self.remove_shadows(frame)
        
        if self.params['edge_enhancement']:
            frame = self.enhance_edges(frame)
        
        if self.params['background_suppression']:
            frame = self.suppress_background(frame, original)
        
        return frame
    
    def suppress_green(self, frame: np.ndarray) -> np.ndarray:
        b, g, r = cv2.split(frame)
        
        g_suppressed = np.clip(g * self.params['green_suppression'], 0, 255).astype(np.uint8)
        r_enhanced = np.clip(r * 1.1, 0, 255).astype(np.uint8)
        b_enhanced = np.clip(b * 1.1, 0, 255).astype(np.uint8)
        
        return cv2.merge([b_enhanced, g_suppressed, r_enhanced])
    
    def enhance_brown(self, frame: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        lower_brown = np.array([5, 50, 50])
        upper_brown = np.array([30, 255, 255])
        
        brown_mask = cv2.inRange(hsv, lower_brown, upper_brown)
        
        hsv[:, :, 1] = np.where(brown_mask > 0, 
                                 np.clip(hsv[:, :, 1] * self.params['brown_enhancement'], 0, 255),
                                 hsv[:, :, 1])
        
        hsv[:, :, 2] = np.where(brown_mask > 0,
                                 np.clip(hsv[:, :, 2] * 1.2, 0, 255),
                                 hsv[:, :, 2])
        
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    def enhance_local_contrast(self, frame: np.ndarray) -> np.ndarray:
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        l_enhanced = self.clahe.apply(l)
        a_enhanced = np.clip(a * 1.2, 0, 255).astype(np.uint8)
        b_enhanced = np.clip(b * 1.2, 0, 255).astype(np.uint8)
        
        lab_enhanced = cv2.merge([l_enhanced, a_enhanced, b_enhanced])
        return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
    
    def bilateral_filter(self, frame: np.ndarray) -> np.ndarray:
        return cv2.bilateralFilter(
            frame,
            self.params['bilateral_diameter'],
            self.params['bilateral_sigma_color'],
            self.params['bilateral_sigma_space']
        )
    
    def enhance_texture(self, frame: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        texture = cv2.subtract(gray, blur)
        
        texture_enhanced = cv2.addWeighted(
            gray, 1.0,
            texture, self.params['texture_enhancement'], 0
        )
        
        return cv2.cvtColor(texture_enhanced, cv2.COLOR_GRAY2BGR)
    
    def enhance_vertical_structures(self, frame: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        vertical = cv2.morphologyEx(gray, cv2.MORPH_OPEN, self.vertical_kernel)
        
        vertical_3ch = cv2.cvtColor(vertical, cv2.COLOR_GRAY2BGR)
        enhanced = cv2.addWeighted(frame, 0.7, vertical_3ch, 0.3, 0)
        
        return enhanced
    
    def remove_shadows(self, frame: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        v_channel = hsv[:, :, 2]
        
        shadow_mask = v_channel < self.params['shadow_threshold']
        
        v_channel[shadow_mask] = np.clip(
            v_channel[shadow_mask] * 1.5, 0, 255
        ).astype(np.uint8)
        
        hsv[:, :, 2] = v_channel
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    def enhance_edges(self, frame: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        edges = np.abs(sobel_x).astype(np.uint8)
        edges = cv2.normalize(edges, None, 0, 255, cv2.NORM_MINMAX)
        
        edges_3ch = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        
        return cv2.addWeighted(frame, 1.0, edges_3ch, self.params['edge_strength'], 0)
    
    def suppress_background(self, frame: np.ndarray, original: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        lower_green = np.array([35, 50, 50])
        upper_green = np.array([85, 255, 255])
        
        lower_brown = np.array([5, 50, 50])
        upper_brown = np.array([30, 255, 255])
        
        green_mask = cv2.inRange(hsv, lower_green, upper_green)
        brown_soil_mask = cv2.inRange(hsv, lower_brown, upper_brown)
        
        background_mask = cv2.bitwise_or(green_mask, brown_soil_mask)
        foreground_mask = cv2.bitwise_not(background_mask)
        
        kernel = np.ones((3, 3), np.uint8)
        foreground_mask = cv2.morphologyEx(foreground_mask, cv2.MORPH_OPEN, kernel)
        foreground_mask = cv2.morphologyEx(foreground_mask, cv2.MORPH_CLOSE, kernel)
        
        foreground = cv2.bitwise_and(frame, frame, mask=foreground_mask)
        
        background = cv2.bitwise_and(original, original, mask=background_mask)
        background_suppressed = cv2.addWeighted(
            background, 0.3,
            np.zeros_like(background), 0.7, 0
        )
        
        result = cv2.add(foreground, background_suppressed)
        
        return result
    
    def get_visualization(self, frame: np.ndarray) -> Dict[str, np.ndarray]:
        results = {}
        
        results['original'] = frame.copy()
        
        enhanced = self.enhance_for_trunk_detection(frame)
        results['enhanced'] = enhanced
        
        diff = cv2.absdiff(frame, enhanced)
        results['difference'] = diff
        
        gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
        _, trunk_mask = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)
        results['trunk_mask'] = trunk_mask
        
        return results

def enhance_for_trunks(frame: np.ndarray) -> np.ndarray:
    enhancer = VineTrunkEnhancer()
    return enhancer.enhance_for_trunk_detection(frame)

def batch_enhance_for_trunks(frames: List[np.ndarray]) -> List[np.ndarray]:
    enhancer = VineTrunkEnhancer()
    return [enhancer.enhance_for_trunk_detection(f) for f in frames]