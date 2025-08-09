"""
Scale Detection Module for Floor Plans

This module implements various methods to detect and extract scale information
from floor plan drawings, including OCR-based scale text detection and 
scale bar detection.
"""

import cv2
import numpy as np
import re
from typing import Tuple, Optional, Dict, List
import pytesseract
from scipy import ndimage
import matplotlib.pyplot as plt


class ScaleDetector:
    """Detects scale information from floor plan images."""
    
    def __init__(self):
        self.detected_scale = None
        self.scale_type = None  # 'text', 'bar', 'manual'
        self.scale_px_to_m = 1.0
        
    def detect_scale(self, 
                    image: np.ndarray,
                    search_regions: List[str] = None) -> Tuple[float, str]:
        """
        Detect scale from image using multiple methods.
        
        Args:
            image: Input floor plan image
            search_regions: List of regions to search ('title_block', 'corners', 'all')
            
        Returns:
            Tuple of (scale_factor_px_to_m, detection_method)
        """
        if search_regions is None:
            search_regions = ['title_block', 'corners']
        
        # Try OCR-based scale text detection first
        for region in search_regions:
            scale_factor, confidence = self._detect_scale_text(image, region)
            if scale_factor is not None and confidence > 0.7:
                self.scale_px_to_m = scale_factor
                self.scale_type = 'text'
                return scale_factor, 'text_ocr'
        
        # Try scale bar detection
        scale_factor = self._detect_scale_bar(image)
        if scale_factor is not None:
            self.scale_px_to_m = scale_factor
            self.scale_type = 'bar'
            return scale_factor, 'scale_bar'
        
        # Fallback: estimate from drawing size (rough heuristic)
        scale_factor = self._estimate_scale_from_size(image)
        self.scale_px_to_m = scale_factor
        self.scale_type = 'estimated'
        return scale_factor, 'estimated'
    
    def _detect_scale_text(self, 
                          image: np.ndarray, 
                          region: str = 'title_block') -> Tuple[Optional[float], float]:
        """
        Detect scale from text using OCR.
        
        Args:
            image: Input image
            region: Search region ('title_block', 'corners', 'all')
            
        Returns:
            Tuple of (scale_factor, confidence)
        """
        # Define search region
        if region == 'title_block':
            # Typically bottom-right corner
            h, w = image.shape[:2]
            roi = image[int(h*0.7):h, int(w*0.7):w]
        elif region == 'corners':
            # Search all four corners
            h, w = image.shape[:2]
            corner_size = min(h, w) // 4
            
            corners = [
                image[0:corner_size, 0:corner_size],  # Top-left
                image[0:corner_size, w-corner_size:w],  # Top-right
                image[h-corner_size:h, 0:corner_size],  # Bottom-left
                image[h-corner_size:h, w-corner_size:w]  # Bottom-right
            ]
            
            best_scale = None
            best_confidence = 0
            
            for corner in corners:
                scale, conf = self._extract_scale_from_roi(corner)
                if conf > best_confidence:
                    best_scale = scale
                    best_confidence = conf
            
            return best_scale, best_confidence
        else:  # 'all'
            roi = image
        
        return self._extract_scale_from_roi(roi)
    
    def _extract_scale_from_roi(self, roi: np.ndarray) -> Tuple[Optional[float], float]:
        """Extract scale information from a specific ROI using OCR."""
        if roi.size == 0:
            return None, 0.0
        
        # Preprocess ROI for better OCR
        if len(roi.shape) == 3:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        else:
            gray = roi.copy()
        
        # Enhance contrast
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Threshold
        _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Scale up for better OCR
        scale_factor = 3
        binary_scaled = cv2.resize(binary, None, fx=scale_factor, fy=scale_factor, 
                                 interpolation=cv2.INTER_CUBIC)
        
        try:
            # OCR with specific configuration for scale text
            custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789:='
            text = pytesseract.image_to_string(binary_scaled, config=custom_config)
            
            # Parse scale patterns
            scale_info = self._parse_scale_text(text)
            return scale_info
            
        except Exception as e:
            print(f"OCR error: {e}")
            return None, 0.0
    
    def _parse_scale_text(self, text: str) -> Tuple[Optional[float], float]:
        """
        Parse scale information from OCR text.
        
        Common patterns:
        - "1:100" or "1:50" (ratio scales)
        - "1/4\" = 1'-0\"" (architectural scales)
        - "Scale: 1:100"
        - "1cm = 1m"
        """
        confidence = 0.0
        
        # Clean text
        text = text.strip().upper().replace(' ', '')
        
        # Pattern 1: Ratio scales (1:100, 1:50, etc.)
        ratio_pattern = r'1:(\d+)'
        match = re.search(ratio_pattern, text)
        if match:
            ratio = int(match.group(1))
            # 1:100 means 1 unit on drawing = 100 units in reality
            # If drawing is in mm and reality is in mm, scale factor is 1/100
            # But we want px to meters, so we need to consider drawing DPI
            scale_factor = 1.0 / ratio  # This is drawing_units to real_units
            
            # Assume drawing is at ~300 DPI and drawn in mm
            # 1 meter = 1000 mm
            # 1 mm at 300 DPI = 300/25.4 ≈ 11.8 pixels
            mm_to_px = 300 / 25.4  # pixels per mm
            scale_factor_px_to_m = scale_factor / mm_to_px * 1000  # px to meters
            
            confidence = 0.9
            return scale_factor_px_to_m, confidence
        
        # Pattern 2: Architectural scales (1/4" = 1'-0")
        arch_pattern = r'1/(\d+)"?=(\d+)\'?-?(\d+)"?'
        match = re.search(arch_pattern, text)
        if match:
            drawing_fraction = int(match.group(1))  # 1/4 inch
            real_feet = int(match.group(2))  # 1 foot
            real_inches = int(match.group(3)) if match.group(3) else 0
            
            # Convert to consistent units (inches)
            drawing_inches = 1.0 / drawing_fraction
            real_total_inches = real_feet * 12 + real_inches
            
            # Scale factor: drawing inches to real inches
            scale_factor = real_total_inches / drawing_inches
            
            # Convert to px to meters
            # Assume 300 DPI: 1 inch = 300 px
            # 1 meter = 39.37 inches
            px_per_inch = 300
            inches_per_meter = 39.37
            scale_factor_px_to_m = scale_factor / px_per_inch / inches_per_meter
            
            confidence = 0.8
            return scale_factor_px_to_m, confidence
        
        # Pattern 3: Metric scales (1cm = 1m)
        metric_pattern = r'1CM=(\d+)M'
        match = re.search(metric_pattern, text)
        if match:
            real_meters = int(match.group(1))
            
            # 1 cm on drawing = real_meters in reality
            # Assume 300 DPI: 1 cm = 300/2.54 ≈ 118 px
            cm_to_px = 300 / 2.54
            scale_factor_px_to_m = real_meters / cm_to_px
            
            confidence = 0.8
            return scale_factor_px_to_m, confidence
        
        return None, confidence
    
    def _detect_scale_bar(self, image: np.ndarray) -> Optional[float]:
        """
        Detect and measure scale bar in the image.
        
        This looks for horizontal bars (typically black rectangles) with 
        labels indicating their length.
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Search in bottom portion of image (common location for scale bars)
        h, w = gray.shape
        search_region = gray[int(h*0.7):h, :]
        
        # Detect horizontal line segments (potential scale bars)
        edges = cv2.Canny(search_region, 50, 150)
        
        # Hough line detection with focus on horizontal lines
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, 
                               minLineLength=50, maxLineGap=10)
        
        if lines is None:
            return None
        
        # Filter for horizontal lines
        horizontal_lines = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            # Check if line is roughly horizontal
            length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
            if length > 50:  # Minimum length for scale bar
                angle = np.abs(np.arctan2(y2-y1, x2-x1) * 180 / np.pi)
                if angle < 10 or angle > 170:  # Nearly horizontal
                    horizontal_lines.append((line[0], length))
        
        if not horizontal_lines:
            return None
        
        # Find the longest horizontal line (likely scale bar)
        horizontal_lines.sort(key=lambda x: x[1], reverse=True)
        best_line, line_length = horizontal_lines[0]
        
        # Try to find associated text near the scale bar
        x1, y1, x2, y2 = best_line
        
        # Adjust coordinates to original image
        y1 += int(h*0.7)
        y2 += int(h*0.7)
        
        # Extract region around scale bar for OCR
        margin = 30
        roi_y1 = max(0, min(y1, y2) - margin)
        roi_y2 = min(h, max(y1, y2) + margin)
        roi_x1 = max(0, min(x1, x2) - margin)
        roi_x2 = min(w, max(x1, x2) + margin)
        
        scale_bar_roi = gray[roi_y1:roi_y2, roi_x1:roi_x2]
        
        # OCR on scale bar region
        try:
            custom_config = r'--oem 3 --psm 8 -c tessedit_char_whitelist=0123456789.mMcCfFtT'
            text = pytesseract.image_to_string(scale_bar_roi, config=custom_config)
            
            # Parse scale bar text for length
            numbers = re.findall(r'\d+\.?\d*', text)
            if numbers:
                # Assume first number is the length
                labeled_length = float(numbers[0])
                
                # Determine units from text
                text_upper = text.upper()
                if 'M' in text_upper and 'MM' not in text_upper:
                    # Meters
                    real_length_m = labeled_length
                elif 'CM' in text_upper:
                    # Centimeters
                    real_length_m = labeled_length / 100
                elif 'MM' in text_upper:
                    # Millimeters
                    real_length_m = labeled_length / 1000
                elif 'FT' in text_upper or "'" in text:
                    # Feet
                    real_length_m = labeled_length * 0.3048
                else:
                    # Default to meters
                    real_length_m = labeled_length
                
                # Calculate scale factor
                scale_factor_px_to_m = real_length_m / line_length
                return scale_factor_px_to_m
                
        except Exception as e:
            print(f"Scale bar OCR error: {e}")
        
        return None
    
    def _estimate_scale_from_size(self, image: np.ndarray) -> float:
        """
        Estimate scale based on image size and typical floor plan dimensions.
        
        This is a rough fallback method.
        """
        h, w = image.shape[:2]
        
        # Typical assumptions:
        # - Floor plans are usually between 10m x 10m (small room) to 50m x 50m (large building)
        # - Most common residential plans are around 15m x 20m
        
        # Estimate based on image size
        min_dimension_px = min(h, w)
        max_dimension_px = max(h, w)
        
        # Heuristic: assume the larger dimension represents ~20-30 meters
        if max_dimension_px > 2000:
            # High resolution image, likely detailed plan
            estimated_real_length_m = 25.0
        elif max_dimension_px > 1000:
            # Medium resolution
            estimated_real_length_m = 20.0
        else:
            # Lower resolution or small plan
            estimated_real_length_m = 15.0
        
        scale_factor_px_to_m = estimated_real_length_m / max_dimension_px
        
        print(f"Warning: Using estimated scale {scale_factor_px_to_m:.6f} px/m. "
              f"Consider providing manual scale for accurate measurements.")
        
        return scale_factor_px_to_m
    
    def set_manual_scale(self, 
                        point1: Tuple[int, int], 
                        point2: Tuple[int, int], 
                        known_distance_m: float) -> float:
        """
        Set scale manually using two points with known distance.
        
        Args:
            point1: First point (x, y)
            point2: Second point (x, y)
            known_distance_m: Known distance between points in meters
            
        Returns:
            Calculated scale factor (px to meters)
        """
        pixel_distance = np.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)
        
        if pixel_distance == 0:
            raise ValueError("Points cannot be identical")
        
        scale_factor = known_distance_m / pixel_distance
        self.scale_px_to_m = scale_factor
        self.scale_type = 'manual'
        
        return scale_factor
    
    def get_scale_info(self) -> Dict[str, any]:
        """Get current scale information."""
        return {
            'scale_px_to_m': self.scale_px_to_m,
            'scale_type': self.scale_type,
            'detected_scale': self.detected_scale
        }


# Example usage and testing
def test_scale_detection():
    """Test the scale detection system."""
    detector = ScaleDetector()
    
    # Create a simple test image with scale text
    test_image = np.ones((800, 1200), dtype=np.uint8) * 255
    
    # Add some "scale text" (in a real scenario this would be from OCR)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(test_image, "Scale 1:100", (900, 750), font, 1, 0, 2)
    
    # Test scale detection
    scale_factor, method = detector.detect_scale(test_image)
    
    print(f"Detected scale: {scale_factor:.6f} px/m")
    print(f"Detection method: {method}")
    
    # Test manual scale setting
    manual_scale = detector.set_manual_scale((100, 100), (200, 100), 5.0)  # 100px = 5m
    print(f"Manual scale: {manual_scale:.6f} px/m")


if __name__ == "__main__":
    test_scale_detection()
