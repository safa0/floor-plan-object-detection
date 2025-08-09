"""
Legend-based Classification Module

This module implements OCR-based legend reading and pattern matching
for classifying detected dashed lines according to their legend symbols.
"""

import cv2
import numpy as np
import re
from typing import List, Dict, Tuple, Optional, Any
import pytesseract
from scipy.spatial.distance import cdist
from scipy import signal
import json
import warnings
warnings.filterwarnings('ignore')


class LinePatternClassifier:
    """Classifies line patterns based on their visual characteristics."""
    
    def __init__(self):
        self.pattern_library = self._initialize_pattern_library()
        
    def _initialize_pattern_library(self) -> Dict[str, Dict]:
        """Initialize library of known line patterns."""
        return {
            'solid': {
                'signature': [1.0],  # Continuous line
                'description': 'Solid line',
                'keywords': ['solid', 'continuous', 'wall', 'partition']
            },
            'dashed': {
                'signature': [1.0, 0.0, 1.0, 0.0],  # Dash-gap pattern
                'description': 'Dashed line',
                'keywords': ['dashed', 'dash', 'hidden', 'service']
            },
            'dot_dash': {
                'signature': [0.2, 0.0, 1.0, 0.0],  # Dot-dash pattern
                'description': 'Dot-dash line',
                'keywords': ['dot-dash', 'center', 'axis']
            },
            'dot_dot_dash': {
                'signature': [0.2, 0.0, 0.2, 0.0, 1.0, 0.0],  # Dot-dot-dash
                'description': 'Dot-dot-dash line',
                'keywords': ['dot-dot-dash', 'phantom', 'alternate']
            },
            'double_dash': {
                'signature': [1.0, 0.0, 1.0, 0.0, 0.0],  # Double dash with longer gap
                'description': 'Double dash line',
                'keywords': ['double-dash', 'boundary', 'property']
            }
        }
    
    def extract_pattern_signature(self, 
                                 intensities: np.ndarray,
                                 normalize_length: int = 20) -> np.ndarray:
        """
        Extract normalized pattern signature from intensity array.
        
        Args:
            intensities: 1D array of pixel intensities along line
            normalize_length: Target length for normalized signature
            
        Returns:
            Normalized pattern signature
        """
        if len(intensities) == 0:
            return np.array([])
        
        # Threshold to binary
        threshold = np.mean(intensities)
        binary_pattern = (intensities < threshold).astype(float)  # 1 for line, 0 for gap
        
        # Find runs of consecutive values
        runs = []
        current_value = binary_pattern[0]
        current_length = 1
        
        for i in range(1, len(binary_pattern)):
            if binary_pattern[i] == current_value:
                current_length += 1
            else:
                runs.append((current_value, current_length))
                current_value = binary_pattern[i]
                current_length = 1
        runs.append((current_value, current_length))
        
        # Convert runs to signature
        if not runs:
            return np.array([])
        
        # Normalize run lengths
        total_length = sum(length for _, length in runs)
        signature = []
        
        for value, length in runs:
            normalized_length = length / total_length
            signature.append(normalized_length if value == 1 else 0)
        
        # Resample to fixed length if needed
        if len(signature) != normalize_length:
            # Simple resampling
            x_old = np.linspace(0, 1, len(signature))
            x_new = np.linspace(0, 1, normalize_length)
            signature = np.interp(x_new, x_old, signature)
        
        return np.array(signature)
    
    def classify_pattern(self, signature: np.ndarray) -> Tuple[str, float]:
        """
        Classify pattern signature against known patterns.
        
        Args:
            signature: Pattern signature array
            
        Returns:
            Tuple of (pattern_name, confidence)
        """
        if len(signature) == 0:
            return 'unknown', 0.0
        
        best_match = 'unknown'
        best_confidence = 0.0
        
        for pattern_name, pattern_info in self.pattern_library.items():
            pattern_sig = np.array(pattern_info['signature'])
            
            # Calculate similarity using Dynamic Time Warping approximation
            confidence = self._calculate_pattern_similarity(signature, pattern_sig)
            
            if confidence > best_confidence:
                best_confidence = confidence
                best_match = pattern_name
        
        return best_match, best_confidence
    
    def _calculate_pattern_similarity(self, sig1: np.ndarray, sig2: np.ndarray) -> float:
        """Calculate similarity between two pattern signatures."""
        # Normalize signatures
        sig1_norm = sig1 / (np.sum(sig1) + 1e-8)
        sig2_norm = sig2 / (np.sum(sig2) + 1e-8)
        
        # Resize to same length
        target_len = max(len(sig1_norm), len(sig2_norm))
        if len(sig1_norm) != target_len:
            x = np.linspace(0, 1, len(sig1_norm))
            x_new = np.linspace(0, 1, target_len)
            sig1_norm = np.interp(x_new, x, sig1_norm)
        
        if len(sig2_norm) != target_len:
            x = np.linspace(0, 1, len(sig2_norm))
            x_new = np.linspace(0, 1, target_len)
            sig2_norm = np.interp(x_new, x, sig2_norm)
        
        # Calculate cross-correlation
        correlation = signal.correlate(sig1_norm, sig2_norm, mode='full')
        max_correlation = np.max(correlation)
        
        # Normalize by signal energy
        energy1 = np.sum(sig1_norm ** 2)
        energy2 = np.sum(sig2_norm ** 2)
        normalization = np.sqrt(energy1 * energy2)
        
        if normalization > 0:
            similarity = max_correlation / normalization
        else:
            similarity = 0.0
        
        return min(max(similarity, 0.0), 1.0)


class LegendReader:
    """Reads and parses legend information from floor plans."""
    
    def __init__(self):
        self.legend_entries = []
        self.pattern_classifier = LinePatternClassifier()
        
    def detect_legend_region(self, image: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """
        Detect legend region in the image.
        
        Args:
            image: Input floor plan image
            
        Returns:
            Legend region as (x, y, width, height) or None if not found
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        h, w = gray.shape
        
        # Common legend locations
        search_regions = [
            (int(w*0.7), int(h*0.7), int(w*0.3), int(h*0.3)),  # Bottom-right
            (0, int(h*0.7), int(w*0.3), int(h*0.3)),           # Bottom-left
            (int(w*0.7), 0, int(w*0.3), int(h*0.3)),           # Top-right
            (0, 0, int(w*0.3), int(h*0.3))                     # Top-left
        ]
        
        best_region = None
        best_score = 0
        
        for x, y, rw, rh in search_regions:
            region = gray[y:y+rh, x:x+rw]
            score = self._evaluate_legend_region(region)
            
            if score > best_score:
                best_score = score
                best_region = (x, y, rw, rh)
        
        return best_region if best_score > 0.3 else None
    
    def _evaluate_legend_region(self, region: np.ndarray) -> float:
        """Evaluate how likely a region is to contain a legend."""
        if region.size == 0:
            return 0.0
        
        # Look for characteristics of legends:
        # 1. Text content
        # 2. Line symbols
        # 3. Structured layout
        
        score = 0.0
        
        # Text detection using edge density
        edges = cv2.Canny(region, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        if 0.05 < edge_density < 0.3:  # Moderate edge density suggests text/symbols
            score += 0.4
        
        # Horizontal line detection (common in legends)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=20, 
                               minLineLength=20, maxLineGap=5)
        if lines is not None:
            horizontal_lines = 0
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = abs(np.arctan2(y2-y1, x2-x1) * 180 / np.pi)
                if angle < 15 or angle > 165:  # Nearly horizontal
                    horizontal_lines += 1
            
            if horizontal_lines >= 2:  # Multiple horizontal lines suggest legend
                score += 0.3
        
        # OCR confidence as indicator of text presence
        try:
            custom_config = r'--oem 3 --psm 6'
            text = pytesseract.image_to_string(region, config=custom_config)
            if len(text.strip()) > 10:  # Reasonable amount of text
                score += 0.3
        except:
            pass
        
        return min(score, 1.0)
    
    def parse_legend(self, image: np.ndarray, legend_region: Tuple[int, int, int, int]) -> List[Dict]:
        """
        Parse legend entries from the legend region.
        
        Args:
            image: Input image
            legend_region: Legend region coordinates (x, y, width, height)
            
        Returns:
            List of legend entry dictionaries
        """
        x, y, w, h = legend_region
        legend_img = image[y:y+h, x:x+w]
        
        if len(legend_img.shape) == 3:
            gray_legend = cv2.cvtColor(legend_img, cv2.COLOR_BGR2GRAY)
        else:
            gray_legend = legend_img.copy()
        
        entries = []
        
        # Detect text regions and associated line symbols
        entries.extend(self._extract_text_based_entries(gray_legend))
        entries.extend(self._extract_symbol_based_entries(gray_legend))
        
        # Adjust coordinates to global image coordinates
        for entry in entries:
            if 'bbox' in entry:
                entry['bbox'] = (
                    entry['bbox'][0] + x,
                    entry['bbox'][1] + y,
                    entry['bbox'][2],
                    entry['bbox'][3]
                )
        
        self.legend_entries = entries
        return entries
    
    def _extract_text_based_entries(self, legend_img: np.ndarray) -> List[Dict]:
        """Extract legend entries based on OCR text detection."""
        entries = []
        
        try:
            # OCR with bounding boxes
            custom_config = r'--oem 3 --psm 6'
            data = pytesseract.image_to_data(legend_img, config=custom_config, output_type=pytesseract.Output.DICT)
            
            # Group text by lines
            text_lines = {}
            for i, text in enumerate(data['text']):
                if text.strip():
                    top = data['top'][i]
                    # Group by approximate line position
                    line_key = top // 20  # Group within 20 pixels vertically
                    if line_key not in text_lines:
                        text_lines[line_key] = []
                    text_lines[line_key].append({
                        'text': text,
                        'left': data['left'][i],
                        'top': data['top'][i],
                        'width': data['width'][i],
                        'height': data['height'][i],
                        'conf': data['conf'][i]
                    })
            
            # Process each text line
            for line_key, words in text_lines.items():
                if not words:
                    continue
                
                # Combine words into full text
                words.sort(key=lambda x: x['left'])  # Sort by horizontal position
                full_text = ' '.join([w['text'] for w in words])
                
                # Skip if text is too short or low confidence
                avg_conf = np.mean([w['conf'] for w in words])
                if len(full_text.strip()) < 3 or avg_conf < 30:
                    continue
                
                # Calculate bounding box for the line
                min_left = min(w['left'] for w in words)
                max_right = max(w['left'] + w['width'] for w in words)
                min_top = min(w['top'] for w in words)
                max_bottom = max(w['top'] + w['height'] for w in words)
                
                # Look for line symbols near this text
                symbol_region = self._find_line_symbol_near_text(
                    legend_img, min_left, min_top, max_right - min_left, max_bottom - min_top
                )
                
                # Parse text for line type information
                line_info = self._parse_line_description(full_text)
                
                entry = {
                    'text': full_text,
                    'bbox': (min_left, min_top, max_right - min_left, max_bottom - min_top),
                    'confidence': avg_conf / 100.0,
                    'line_type': line_info.get('type', 'unknown'),
                    'keywords': line_info.get('keywords', []),
                    'symbol_region': symbol_region,
                    'extraction_method': 'text_based'
                }
                
                entries.append(entry)
        
        except Exception as e:
            print(f"OCR error in legend parsing: {e}")
        
        return entries
    
    def _extract_symbol_based_entries(self, legend_img: np.ndarray) -> List[Dict]:
        """Extract legend entries based on line symbol detection."""
        entries = []
        
        # Detect horizontal line segments in legend
        edges = cv2.Canny(legend_img, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=15, 
                               minLineLength=20, maxLineGap=3)
        
        if lines is None:
            return entries
        
        # Filter for horizontal lines
        horizontal_lines = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = abs(np.arctan2(y2-y1, x2-x1) * 180 / np.pi)
            if angle < 15 or angle > 165:  # Nearly horizontal
                length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                if length > 15:  # Minimum length for symbol
                    horizontal_lines.append({
                        'coords': (x1, y1, x2, y2),
                        'length': length,
                        'y_center': (y1 + y2) / 2
                    })
        
        # Group lines by vertical position (potential legend entries)
        if not horizontal_lines:
            return entries
        
        # Sort by vertical position
        horizontal_lines.sort(key=lambda x: x['y_center'])
        
        # Group lines that are close vertically
        line_groups = []
        current_group = [horizontal_lines[0]]
        
        for i in range(1, len(horizontal_lines)):
            if horizontal_lines[i]['y_center'] - current_group[-1]['y_center'] < 30:
                current_group.append(horizontal_lines[i])
            else:
                line_groups.append(current_group)
                current_group = [horizontal_lines[i]]
        line_groups.append(current_group)
        
        # Analyze each group for pattern
        for group in line_groups:
            if not group:
                continue
            
            # Get the region around this group of lines
            min_y = min(line['y_center'] for line in group) - 10
            max_y = max(line['y_center'] for line in group) + 10
            min_x = min(min(line['coords'][0], line['coords'][2]) for line in group) - 10
            max_x = max(max(line['coords'][0], line['coords'][2]) for line in group) + 10
            
            # Ensure bounds
            min_y = max(0, int(min_y))
            max_y = min(legend_img.shape[0], int(max_y))
            min_x = max(0, int(min_x))
            max_x = min(legend_img.shape[1], int(max_x))
            
            symbol_region = legend_img[min_y:max_y, min_x:max_x]
            
            if symbol_region.size == 0:
                continue
            
            # Extract pattern signature from the symbol
            # Sample along the longest line in the group
            longest_line = max(group, key=lambda x: x['length'])
            x1, y1, x2, y2 = longest_line['coords']
            
            # Sample intensities along line
            intensities = self._sample_line_intensities(legend_img, x1, y1, x2, y2)
            pattern_signature = self.pattern_classifier.extract_pattern_signature(intensities)
            
            # Classify pattern
            pattern_type, confidence = self.pattern_classifier.classify_pattern(pattern_signature)
            
            if confidence > 0.3:  # Minimum confidence threshold
                entry = {
                    'text': f'Line pattern: {pattern_type}',
                    'bbox': (min_x, min_y, max_x - min_x, max_y - min_y),
                    'confidence': confidence,
                    'line_type': pattern_type,
                    'pattern_signature': pattern_signature.tolist(),
                    'symbol_lines': [line['coords'] for line in group],
                    'extraction_method': 'symbol_based'
                }
                
                entries.append(entry)
        
        return entries
    
    def _find_line_symbol_near_text(self, 
                                   legend_img: np.ndarray,
                                   text_x: int, text_y: int,
                                   text_w: int, text_h: int) -> Optional[Dict]:
        """Find line symbol near detected text."""
        # Search regions around text
        search_margin = 20
        
        # Check left side of text (common legend layout)
        search_x = max(0, text_x - search_margin - 50)
        search_w = text_x - search_x + search_margin
        search_y = max(0, text_y - search_margin)
        search_h = text_h + 2 * search_margin
        
        if search_w <= 0 or search_h <= 0:
            return None
        
        search_region = legend_img[search_y:search_y+search_h, search_x:search_x+search_w]
        
        # Look for horizontal lines in search region
        edges = cv2.Canny(search_region, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=10, 
                               minLineLength=15, maxLineGap=3)
        
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = abs(np.arctan2(y2-y1, x2-x1) * 180 / np.pi)
                if angle < 15 or angle > 165:  # Nearly horizontal
                    # Found a symbol line
                    return {
                        'region_bbox': (search_x, search_y, search_w, search_h),
                        'line_coords': (x1 + search_x, y1 + search_y, 
                                      x2 + search_x, y2 + search_y),
                        'length': np.sqrt((x2-x1)**2 + (y2-y1)**2)
                    }
        
        return None
    
    def _sample_line_intensities(self, 
                                image: np.ndarray,
                                x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
        """Sample pixel intensities along a line."""
        length = int(np.sqrt((x2 - x1)**2 + (y2 - y1)**2))
        if length == 0:
            return np.array([])
        
        num_samples = max(length, 20)
        t = np.linspace(0, 1, num_samples)
        
        x_samples = (x1 + t * (x2 - x1)).astype(int)
        y_samples = (y1 + t * (y2 - y1)).astype(int)
        
        # Ensure coordinates are within bounds
        x_samples = np.clip(x_samples, 0, image.shape[1] - 1)
        y_samples = np.clip(y_samples, 0, image.shape[0] - 1)
        
        intensities = image[y_samples, x_samples]
        return intensities.astype(np.float32)
    
    def _parse_line_description(self, text: str) -> Dict[str, Any]:
        """Parse line type information from text description."""
        text_upper = text.upper()
        
        # Common patterns and keywords
        patterns = {
            'dashed': ['DASH', 'HIDDEN', 'SERVICE', 'UTILITY'],
            'dot_dash': ['DOT-DASH', 'CENTER', 'AXIS', 'CENTERLINE'],
            'dot_dot_dash': ['DOT-DOT-DASH', 'PHANTOM', 'ALTERNATE'],
            'boundary': ['BOUNDARY', 'PROPERTY', 'LIMIT'],
            'construction': ['CONSTRUCTION', 'DEMOLITION', 'NEW', 'EXISTING']
        }
        
        detected_type = 'unknown'
        keywords = []
        
        for pattern_type, pattern_keywords in patterns.items():
            for keyword in pattern_keywords:
                if keyword in text_upper:
                    detected_type = pattern_type
                    keywords.append(keyword.lower())
        
        # Extract any numeric codes (like "SP8-DRÄN" or "TV2 KTV1-15")
        codes = re.findall(r'[A-Z]+\d+[-\w]*', text)
        
        return {
            'type': detected_type,
            'keywords': keywords,
            'codes': codes,
            'original_text': text
        }
    
    def classify_detected_line(self, 
                              line_signature: np.ndarray,
                              line_geometry: Dict) -> Tuple[str, float, Dict]:
        """
        Classify a detected line against legend entries.
        
        Args:
            line_signature: Pattern signature of the detected line
            line_geometry: Geometric properties of the line
            
        Returns:
            Tuple of (classification, confidence, matching_legend_entry)
        """
        if not self.legend_entries:
            # Fallback to pattern classification without legend
            pattern_type, confidence = self.pattern_classifier.classify_pattern(line_signature)
            return pattern_type, confidence, {}
        
        best_match = None
        best_confidence = 0.0
        best_entry = {}
        
        for entry in self.legend_entries:
            confidence = 0.0
            
            # Method 1: Pattern signature matching
            if 'pattern_signature' in entry:
                pattern_conf = self.pattern_classifier._calculate_pattern_similarity(
                    line_signature, np.array(entry['pattern_signature'])
                )
                confidence += pattern_conf * 0.6
            
            # Method 2: Text-based classification
            if 'line_type' in entry and entry['line_type'] != 'unknown':
                pattern_type, pattern_conf = self.pattern_classifier.classify_pattern(line_signature)
                if pattern_type == entry['line_type']:
                    confidence += pattern_conf * 0.4
            
            # Method 3: Keyword matching (if available from geometry analysis)
            if 'keywords' in entry and 'description' in line_geometry:
                keyword_matches = sum(1 for kw in entry['keywords'] 
                                    if kw.lower() in line_geometry['description'].lower())
                if keyword_matches > 0:
                    confidence += (keyword_matches / len(entry['keywords'])) * 0.3
            
            if confidence > best_confidence:
                best_confidence = confidence
                best_match = entry.get('line_type', 'unknown')
                best_entry = entry
        
        return best_match or 'unknown', best_confidence, best_entry


def main():
    """Example usage of legend classification system."""
    print("Legend-based Line Classification System")
    print("======================================")
    
    # Create a test legend reader
    reader = LegendReader()
    
    print("Pattern library:")
    for pattern_name, info in reader.pattern_classifier.pattern_library.items():
        print(f"  {pattern_name}: {info['description']}")
    
    # Test pattern classification
    test_signature = np.array([1.0, 0.0, 1.0, 0.0, 1.0, 0.0])  # Dashed pattern
    pattern_type, confidence = reader.pattern_classifier.classify_pattern(test_signature)
    print(f"\nTest pattern classification: {pattern_type} (confidence: {confidence:.3f})")


if __name__ == "__main__":
    main()
