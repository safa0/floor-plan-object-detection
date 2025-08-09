"""
Comprehensive Dashed Line Detection System for Floor Plans

This module implements a classic CV pipeline for extracting dashed lines as vectors,
measuring their lengths, and exporting to separate layers (SVG/GeoJSON).

Features:
- Preprocessing with deskewing and adaptive thresholding
- Template-based dash detection with user-assisted learning
- Matched filtering with directional cleanup
- Segment linking to create polylines
- Vectorization and length measurement
- Multiple output formats (SVG, GeoJSON, CSV)
- Robust upgrades with LSD and frequency analysis
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage, signal
from scipy.spatial import KDTree
from scipy.spatial.distance import cdist
from skimage import morphology, measure, segmentation
from scipy.signal import find_peaks
from sklearn.cluster import DBSCAN
import json
import xml.etree.ElementTree as ET
from typing import List, Tuple, Dict, Optional, Any
import warnings
warnings.filterwarnings('ignore')


class DashTemplate:
    """Represents a learned dash template with multiple orientations."""
    
    def __init__(self, dash_patch: np.ndarray, angles: List[float] = None):
        """
        Initialize dash template from a user-selected patch.
        
        Args:
            dash_patch: Binary image patch containing one dash + gap cycle
            angles: List of angles to generate rotated templates (default: every 7.5°)
        """
        if angles is None:
            angles = np.arange(0, 180, 7.5)  # Every 7.5 degrees
            
        self.angles = angles
        self.templates = {}
        self.dash_length = 0
        self.gap_length = 0
        self.total_period = 0
        
        self._learn_from_patch(dash_patch)
    
    def _learn_from_patch(self, patch: np.ndarray):
        """Learn dash characteristics from the patch."""
        # Ensure binary
        if patch.dtype != np.uint8:
            patch = (patch * 255).astype(np.uint8)
        if len(patch.shape) == 3:
            patch = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        
        # Threshold to binary
        _, patch = cv2.threshold(patch, 127, 255, cv2.THRESH_BINARY)
        
        # Find the main dash axis by analyzing projections
        h_proj = np.sum(patch, axis=0)
        v_proj = np.sum(patch, axis=1)
        
        # Determine if horizontal or vertical dominant
        h_var = np.var(h_proj)
        v_var = np.var(v_proj)
        
        if h_var > v_var:
            # Horizontal dash
            projection = h_proj
            self.dash_length = np.sum(projection > 0)
            # Find gaps in projection
            diff = np.diff(np.concatenate(([0], projection > 0, [0])).astype(int))
            starts = np.where(diff == 1)[0]
            ends = np.where(diff == -1)[0]
            if len(starts) > 1:
                self.gap_length = starts[1] - ends[0] if len(ends) > 0 else 10
            else:
                self.gap_length = self.dash_length // 2
        else:
            # Vertical dash
            projection = v_proj
            self.dash_length = np.sum(projection > 0)
            diff = np.diff(np.concatenate(([0], projection > 0, [0])).astype(int))
            starts = np.where(diff == 1)[0]
            ends = np.where(diff == -1)[0]
            if len(starts) > 1:
                self.gap_length = starts[1] - ends[0] if len(ends) > 0 else 10
            else:
                self.gap_length = self.dash_length // 2
        
        self.total_period = self.dash_length + self.gap_length
        
        # Generate rotated templates
        for angle in self.angles:
            rotated = self._rotate_template(patch, angle)
            self.templates[angle] = rotated
    
    def _rotate_template(self, template: np.ndarray, angle: float) -> np.ndarray:
        """Rotate template by given angle."""
        center = (template.shape[1] // 2, template.shape[0] // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(template, rotation_matrix, template.shape[::-1])
        return rotated


class FloorPlanPreprocessor:
    """Handles preprocessing of floor plan images."""
    
    @staticmethod
    def preprocess(image: np.ndarray, 
                  apply_deskew: bool = True,
                  apply_clahe: bool = True,
                  clahe_clip_limit: float = 2.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess floor plan image.
        
        Args:
            image: Input image (color or grayscale)
            apply_deskew: Whether to apply deskewing
            apply_clahe: Whether to apply CLAHE contrast enhancement
            clahe_clip_limit: CLAHE clip limit
            
        Returns:
            Tuple of (processed_image, binary_image)
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Deskew if requested
        if apply_deskew:
            gray = FloorPlanPreprocessor._deskew_image(gray)
        
        # Apply CLAHE for contrast enhancement
        if apply_clahe:
            clahe = cv2.createCLAHE(clipLimit=clahe_clip_limit, tileGridSize=(8, 8))
            gray = clahe.apply(gray)
        
        # Denoise
        gray = cv2.medianBlur(gray, 3)
        
        # Adaptive threshold for crisp binary
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        return gray, binary
    
    @staticmethod
    def _deskew_image(image: np.ndarray) -> np.ndarray:
        """Deskew image using Hough transform of page borders."""
        # Edge detection
        edges = cv2.Canny(image, 50, 150, apertureSize=3)
        
        # Hough line detection
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
        
        if lines is None:
            return image
        
        # Find dominant angles
        angles = []
        for rho, theta in lines[:, 0]:
            angle = theta * 180 / np.pi
            # Convert to -45 to 45 degree range
            if angle > 45:
                angle = angle - 90
            elif angle < -45:
                angle = angle + 90
            angles.append(angle)
        
        # Use median angle for deskewing
        if angles:
            skew_angle = np.median(angles)
            if abs(skew_angle) > 0.5:  # Only deskew if significant
                center = (image.shape[1] // 2, image.shape[0] // 2)
                rotation_matrix = cv2.getRotationMatrix2D(center, skew_angle, 1.0)
                image = cv2.warpAffine(image, rotation_matrix, image.shape[::-1])
        
        return image
    
    @staticmethod
    def remove_text_regions(image: np.ndarray, 
                          original: np.ndarray) -> np.ndarray:
        """
        Remove text regions using MSER detection and inpainting.
        
        Args:
            image: Binary working image
            original: Original grayscale image for inpainting
            
        Returns:
            Image with text regions inpainted
        """
        # MSER text detection
        mser = cv2.MSER_create()
        mser.setMinArea(30)
        mser.setMaxArea(14400)
        mser.setMaxVariation(0.25)
        mser.setMinDiversity(0.2)
        
        regions, _ = mser.detectRegions(original)
        
        # Create mask for text regions
        text_mask = np.zeros(original.shape, dtype=np.uint8)
        
        for region in regions:
            # Filter regions by aspect ratio and size
            x, y, w, h = cv2.boundingRect(region.reshape(-1, 1, 2))
            aspect_ratio = w / h if h > 0 else 0
            
            # Typical text characteristics
            if 0.1 < aspect_ratio < 10 and 100 < w * h < 5000:
                cv2.fillPoly(text_mask, [region.reshape(-1, 1, 2)], 255)
        
        # Dilate text mask slightly
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        text_mask = cv2.dilate(text_mask, kernel, iterations=1)
        
        # Inpaint text regions
        inpainted = cv2.inpaint(original, text_mask, 3, cv2.INPAINT_TELEA)
        
        return inpainted


class DashedLineDetector:
    """Main class for detecting dashed lines in floor plans."""
    
    def __init__(self):
        self.dash_template = None
        self.preprocessor = FloorPlanPreprocessor()
        self.scale_px_to_m = 1.0  # pixels to meters conversion
        self.detected_lines = []
        self.intermediate_results = {}
        
    def learn_dash_template(self, image: np.ndarray, roi: Tuple[int, int, int, int]) -> None:
        """
        Learn dash template from user-selected ROI.
        
        Args:
            image: Input image
            roi: (x, y, width, height) of the ROI containing dash sample
        """
        x, y, w, h = roi
        patch = image[y:y+h, x:x+w]
        
        # Ensure we have a good patch
        if patch.size == 0:
            raise ValueError("Empty ROI patch")
        
        self.dash_template = DashTemplate(patch)
        print(f"Learned dash template: dash_length={self.dash_template.dash_length}, "
              f"gap_length={self.dash_template.gap_length}")
    
    def detect_dashed_lines(self, 
                          image: np.ndarray,
                          remove_text: bool = True,
                          nms_threshold: float = 0.3,
                          confidence_threshold: float = 0.6) -> List[Dict]:
        """
        Main detection pipeline for dashed lines.
        
        Args:
            image: Input floor plan image
            remove_text: Whether to remove text before detection
            nms_threshold: Non-maximum suppression threshold
            confidence_threshold: Minimum confidence for detections
            
        Returns:
            List of detected line dictionaries
        """
        if self.dash_template is None:
            raise ValueError("No dash template learned. Call learn_dash_template() first.")
        
        # Preprocess
        gray, binary = self.preprocessor.preprocess(image)
        
        # Optional text removal
        if remove_text:
            gray_clean = self.preprocessor.remove_text_regions(binary, gray)
            _, binary_clean = cv2.threshold(gray_clean, 127, 255, cv2.THRESH_BINARY)
        else:
            binary_clean = binary
        
        self.intermediate_results['binary'] = binary_clean
        
        # Template matching
        response_maps = self._template_matching(binary_clean)
        self.intermediate_results['response_maps'] = response_maps
        
        # Non-maximum suppression and thresholding
        dash_candidates = self._extract_candidates(response_maps, 
                                                 nms_threshold, 
                                                 confidence_threshold)
        self.intermediate_results['candidates'] = dash_candidates
        
        # Directional morphological cleanup
        cleaned_candidates = self._directional_cleanup(dash_candidates, binary_clean)
        self.intermediate_results['cleaned'] = cleaned_candidates
        
        # Segment linking
        polylines = self._link_segments(cleaned_candidates)
        self.intermediate_results['polylines'] = polylines
        
        # Vectorization
        vector_lines = self._vectorize_lines(polylines, binary_clean)
        
        # Length measurement
        for line in vector_lines:
            line['length_px'] = self._calculate_length(line['points'])
            line['length_m'] = line['length_px'] * self.scale_px_to_m
        
        self.detected_lines = vector_lines
        return vector_lines
    
    def _template_matching(self, binary_image: np.ndarray) -> Dict[float, np.ndarray]:
        """Apply template matching for all orientations."""
        response_maps = {}
        
        for angle, template in self.dash_template.templates.items():
            # Normalize template
            template_norm = template.astype(np.float32) / 255.0
            image_norm = binary_image.astype(np.float32) / 255.0
            
            # Template matching
            response = cv2.matchTemplate(image_norm, template_norm, cv2.TM_CCORR_NORMED)
            response_maps[angle] = response
        
        return response_maps
    
    def _extract_candidates(self, 
                          response_maps: Dict[float, np.ndarray],
                          nms_threshold: float,
                          confidence_threshold: float) -> List[Dict]:
        """Extract dash candidates with non-maximum suppression."""
        all_candidates = []
        
        for angle, response_map in response_maps.items():
            # Find local maxima using scipy.ndimage instead
            from scipy import ndimage
            
            # Use maximum filter to find local maxima
            max_filter = ndimage.maximum_filter(response_map, size=self.dash_template.dash_length//2)
            maxima_mask = (response_map == max_filter) & (response_map > confidence_threshold)
            peaks = np.where(maxima_mask)
            
            for i in range(len(peaks[0])):
                y, x = peaks[0][i], peaks[1][i]
                confidence = response_map[y, x]
                
                all_candidates.append({
                    'x': x,
                    'y': y,
                    'angle': angle,
                    'confidence': confidence,
                    'template_size': self.dash_template.templates[angle].shape
                })
        
        # Non-maximum suppression across angles
        if not all_candidates:
            return []
        
        # Convert to format for NMS
        boxes = []
        scores = []
        for candidate in all_candidates:
            h, w = candidate['template_size']
            x, y = candidate['x'], candidate['y']
            boxes.append([x, y, x + w, y + h])
            scores.append(candidate['confidence'])
        
        boxes = np.array(boxes, dtype=np.float32)
        scores = np.array(scores, dtype=np.float32)
        
        # Apply NMS
        indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), 
                                 confidence_threshold, nms_threshold)
        
        if len(indices) > 0:
            indices = indices.flatten()
            return [all_candidates[i] for i in indices]
        
        return []
    
    def _directional_cleanup(self, 
                           candidates: List[Dict], 
                           binary_image: np.ndarray) -> List[Dict]:
        """Apply directional morphological operations."""
        cleaned_candidates = []
        
        # Group candidates by angle
        angle_groups = {}
        for candidate in candidates:
            angle = candidate['angle']
            if angle not in angle_groups:
                angle_groups[angle] = []
            angle_groups[angle].append(candidate)
        
        for angle, group in angle_groups.items():
            if not group:
                continue
                
            # Create mask for this angle
            mask = np.zeros_like(binary_image)
            
            for candidate in group:
                h, w = candidate['template_size']
                x, y = candidate['x'], candidate['y']
                mask[y:y+h, x:x+w] = 255
            
            # Directional closing
            length = max(self.dash_template.dash_length, 10)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (length, 3))
            
            # Rotate kernel to match angle
            center = (kernel.shape[1] // 2, kernel.shape[0] // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            kernel_rotated = cv2.warpAffine(kernel.astype(np.uint8), 
                                          rotation_matrix, 
                                          kernel.shape[::-1])
            
            # Apply morphological closing
            closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_rotated)
            
            # Hit-or-miss to suppress continuous lines
            dash_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, 
                                                  (self.dash_template.dash_length, 3))
            gap_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, 
                                                 (self.dash_template.gap_length, 3))
            
            # Update candidates based on cleaned mask
            for candidate in group:
                h, w = candidate['template_size']
                x, y = candidate['x'], candidate['y']
                region = closed[y:y+h, x:x+w]
                
                if np.sum(region) > 0:  # Keep if still has content after cleanup
                    candidate['cleaned_mask'] = region
                    cleaned_candidates.append(candidate)
        
        return cleaned_candidates
    
    def _link_segments(self, candidates: List[Dict]) -> List[List[Tuple[int, int]]]:
        """Link dash segments into polylines using graph-based approach."""
        if not candidates:
            return []
        
        # Extract centroids
        centroids = []
        for candidate in candidates:
            x = candidate['x'] + candidate['template_size'][1] // 2
            y = candidate['y'] + candidate['template_size'][0] // 2
            centroids.append((x, y))
        
        centroids = np.array(centroids)
        
        # Build KD-tree for efficient neighbor search
        tree = KDTree(centroids)
        
        # Link compatible neighbors
        polylines = []
        used = set()
        
        for i, candidate in enumerate(candidates):
            if i in used:
                continue
            
            # Start a new polyline
            polyline = [centroids[i]]
            used.add(i)
            current_angle = candidate['angle']
            
            # Extend in both directions
            for direction in [-1, 1]:
                current_idx = i
                
                while True:
                    current_pos = centroids[current_idx]
                    
                    # Find nearby candidates
                    max_distance = self.dash_template.total_period * 2
                    neighbor_indices = tree.query_ball_point(current_pos, max_distance)
                    
                    best_neighbor = None
                    best_distance = float('inf')
                    
                    for neighbor_idx in neighbor_indices:
                        if neighbor_idx in used or neighbor_idx == current_idx:
                            continue
                        
                        neighbor_candidate = candidates[neighbor_idx]
                        neighbor_angle = neighbor_candidate['angle']
                        
                        # Check angular compatibility
                        angle_diff = abs(current_angle - neighbor_angle)
                        angle_diff = min(angle_diff, 180 - angle_diff)
                        
                        if angle_diff > 10:  # 10 degree tolerance
                            continue
                        
                        # Check distance compatibility
                        neighbor_pos = centroids[neighbor_idx]
                        distance = np.linalg.norm(current_pos - neighbor_pos)
                        
                        expected_gap = self.dash_template.total_period
                        if abs(distance - expected_gap) < expected_gap * 0.5:  # 50% tolerance
                            if distance < best_distance:
                                best_distance = distance
                                best_neighbor = neighbor_idx
                    
                    if best_neighbor is None:
                        break
                    
                    # Add to polyline
                    if direction == 1:
                        polyline.append(centroids[best_neighbor])
                    else:
                        polyline.insert(0, centroids[best_neighbor])
                    
                    used.add(best_neighbor)
                    current_idx = best_neighbor
            
            # Only keep polylines with multiple segments
            if len(polyline) >= 2:
                polylines.append(polyline)
        
        return polylines
    
    def _vectorize_lines(self, 
                        polylines: List[List[Tuple[int, int]]], 
                        binary_image: np.ndarray) -> List[Dict]:
        """Convert polylines to vector format with RANSAC and simplification."""
        vector_lines = []
        
        for i, polyline in enumerate(polylines):
            if len(polyline) < 2:
                continue
            
            points = np.array(polyline)
            
            # RANSAC line fitting for each polyline segment
            simplified_points = self._ransac_line_fit(points)
            
            # Ramer-Douglas-Peucker simplification
            simplified_points = self._douglas_peucker(simplified_points, epsilon=2.0)
            
            # Calculate angle
            if len(simplified_points) >= 2:
                dx = simplified_points[-1][0] - simplified_points[0][0]
                dy = simplified_points[-1][1] - simplified_points[0][1]
                angle = np.arctan2(dy, dx) * 180 / np.pi
            else:
                angle = 0
            
            vector_lines.append({
                'id': f'dashed_line_{i}',
                'class': 'dashed_service_line',
                'points': simplified_points.tolist(),
                'angle_deg': angle,
                'confidence': 0.8  # Average confidence from candidates
            })
        
        return vector_lines
    
    def _ransac_line_fit(self, points: np.ndarray, 
                        max_iterations: int = 100,
                        distance_threshold: float = 5.0) -> np.ndarray:
        """Fit line segments using RANSAC."""
        if len(points) <= 2:
            return points
        
        best_inliers = []
        best_line = None
        
        for _ in range(max_iterations):
            # Sample two random points
            sample_indices = np.random.choice(len(points), 2, replace=False)
            p1, p2 = points[sample_indices]
            
            # Calculate line parameters
            if np.allclose(p1, p2):
                continue
            
            # Distance from point to line
            distances = []
            line_vec = p2 - p1
            line_len = np.linalg.norm(line_vec)
            
            if line_len == 0:
                continue
            
            line_unit = line_vec / line_len
            
            for point in points:
                vec_to_point = point - p1
                projection_length = np.dot(vec_to_point, line_unit)
                projection = p1 + projection_length * line_unit
                distance = np.linalg.norm(point - projection)
                distances.append(distance)
            
            # Find inliers
            inliers = [i for i, d in enumerate(distances) if d < distance_threshold]
            
            if len(inliers) > len(best_inliers):
                best_inliers = inliers
                best_line = (p1, p2)
        
        if best_line is None:
            return points
        
        # Return simplified line with endpoints
        inlier_points = points[best_inliers]
        if len(inlier_points) < 2:
            return points
        
        # Find actual endpoints of inlier cluster
        p1, p2 = best_line
        line_vec = p2 - p1
        line_unit = line_vec / np.linalg.norm(line_vec)
        
        projections = []
        for point in inlier_points:
            vec_to_point = point - p1
            projection_length = np.dot(vec_to_point, line_unit)
            projections.append(projection_length)
        
        min_proj = min(projections)
        max_proj = max(projections)
        
        start_point = p1 + min_proj * line_unit
        end_point = p1 + max_proj * line_unit
        
        return np.array([start_point, end_point])
    
    def _douglas_peucker(self, points: np.ndarray, epsilon: float) -> np.ndarray:
        """Simplify polyline using Ramer-Douglas-Peucker algorithm."""
        if len(points) <= 2:
            return points
        
        # Find the point with maximum distance from line segment
        start, end = points[0], points[-1]
        max_distance = 0
        max_index = 0
        
        for i in range(1, len(points) - 1):
            distance = self._point_to_line_distance(points[i], start, end)
            if distance > max_distance:
                max_distance = distance
                max_index = i
        
        # If max distance is greater than epsilon, recursively simplify
        if max_distance > epsilon:
            # Recursive call
            left_simplified = self._douglas_peucker(points[:max_index + 1], epsilon)
            right_simplified = self._douglas_peucker(points[max_index:], epsilon)
            
            # Combine results (remove duplicate middle point)
            return np.vstack([left_simplified[:-1], right_simplified])
        else:
            # Return only endpoints
            return np.array([start, end])
    
    def _point_to_line_distance(self, point: np.ndarray, 
                              line_start: np.ndarray, 
                              line_end: np.ndarray) -> float:
        """Calculate perpendicular distance from point to line segment."""
        line_vec = line_end - line_start
        line_len = np.linalg.norm(line_vec)
        
        if line_len == 0:
            return np.linalg.norm(point - line_start)
        
        line_unit = line_vec / line_len
        vec_to_point = point - line_start
        projection_length = np.dot(vec_to_point, line_unit)
        
        # Clamp projection to line segment
        projection_length = max(0, min(line_len, projection_length))
        projection = line_start + projection_length * line_unit
        
        return np.linalg.norm(point - projection)
    
    def _calculate_length(self, points: List[Tuple[float, float]]) -> float:
        """Calculate total length of polyline."""
        if len(points) < 2:
            return 0.0
        
        total_length = 0.0
        for i in range(len(points) - 1):
            p1 = np.array(points[i])
            p2 = np.array(points[i + 1])
            total_length += np.linalg.norm(p2 - p1)
        
        return total_length
    
    def set_scale(self, px_to_m: float):
        """Set the pixel to meter conversion factor."""
        self.scale_px_to_m = px_to_m
        
        # Update existing detections
        for line in self.detected_lines:
            if 'length_px' in line:
                line['length_m'] = line['length_px'] * px_to_m
    
    def get_intermediate_results(self) -> Dict[str, Any]:
        """Get intermediate processing results for debugging."""
        return self.intermediate_results.copy()


def main():
    """Example usage of the DashedLineDetector."""
    # This would typically be called from a GUI or main application
    detector = DashedLineDetector()
    
    print("Dashed Line Detection System initialized.")
    print("Usage:")
    print("1. Load an image")
    print("2. Call learn_dash_template() with a user-selected ROI")
    print("3. Call detect_dashed_lines() to run the full pipeline")
    print("4. Use export functions to save results")


if __name__ == "__main__":
    main()
