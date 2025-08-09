"""
Integrated Dashed Line Detection System

This is the main module that integrates all components:
- Classic CV pipeline with template matching
- Robust detection methods (LSD, Gabor, frequency analysis)
- Scale detection and measurement
- Legend-based classification
- Multiple output formats

Usage:
    detector = IntegratedDashDetector()
    detector.load_image("floor_plan.jpg")
    detector.learn_template_from_roi(x, y, width, height)
    results = detector.detect_all()
    detector.export_results("output_folder")
"""

import cv2
import numpy as np
import os
from typing import List, Dict, Tuple, Optional, Any
import json
import warnings
warnings.filterwarnings('ignore')

# Import our modules
from dash_line_detector import DashedLineDetector, FloorPlanPreprocessor
from robust_detectors import RobustDashDetector
from scale_detector import ScaleDetector
from legend_classifier import LegendReader
from output_exporter import OutputExporter


class IntegratedDashDetector:
    """Main integrated system for dashed line detection."""
    
    def __init__(self):
        # Core components
        self.classic_detector = DashedLineDetector()
        self.robust_detector = RobustDashDetector()
        self.scale_detector = ScaleDetector()
        self.legend_reader = LegendReader()
        
        # Image data
        self.original_image = None
        self.processed_image = None
        self.image_path = None
        
        # Detection results
        self.classic_results = []
        self.robust_results = {}
        self.combined_results = []
        self.legend_entries = []
        
        # Configuration
        self.config = {
            'preprocessing': {
                'apply_deskew': True,
                'apply_clahe': True,
                'remove_text': True
            },
            'detection': {
                'use_classic': True,
                'use_robust': True,
                'combine_methods': True,
                'consensus_threshold': 2
            },
            'classification': {
                'use_legend': True,
                'min_confidence': 0.3
            },
            'output': {
                'export_svg': True,
                'export_geojson': True,
                'export_dxf': True,
                'export_csv': True,
                'export_overlay': True
            }
        }
    
    def load_image(self, image_path: str) -> bool:
        """
        Load and preprocess floor plan image.
        
        Args:
            image_path: Path to the floor plan image
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.original_image = cv2.imread(image_path)
            if self.original_image is None:
                print(f"Error: Could not load image from {image_path}")
                return False
            
            self.image_path = image_path
            
            # Preprocess image
            preprocessor = FloorPlanPreprocessor()
            self.processed_image, _ = preprocessor.preprocess(
                self.original_image,
                apply_deskew=self.config['preprocessing']['apply_deskew'],
                apply_clahe=self.config['preprocessing']['apply_clahe']
            )
            
            print(f"Image loaded successfully: {image_path}")
            print(f"Image size: {self.original_image.shape}")
            
            return True
            
        except Exception as e:
            print(f"Error loading image: {e}")
            return False
    
    def learn_template_from_roi(self, x: int, y: int, width: int, height: int) -> bool:
        """
        Learn dash template from user-selected ROI.
        
        Args:
            x, y: Top-left corner of ROI
            width, height: ROI dimensions
            
        Returns:
            True if successful, False otherwise
        """
        if self.processed_image is None:
            print("Error: No image loaded")
            return False
        
        try:
            # Validate ROI bounds
            h, w = self.processed_image.shape[:2]
            if x < 0 or y < 0 or x + width > w or y + height > h:
                print("Error: ROI out of image bounds")
                return False
            
            roi = (x, y, width, height)
            self.classic_detector.learn_dash_template(self.processed_image, roi)
            
            template_info = self.classic_detector.dash_template
            print(f"Template learned successfully:")
            print(f"  Dash length: {template_info.dash_length} px")
            print(f"  Gap length: {template_info.gap_length} px")
            print(f"  Total period: {template_info.total_period} px")
            
            return True
            
        except Exception as e:
            print(f"Error learning template: {e}")
            return False
    
    def detect_scale(self) -> bool:
        """
        Detect scale information from the image.
        
        Returns:
            True if scale detected, False otherwise
        """
        if self.original_image is None:
            print("Error: No image loaded")
            return False
        
        try:
            scale_factor, method = self.scale_detector.detect_scale(self.original_image)
            self.classic_detector.set_scale(scale_factor)
            
            print(f"Scale detected: {scale_factor:.6f} px/m (method: {method})")
            
            return True
            
        except Exception as e:
            print(f"Error detecting scale: {e}")
            return False
    
    def set_manual_scale(self, point1: Tuple[int, int], point2: Tuple[int, int], 
                        distance_m: float) -> bool:
        """
        Set scale manually using two points with known distance.
        
        Args:
            point1, point2: Two points on the image
            distance_m: Known distance between points in meters
            
        Returns:
            True if successful, False otherwise
        """
        try:
            scale_factor = self.scale_detector.set_manual_scale(point1, point2, distance_m)
            self.classic_detector.set_scale(scale_factor)
            
            print(f"Manual scale set: {scale_factor:.6f} px/m")
            return True
            
        except Exception as e:
            print(f"Error setting manual scale: {e}")
            return False
    
    def detect_legend(self) -> bool:
        """
        Detect and parse legend information.
        
        Returns:
            True if legend found, False otherwise
        """
        if self.original_image is None:
            print("Error: No image loaded")
            return False
        
        try:
            # Detect legend region
            legend_region = self.legend_reader.detect_legend_region(self.original_image)
            
            if legend_region is None:
                print("Warning: No legend region detected")
                return False
            
            # Parse legend entries
            self.legend_entries = self.legend_reader.parse_legend(
                self.original_image, legend_region
            )
            
            print(f"Legend detected with {len(self.legend_entries)} entries:")
            for i, entry in enumerate(self.legend_entries):
                print(f"  {i+1}. {entry.get('text', 'N/A')} - {entry.get('line_type', 'unknown')}")
            
            return True
            
        except Exception as e:
            print(f"Error detecting legend: {e}")
            return False
    
    def detect_classic(self) -> bool:
        """
        Run classic template-based detection.
        
        Returns:
            True if successful, False otherwise
        """
        if self.processed_image is None:
            print("Error: No image loaded")
            return False
        
        if self.classic_detector.dash_template is None:
            print("Error: No template learned. Call learn_template_from_roi() first.")
            return False
        
        try:
            print("Running classic template-based detection...")
            
            self.classic_results = self.classic_detector.detect_dashed_lines(
                self.original_image,
                remove_text=self.config['preprocessing']['remove_text']
            )
            
            print(f"Classic detection found {len(self.classic_results)} dashed lines")
            
            return True
            
        except Exception as e:
            print(f"Error in classic detection: {e}")
            return False
    
    def detect_robust(self) -> bool:
        """
        Run robust detection methods.
        
        Returns:
            True if successful, False otherwise
        """
        if self.processed_image is None:
            print("Error: No image loaded")
            return False
        
        try:
            print("Running robust detection methods...")
            
            # Get expected period from template if available
            expected_period = None
            if self.classic_detector.dash_template is not None:
                expected_period = self.classic_detector.dash_template.total_period
            
            self.robust_results = self.robust_detector.detect_all_methods(
                self.processed_image, expected_period
            )
            
            total_detections = sum(len(results) for results in self.robust_results.values())
            print(f"Robust detection found {total_detections} total detections:")
            for method, results in self.robust_results.items():
                print(f"  {method}: {len(results)} detections")
            
            return True
            
        except Exception as e:
            print(f"Error in robust detection: {e}")
            return False
    
    def combine_detections(self) -> bool:
        """
        Combine results from different detection methods.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            print("Combining detection results...")
            
            all_detections = []
            
            # Add classic results
            if self.config['detection']['use_classic'] and self.classic_results:
                for line in self.classic_results:
                    line['detection_method'] = 'classic'
                    all_detections.append(line)
            
            # Add robust results
            if self.config['detection']['use_robust'] and self.robust_results:
                if self.config['detection']['combine_methods']:
                    # Use consensus from robust detector
                    consensus_lines = self.robust_detector.combine_detections(
                        self.robust_results, 
                        self.config['detection']['consensus_threshold']
                    )
                    all_detections.extend(consensus_lines)
                else:
                    # Add all robust detections
                    for method, results in self.robust_results.items():
                        for line in results:
                            line['detection_method'] = method
                            all_detections.append(line)
            
            # Remove duplicates and merge similar detections
            self.combined_results = self._merge_similar_detections(all_detections)
            
            print(f"Combined results: {len(self.combined_results)} unique dashed lines")
            
            return True
            
        except Exception as e:
            print(f"Error combining detections: {e}")
            return False
    
    def classify_lines(self) -> bool:
        """
        Classify detected lines using legend information.
        
        Returns:
            True if successful, False otherwise
        """
        if not self.combined_results:
            print("Error: No detection results to classify")
            return False
        
        try:
            print("Classifying detected lines...")
            
            classified_count = 0
            
            for line in self.combined_results:
                # Extract pattern signature if available
                pattern_signature = line.get('pattern_signature', [])
                if not pattern_signature and 'points' in line:
                    # Generate signature from points if not available
                    pattern_signature = self._generate_pattern_signature(line)
                
                if pattern_signature:
                    # Classify using legend
                    if self.config['classification']['use_legend'] and self.legend_entries:
                        classification, confidence, legend_entry = \
                            self.legend_reader.classify_detected_line(
                                np.array(pattern_signature), 
                                line
                            )
                    else:
                        # Fallback to pattern classification
                        classification, confidence = \
                            self.legend_reader.pattern_classifier.classify_pattern(
                                np.array(pattern_signature)
                            )
                        legend_entry = {}
                    
                    if confidence >= self.config['classification']['min_confidence']:
                        line['classified_type'] = classification
                        line['classification_confidence'] = confidence
                        line['legend_entry'] = legend_entry
                        classified_count += 1
                    else:
                        line['classified_type'] = 'unknown'
                        line['classification_confidence'] = confidence
            
            print(f"Successfully classified {classified_count}/{len(self.combined_results)} lines")
            
            return True
            
        except Exception as e:
            print(f"Error classifying lines: {e}")
            return False
    
    def detect_all(self) -> List[Dict]:
        """
        Run complete detection pipeline.
        
        Returns:
            List of detected and classified line dictionaries
        """
        if self.original_image is None:
            print("Error: No image loaded")
            return []
        
        print("Starting complete detection pipeline...")
        
        # Step 1: Detect scale
        self.detect_scale()
        
        # Step 2: Detect legend
        self.detect_legend()
        
        # Step 3: Classic detection (if template is available)
        if (self.config['detection']['use_classic'] and 
            self.classic_detector.dash_template is not None):
            self.detect_classic()
        
        # Step 4: Robust detection
        if self.config['detection']['use_robust']:
            self.detect_robust()
        
        # Step 5: Combine results
        self.combine_detections()
        
        # Step 6: Classify lines
        self.classify_lines()
        
        print(f"Detection pipeline complete. Found {len(self.combined_results)} dashed lines.")
        
        return self.combined_results
    
    def export_results(self, output_dir: str = "output") -> Dict[str, str]:
        """
        Export results to all supported formats.
        
        Args:
            output_dir: Output directory
            
        Returns:
            Dictionary mapping format names to output file paths
        """
        if not self.combined_results:
            print(f"Error: No results to export. Combined results: {len(self.combined_results) if self.combined_results else 0}")
            return {}
        
        if self.original_image is None:
            print("Error: No original image available")
            return {}
        
        try:
            # Create exporter
            h, w = self.original_image.shape[:2]
            exporter = OutputExporter(w, h)
            
            # Generate base filename
            if self.image_path:
                base_name = os.path.splitext(os.path.basename(self.image_path))[0]
            else:
                base_name = "detected_lines"
            
            # Export all formats
            output_files = exporter.export_all_formats(
                self.combined_results,
                self.original_image,
                base_name,
                output_dir
            )
            
            # Export configuration and metadata
            metadata = {
                'image_path': self.image_path,
                'image_size': [w, h],
                'detection_config': self.config,
                'scale_info': self.scale_detector.get_scale_info(),
                'legend_entries': len(self.legend_entries),
                'total_detections': len(self.combined_results),
                'detection_summary': self._generate_detection_summary()
            }
            
            metadata_file = os.path.join(output_dir, f"{base_name}_metadata.json")
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            output_files['metadata'] = metadata_file
            
            print(f"Results exported to: {output_dir}")
            return output_files
            
        except Exception as e:
            print(f"Error exporting results: {e}")
            return {}
    
    def _merge_similar_detections(self, detections: List[Dict]) -> List[Dict]:
        """Merge similar detections from different methods."""
        if not detections:
            return []
        
        # Simple implementation: remove exact duplicates and very close lines
        unique_detections = []
        
        for detection in detections:
            is_duplicate = False
            
            for existing in unique_detections:
                if self._are_similar_detections(detection, existing):
                    # Merge properties
                    self._merge_detection_properties(existing, detection)
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_detections.append(detection)
        
        return unique_detections
    
    def _are_similar_detections(self, det1: Dict, det2: Dict, 
                               distance_threshold: float = 20.0) -> bool:
        """Check if two detections are similar enough to be considered the same line."""
        # Compare by endpoints if available
        if 'points' in det1 and 'points' in det2:
            points1 = det1['points']
            points2 = det2['points']
            
            if len(points1) >= 2 and len(points2) >= 2:
                # Compare start and end points
                start1, end1 = points1[0], points1[-1]
                start2, end2 = points2[0], points2[-1]
                
                dist_start = np.sqrt((start1[0] - start2[0])**2 + (start1[1] - start2[1])**2)
                dist_end = np.sqrt((end1[0] - end2[0])**2 + (end1[1] - end2[1])**2)
                
                return (dist_start < distance_threshold and dist_end < distance_threshold)
        
        # Compare by start/end if available
        if ('start' in det1 and 'end' in det1 and 
            'start' in det2 and 'end' in det2):
            start1, end1 = det1['start'], det1['end']
            start2, end2 = det2['start'], det2['end']
            
            dist_start = np.sqrt((start1[0] - start2[0])**2 + (start1[1] - start2[1])**2)
            dist_end = np.sqrt((end1[0] - end2[0])**2 + (end1[1] - end2[1])**2)
            
            return (dist_start < distance_threshold and dist_end < distance_threshold)
        
        return False
    
    def _merge_detection_properties(self, target: Dict, source: Dict) -> None:
        """Merge properties from source detection into target."""
        # Combine detection methods
        target_methods = target.get('detection_methods', [target.get('detection_method', 'unknown')])
        source_methods = source.get('detection_methods', [source.get('detection_method', 'unknown')])
        
        all_methods = list(set(target_methods + source_methods))
        target['detection_methods'] = all_methods
        target['method_consensus'] = len(all_methods)
        
        # Average confidence scores
        confidence_props = ['confidence', 'period_score', 'periodic_score', 'classification_confidence']
        for prop in confidence_props:
            if prop in target and prop in source:
                target[f'avg_{prop}'] = (target[prop] + source[prop]) / 2
                target[prop] = max(target[prop], source[prop])  # Keep higher confidence
            elif prop in source:
                target[prop] = source[prop]
    
    def _generate_pattern_signature(self, line: Dict) -> List[float]:
        """Generate pattern signature from line points."""
        if 'points' not in line or len(line['points']) < 2:
            return []
        
        # Simple implementation: just return a dashed pattern signature
        # In a real implementation, you would sample the actual image along the line
        return [1.0, 0.0, 1.0, 0.0]  # Basic dashed pattern
    
    def _generate_detection_summary(self) -> Dict[str, Any]:
        """Generate summary statistics of detection results."""
        if not self.combined_results:
            return {}
        
        summary = {
            'total_lines': len(self.combined_results),
            'total_length_m': sum(line.get('length_m', 0) for line in self.combined_results),
            'line_types': {},
            'detection_methods': {},
            'confidence_stats': {}
        }
        
        # Count by line type
        for line in self.combined_results:
            line_type = line.get('classified_type', line.get('class', 'unknown'))
            summary['line_types'][line_type] = summary['line_types'].get(line_type, 0) + 1
        
        # Count by detection method
        for line in self.combined_results:
            methods = line.get('detection_methods', [line.get('detection_method', 'unknown')])
            for method in methods:
                summary['detection_methods'][method] = summary['detection_methods'].get(method, 0) + 1
        
        # Confidence statistics
        confidences = [line.get('confidence', 0) for line in self.combined_results if 'confidence' in line]
        if confidences:
            summary['confidence_stats'] = {
                'mean': np.mean(confidences),
                'std': np.std(confidences),
                'min': np.min(confidences),
                'max': np.max(confidences)
            }
        
        return summary
    
    def get_detection_statistics(self) -> Dict[str, Any]:
        """Get comprehensive detection statistics."""
        return {
            'image_info': {
                'path': self.image_path,
                'size': self.original_image.shape if self.original_image is not None else None
            },
            'scale_info': self.scale_detector.get_scale_info(),
            'legend_info': {
                'entries_found': len(self.legend_entries),
                'entries': [{'text': e.get('text', 'N/A'), 'type': e.get('line_type', 'unknown')} 
                           for e in self.legend_entries]
            },
            'detection_summary': self._generate_detection_summary(),
            'config': self.config
        }


def main():
    """Example usage of the integrated detection system."""
    print("Integrated Dashed Line Detection System")
    print("======================================")
    
    # Example usage
    detector = IntegratedDashDetector()
    
    print("\nUsage:")
    print("1. detector.load_image('path/to/floor_plan.jpg')")
    print("2. detector.learn_template_from_roi(x, y, width, height)  # Optional")
    print("3. results = detector.detect_all()")
    print("4. detector.export_results('output_folder')")
    
    print("\nFor manual scale setting:")
    print("detector.set_manual_scale((x1, y1), (x2, y2), distance_in_meters)")
    
    print("\nConfiguration options available in detector.config")


if __name__ == "__main__":
    main()
