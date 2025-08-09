"""
Robust Detection Methods for Dashed Lines

This module implements advanced detection techniques:
- Line Segment Detector (LSD) with periodicity filtering
- Frequency/periodicity analysis using FFT and autocorrelation
- Gabor and steerable filters
- Machine learning segmentation approaches
"""

import cv2
import numpy as np
from scipy import ndimage, signal, fft
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import DBSCAN
from skimage import filters, morphology, segmentation
from skimage import measure  # Add measure for regionprops
from skimage.filters import gabor
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Optional, Any
import warnings
warnings.filterwarnings('ignore')


class LineSegmentDetector:
    """Enhanced Line Segment Detector with periodicity filtering."""
    
    def __init__(self):
        self.lsd = cv2.createLineSegmentDetector()
        self.min_line_length = 50
        self.max_line_gap = 10
        
    def detect_line_segments(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect line segments using LSD algorithm.
        
        Args:
            image: Input binary image
            
        Returns:
            List of line segments as (x1, y1, x2, y2) tuples
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply LSD
        lines = self.lsd.detect(gray)[0]
        
        if lines is None:
            return []
        
        # Convert to integer coordinates and filter by length
        line_segments = []
        for line in lines:
            x1, y1, x2, y2 = line[0].astype(int)
            length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            
            if length >= self.min_line_length:
                line_segments.append((x1, y1, x2, y2))
        
        return line_segments
    
    def filter_by_periodicity(self, 
                             image: np.ndarray,
                             line_segments: List[Tuple[int, int, int, int]],
                             expected_period: float = None,
                             tolerance: float = 0.3) -> List[Dict]:
        """
        Filter line segments by dash periodicity along their axis.
        
        Args:
            image: Input image
            line_segments: List of line segments
            expected_period: Expected dash period in pixels
            tolerance: Tolerance for period matching
            
        Returns:
            List of dashed line dictionaries with periodicity scores
        """
        dashed_lines = []
        
        for i, (x1, y1, x2, y2) in enumerate(line_segments):
            # Sample intensities along line
            intensities = self._sample_line_intensities(image, x1, y1, x2, y2)
            
            if len(intensities) < 20:  # Need sufficient samples
                continue
            
            # Analyze periodicity
            period_score, detected_period = self._analyze_periodicity(intensities, expected_period)
            
            if period_score > 0.3:  # Threshold for dash-like patterns
                line_dict = {
                    'start': (x1, y1),
                    'end': (x2, y2),
                    'length': np.sqrt((x2 - x1)**2 + (y2 - y1)**2),
                    'angle': np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi,
                    'period_score': period_score,
                    'detected_period': detected_period,
                    'intensities': intensities,
                    'id': f'lsd_line_{i}'
                }
                dashed_lines.append(line_dict)
        
        return dashed_lines
    
    def _sample_line_intensities(self, 
                                image: np.ndarray,
                                x1: int, y1: int, x2: int, y2: int,
                                num_samples: int = None) -> np.ndarray:
        """Sample pixel intensities along a line segment."""
        if num_samples is None:
            length = int(np.sqrt((x2 - x1)**2 + (y2 - y1)**2))
            num_samples = max(length, 20)
        
        # Generate sample points along line
        t = np.linspace(0, 1, num_samples)
        x_samples = x1 + t * (x2 - x1)
        y_samples = y1 + t * (y2 - y1)
        
        # Ensure coordinates are within image bounds
        x_samples = np.clip(x_samples, 0, image.shape[1] - 1).astype(int)
        y_samples = np.clip(y_samples, 0, image.shape[0] - 1).astype(int)
        
        # Sample intensities
        intensities = image[y_samples, x_samples]
        
        return intensities.astype(np.float32)
    
    def _analyze_periodicity(self, 
                           intensities: np.ndarray,
                           expected_period: Optional[float] = None) -> Tuple[float, float]:
        """
        Analyze periodicity in intensity signal using FFT and autocorrelation.
        
        Args:
            intensities: 1D intensity signal
            expected_period: Expected period in samples
            
        Returns:
            Tuple of (periodicity_score, detected_period)
        """
        if len(intensities) < 10:
            return 0.0, 0.0
        
        # Normalize intensities
        intensities = intensities - np.mean(intensities)
        if np.std(intensities) > 0:
            intensities = intensities / np.std(intensities)
        
        # Method 1: FFT-based frequency analysis
        fft_score, fft_period = self._fft_periodicity(intensities, expected_period)
        
        # Method 2: Autocorrelation-based analysis
        autocorr_score, autocorr_period = self._autocorrelation_periodicity(intensities, expected_period)
        
        # Combine scores
        combined_score = max(fft_score, autocorr_score)
        best_period = fft_period if fft_score > autocorr_score else autocorr_period
        
        return combined_score, best_period
    
    def _fft_periodicity(self, 
                        intensities: np.ndarray,
                        expected_period: Optional[float] = None) -> Tuple[float, float]:
        """Analyze periodicity using FFT."""
        # Apply window to reduce edge effects
        windowed = intensities * np.hanning(len(intensities))
        
        # Compute FFT
        fft_result = np.fft.fft(windowed)
        magnitude = np.abs(fft_result)
        
        # Focus on positive frequencies (excluding DC)
        freqs = np.fft.fftfreq(len(intensities))
        positive_freqs = freqs[1:len(freqs)//2]
        positive_magnitude = magnitude[1:len(magnitude)//2]
        
        if len(positive_magnitude) == 0:
            return 0.0, 0.0
        
        # Find dominant frequency
        peak_idx = np.argmax(positive_magnitude)
        dominant_freq = positive_freqs[peak_idx]
        peak_magnitude = positive_magnitude[peak_idx]
        
        # Convert frequency to period
        if dominant_freq != 0:
            detected_period = 1.0 / abs(dominant_freq)
        else:
            detected_period = 0.0
        
        # Calculate periodicity score
        # High score if there's a strong peak relative to noise
        noise_level = np.median(positive_magnitude)
        signal_to_noise = peak_magnitude / (noise_level + 1e-8)
        
        # Normalize score
        fft_score = min(signal_to_noise / 10.0, 1.0)
        
        # Bonus if period matches expected
        if expected_period is not None and detected_period > 0:
            period_error = abs(detected_period - expected_period) / expected_period
            if period_error < 0.5:  # Within 50%
                fft_score *= (1.0 + (1.0 - period_error))
        
        return fft_score, detected_period
    
    def _autocorrelation_periodicity(self, 
                                   intensities: np.ndarray,
                                   expected_period: Optional[float] = None) -> Tuple[float, float]:
        """Analyze periodicity using autocorrelation."""
        # Compute autocorrelation
        autocorr = np.correlate(intensities, intensities, mode='full')
        autocorr = autocorr[len(autocorr)//2:]  # Keep positive lags only
        
        # Normalize
        if len(autocorr) > 0 and autocorr[0] != 0:
            autocorr = autocorr / autocorr[0]
        
        # Find peaks in autocorrelation (ignoring lag 0)
        if len(autocorr) < 3:
            return 0.0, 0.0
        
        # Look for the first significant peak after lag 0
        min_lag = max(3, int(len(intensities) * 0.1))  # Minimum 10% of signal length
        max_lag = min(len(autocorr) - 1, int(len(intensities) * 0.8))  # Maximum 80%
        
        if min_lag >= max_lag:
            return 0.0, 0.0
        
        search_region = autocorr[min_lag:max_lag]
        
        if len(search_region) == 0:
            return 0.0, 0.0
        
        # Find peaks
        peaks, _ = signal.find_peaks(search_region, height=0.1, distance=5)
        
        if len(peaks) == 0:
            return 0.0, 0.0
        
        # Get the strongest peak
        peak_values = search_region[peaks]
        best_peak_idx = np.argmax(peak_values)
        best_peak_lag = peaks[best_peak_idx] + min_lag
        best_peak_value = peak_values[best_peak_idx]
        
        # Calculate score based on peak strength
        autocorr_score = min(best_peak_value * 2.0, 1.0)  # Scale and cap at 1.0
        
        # Detected period is the lag of the best peak
        detected_period = float(best_peak_lag)
        
        # Bonus if period matches expected
        if expected_period is not None and detected_period > 0:
            period_error = abs(detected_period - expected_period) / expected_period
            if period_error < 0.5:  # Within 50%
                autocorr_score *= (1.0 + (1.0 - period_error))
        
        return autocorr_score, detected_period


class GaborFilterBank:
    """Gabor filter bank for detecting oriented line patterns."""
    
    def __init__(self, 
                 orientations: List[float] = None,
                 frequencies: List[float] = None,
                 sigma_x: float = 2.0,
                 sigma_y: float = 2.0):
        """
        Initialize Gabor filter bank.
        
        Args:
            orientations: List of orientation angles in degrees
            frequencies: List of spatial frequencies
            sigma_x: Standard deviation in x direction
            sigma_y: Standard deviation in y direction
        """
        if orientations is None:
            orientations = np.arange(0, 180, 15)  # Every 15 degrees
        if frequencies is None:
            frequencies = [0.1, 0.2, 0.3, 0.4]  # Different spatial frequencies
        
        self.orientations = orientations
        self.frequencies = frequencies
        self.sigma_x = sigma_x
        self.sigma_y = sigma_y
        self.responses = {}
    
    def apply_filters(self, image: np.ndarray) -> Dict[Tuple[float, float], np.ndarray]:
        """
        Apply Gabor filter bank to image.
        
        Args:
            image: Input grayscale image
            
        Returns:
            Dictionary mapping (frequency, orientation) to filter response
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Normalize image
        gray = gray.astype(np.float32) / 255.0
        
        responses = {}
        
        for freq in self.frequencies:
            for orientation in self.orientations:
                # Convert orientation to radians
                theta = np.radians(orientation)
                
                # Apply Gabor filter
                real_response, _ = gabor(gray, frequency=freq, theta=theta,
                                       sigma_x=self.sigma_x, sigma_y=self.sigma_y)
                
                responses[(freq, orientation)] = real_response
        
        self.responses = responses
        return responses
    
    def detect_dashed_patterns(self, 
                              threshold: float = 0.1,
                              min_response_area: int = 100) -> List[Dict]:
        """
        Detect dashed patterns from Gabor responses.
        
        Args:
            threshold: Minimum response threshold
            min_response_area: Minimum area of response region
            
        Returns:
            List of detected pattern dictionaries
        """
        detected_patterns = []
        
        for (freq, orientation), response in self.responses.items():
            # Threshold response
            binary_response = response > threshold
            
            # Find connected components
            labeled_regions = morphology.label(binary_response)
            regions = measure.regionprops(labeled_regions)
            
            for region in regions:
                if region.area >= min_response_area:
                    # Check if region shows dash-like characteristics
                    bbox = region.bbox
                    region_response = response[bbox[0]:bbox[2], bbox[1]:bbox[3]]
                    
                    # Analyze pattern within region
                    if self._is_dashed_pattern(region_response, orientation):
                        pattern_dict = {
                            'centroid': region.centroid,
                            'orientation': orientation,
                            'frequency': freq,
                            'area': region.area,
                            'bbox': bbox,
                            'response_strength': np.mean(region_response),
                            'eccentricity': region.eccentricity,
                            'major_axis_length': region.major_axis_length,
                            'minor_axis_length': region.minor_axis_length
                        }
                        detected_patterns.append(pattern_dict)
        
        return detected_patterns
    
    def _is_dashed_pattern(self, response: np.ndarray, orientation: float) -> bool:
        """
        Check if a response region represents a dashed pattern.
        
        This analyzes the spatial distribution of responses to determine
        if they form a dash-like (periodic) pattern.
        """
        if response.size < 10:
            return False
        
        # Project response along the perpendicular direction to the orientation
        if 45 <= orientation < 135:  # More vertical
            projection = np.mean(response, axis=1)  # Project horizontally
        else:  # More horizontal
            projection = np.mean(response, axis=0)  # Project vertically
        
        if len(projection) < 5:
            return False
        
        # Look for periodicity in the projection
        # Simple method: count zero-crossings and peaks
        projection_smooth = ndimage.gaussian_filter1d(projection, sigma=1.0)
        
        # Find peaks
        peaks, _ = signal.find_peaks(projection_smooth, height=np.mean(projection_smooth))
        
        # A dashed pattern should have multiple peaks with some regularity
        if len(peaks) >= 2:
            # Check spacing regularity
            if len(peaks) > 2:
                spacings = np.diff(peaks)
                spacing_cv = np.std(spacings) / np.mean(spacings) if np.mean(spacings) > 0 else float('inf')
                
                # Regular spacing indicates dashed pattern
                if spacing_cv < 0.5:  # Coefficient of variation < 50%
                    return True
        
        return False


class FrequencyAnalyzer:
    """Frequency-domain analysis for dashed line detection."""
    
    def __init__(self):
        self.min_period = 5  # Minimum dash period in pixels
        self.max_period = 100  # Maximum dash period in pixels
        
    def analyze_line_frequency(self, 
                              image: np.ndarray,
                              line_coords: Tuple[int, int, int, int],
                              width: int = 5) -> Dict[str, float]:
        """
        Analyze frequency characteristics along a line.
        
        Args:
            image: Input image
            line_coords: Line coordinates (x1, y1, x2, y2)
            width: Sampling width perpendicular to line
            
        Returns:
            Dictionary with frequency analysis results
        """
        x1, y1, x2, y2 = line_coords
        
        # Sample intensities along line with some width
        intensities = self._sample_line_with_width(image, x1, y1, x2, y2, width)
        
        if len(intensities) < self.min_period * 2:
            return {'periodic_score': 0.0, 'dominant_period': 0.0, 'snr': 0.0}
        
        # Frequency analysis
        periodic_score, dominant_period, snr = self._frequency_analysis(intensities)
        
        return {
            'periodic_score': periodic_score,
            'dominant_period': dominant_period,
            'snr': snr,
            'line_length': np.sqrt((x2 - x1)**2 + (y2 - y1)**2),
            'num_cycles': len(intensities) / dominant_period if dominant_period > 0 else 0
        }
    
    def _sample_line_with_width(self, 
                               image: np.ndarray,
                               x1: int, y1: int, x2: int, y2: int,
                               width: int) -> np.ndarray:
        """Sample intensities along line with perpendicular width."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Line parameters
        length = int(np.sqrt((x2 - x1)**2 + (y2 - y1)**2))
        if length == 0:
            return np.array([])
        
        # Direction vectors
        dx = (x2 - x1) / length
        dy = (y2 - y1) / length
        
        # Perpendicular direction
        perp_dx = -dy
        perp_dy = dx
        
        # Sample points along line
        num_samples = max(length, 20)
        t = np.linspace(0, 1, num_samples)
        
        line_intensities = []
        
        for ti in t:
            # Point on line
            px = x1 + ti * (x2 - x1)
            py = y1 + ti * (y2 - y1)
            
            # Sample across width
            width_samples = []
            for w in range(-width//2, width//2 + 1):
                sample_x = int(px + w * perp_dx)
                sample_y = int(py + w * perp_dy)
                
                # Check bounds
                if (0 <= sample_x < gray.shape[1] and 
                    0 <= sample_y < gray.shape[0]):
                    width_samples.append(gray[sample_y, sample_x])
            
            if width_samples:
                # Average across width
                line_intensities.append(np.mean(width_samples))
        
        return np.array(line_intensities)
    
    def _frequency_analysis(self, intensities: np.ndarray) -> Tuple[float, float, float]:
        """Perform frequency analysis on intensity signal."""
        if len(intensities) < 10:
            return 0.0, 0.0, 0.0
        
        # Normalize and window
        intensities = intensities - np.mean(intensities)
        if np.std(intensities) > 0:
            intensities = intensities / np.std(intensities)
        
        # Apply window
        windowed = intensities * np.hanning(len(intensities))
        
        # FFT
        fft_result = np.fft.fft(windowed)
        magnitude = np.abs(fft_result)
        freqs = np.fft.fftfreq(len(intensities))
        
        # Focus on positive frequencies
        positive_freqs = freqs[1:len(freqs)//2]
        positive_magnitude = magnitude[1:len(magnitude)//2]
        
        if len(positive_magnitude) == 0:
            return 0.0, 0.0, 0.0
        
        # Convert frequencies to periods and filter by valid range
        periods = 1.0 / np.abs(positive_freqs)
        valid_mask = (periods >= self.min_period) & (periods <= self.max_period)
        
        if not np.any(valid_mask):
            return 0.0, 0.0, 0.0
        
        valid_periods = periods[valid_mask]
        valid_magnitudes = positive_magnitude[valid_mask]
        
        # Find dominant period
        peak_idx = np.argmax(valid_magnitudes)
        dominant_period = valid_periods[peak_idx]
        peak_magnitude = valid_magnitudes[peak_idx]
        
        # Calculate SNR
        noise_level = np.median(valid_magnitudes)
        snr = peak_magnitude / (noise_level + 1e-8)
        
        # Periodic score based on SNR and period consistency
        periodic_score = min(snr / 5.0, 1.0)  # Normalize to 0-1
        
        # Bonus for multiple harmonics
        harmonic_bonus = self._check_harmonics(valid_periods, valid_magnitudes, dominant_period)
        periodic_score *= (1.0 + harmonic_bonus * 0.5)
        periodic_score = min(periodic_score, 1.0)
        
        return periodic_score, dominant_period, snr
    
    def _check_harmonics(self, 
                        periods: np.ndarray, 
                        magnitudes: np.ndarray,
                        fundamental_period: float) -> float:
        """Check for harmonic peaks that support periodicity."""
        harmonic_score = 0.0
        
        # Check for harmonics (half period, third period, etc.)
        for harmonic in [2, 3, 4]:
            expected_period = fundamental_period / harmonic
            
            # Find closest period
            period_diffs = np.abs(periods - expected_period)
            closest_idx = np.argmin(period_diffs)
            
            if period_diffs[closest_idx] < fundamental_period * 0.2:  # Within 20%
                # Relative strength of harmonic
                relative_strength = magnitudes[closest_idx] / np.max(magnitudes)
                harmonic_score += relative_strength / harmonic  # Weight by harmonic order
        
        return min(harmonic_score, 1.0)


class RobustDashDetector:
    """Combines multiple robust detection methods."""
    
    def __init__(self):
        self.lsd = LineSegmentDetector()
        self.gabor_bank = GaborFilterBank()
        self.freq_analyzer = FrequencyAnalyzer()
        self.detection_results = {}
        
    def detect_all_methods(self, 
                          image: np.ndarray,
                          expected_period: Optional[float] = None) -> Dict[str, List[Dict]]:
        """
        Apply all robust detection methods.
        
        Args:
            image: Input image
            expected_period: Expected dash period for filtering
            
        Returns:
            Dictionary with results from each method
        """
        results = {}
        
        # Method 1: LSD with periodicity filtering
        print("Applying Line Segment Detector...")
        line_segments = self.lsd.detect_line_segments(image)
        lsd_dashed = self.lsd.filter_by_periodicity(image, line_segments, expected_period)
        results['lsd'] = lsd_dashed
        
        # Method 2: Gabor filter bank
        print("Applying Gabor filter bank...")
        gabor_responses = self.gabor_bank.apply_filters(image)
        gabor_patterns = self.gabor_bank.detect_dashed_patterns()
        results['gabor'] = gabor_patterns
        
        # Method 3: Frequency analysis on detected lines
        print("Analyzing frequency characteristics...")
        freq_results = []
        for line in lsd_dashed:
            start = line['start']
            end = line['end']
            line_coords = (start[0], start[1], end[0], end[1])
            freq_analysis = self.freq_analyzer.analyze_line_frequency(image, line_coords)
            
            if freq_analysis['periodic_score'] > 0.3:
                line_with_freq = line.copy()
                line_with_freq.update(freq_analysis)
                freq_results.append(line_with_freq)
        
        results['frequency'] = freq_results
        
        self.detection_results = results
        return results
    
    def combine_detections(self, 
                          detection_results: Dict[str, List[Dict]],
                          consensus_threshold: int = 2) -> List[Dict]:
        """
        Combine detections from multiple methods using consensus.
        
        Args:
            detection_results: Results from different detection methods
            consensus_threshold: Minimum number of methods that must agree
            
        Returns:
            List of combined detection results
        """
        all_detections = []
        
        # Collect all detections with method labels
        for method, detections in detection_results.items():
            for detection in detections:
                detection['detection_method'] = method
                all_detections.append(detection)
        
        if not all_detections:
            return []
        
        # Group detections by spatial proximity
        grouped_detections = self._group_by_proximity(all_detections)
        
        # Filter groups by consensus
        consensus_detections = []
        
        for group in grouped_detections:
            if len(group) >= consensus_threshold:
                # Merge group into single detection
                merged = self._merge_detection_group(group)
                consensus_detections.append(merged)
        
        return consensus_detections
    
    def _group_by_proximity(self, 
                           detections: List[Dict],
                           distance_threshold: float = 20.0) -> List[List[Dict]]:
        """Group detections by spatial proximity."""
        if not detections:
            return []
        
        # Extract centroids for clustering
        centroids = []
        for detection in detections:
            if 'centroid' in detection:
                centroids.append(detection['centroid'])
            elif 'start' in detection and 'end' in detection:
                start = detection['start']
                end = detection['end']
                centroid = ((start[0] + end[0]) / 2, (start[1] + end[1]) / 2)
                centroids.append(centroid)
            else:
                centroids.append((0, 0))  # Fallback
        
        centroids = np.array(centroids)
        
        if len(centroids) < 2:
            return [detections]
        
        # DBSCAN clustering
        clustering = DBSCAN(eps=distance_threshold, min_samples=1)
        cluster_labels = clustering.fit_predict(centroids)
        
        # Group by cluster labels
        groups = {}
        for i, label in enumerate(cluster_labels):
            if label not in groups:
                groups[label] = []
            groups[label].append(detections[i])
        
        return list(groups.values())
    
    def _merge_detection_group(self, group: List[Dict]) -> Dict:
        """Merge a group of detections into a single detection."""
        if len(group) == 1:
            return group[0]
        
        # Calculate consensus properties
        methods = [det.get('detection_method', 'unknown') for det in group]
        method_counts = {method: methods.count(method) for method in set(methods)}
        
        # Start with the detection from the most confident method
        base_detection = max(group, key=lambda x: x.get('confidence', x.get('period_score', x.get('periodic_score', 0))))
        
        merged = base_detection.copy()
        
        # Update with consensus information
        merged['detection_methods'] = list(method_counts.keys())
        merged['method_consensus'] = len(set(methods))
        merged['total_detections'] = len(group)
        
        # Average numeric properties
        numeric_props = ['confidence', 'period_score', 'periodic_score', 'response_strength']
        for prop in numeric_props:
            values = [det.get(prop, 0) for det in group if prop in det]
            if values:
                merged[f'avg_{prop}'] = np.mean(values)
                merged[f'std_{prop}'] = np.std(values)
        
        # Combine spatial information
        if 'start' in merged and 'end' in merged:
            # Keep the longest line from the group
            longest_line = max(group, key=lambda x: x.get('length', 0))
            merged['start'] = longest_line['start']
            merged['end'] = longest_line['end']
            merged['length'] = longest_line['length']
        
        return merged


def main():
    """Example usage of robust detection methods."""
    print("Robust Dashed Line Detection System")
    print("===================================")
    
    # Create example test image
    test_image = np.ones((400, 600), dtype=np.uint8) * 255
    
    # Add some dashed lines
    for i in range(5):
        y = 50 + i * 60
        for x in range(50, 550, 20):
            cv2.rectangle(test_image, (x, y), (x + 10, y + 3), 0, -1)
    
    # Test robust detector
    detector = RobustDashDetector()
    results = detector.detect_all_methods(test_image, expected_period=20)
    
    print(f"Detection Results:")
    for method, detections in results.items():
        print(f"  {method}: {len(detections)} detections")
    
    # Combine results
    combined = detector.combine_detections(results, consensus_threshold=2)
    print(f"Combined (consensus): {len(combined)} detections")


if __name__ == "__main__":
    main()
