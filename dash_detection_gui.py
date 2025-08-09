"""
Streamlit GUI for Dashed Line Detection System

This provides a user-friendly web interface for the integrated dashed line detection system.
"""

import streamlit as st
import cv2
import numpy as np
import os
import tempfile
from PIL import Image
import pandas as pd
import json
from typing import Optional, Tuple
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Import our integrated system
from integrated_dash_detector import IntegratedDashDetector


class DashDetectionGUI:
    """Streamlit GUI for dashed line detection."""
    
    def __init__(self):
        self.detector = IntegratedDashDetector()
        self.temp_dir = tempfile.mkdtemp()
        
        # Session state initialization
        if 'image_loaded' not in st.session_state:
            st.session_state.image_loaded = False
        if 'template_learned' not in st.session_state:
            st.session_state.template_learned = False
        if 'detection_results' not in st.session_state:
            st.session_state.detection_results = []
        if 'roi_coordinates' not in st.session_state:
            st.session_state.roi_coordinates = None
    
    def run(self):
        """Run the Streamlit application."""
        st.set_page_config(
            page_title="Dashed Line Detection",
            page_icon="📐",
            layout="wide"
        )
        
        st.title("🏗️ Floor Plan Dashed Line Detection System")
        st.markdown("---")
        
        # Sidebar for configuration
        self.setup_sidebar()
        
        # Main content
        col1, col2 = st.columns([2, 1])
        
        with col1:
            self.image_upload_section()
            if st.session_state.image_loaded:
                self.template_learning_section()
                self.scale_setting_section()
                self.detection_section()
        
        with col2:
            self.results_section()
    
    def setup_sidebar(self):
        """Setup the configuration sidebar."""
        st.sidebar.header("⚙️ Configuration")
        
        # Preprocessing options
        st.sidebar.subheader("Preprocessing")
        self.detector.config['preprocessing']['apply_deskew'] = st.sidebar.checkbox(
            "Apply Deskewing", 
            value=self.detector.config['preprocessing']['apply_deskew']
        )
        self.detector.config['preprocessing']['apply_clahe'] = st.sidebar.checkbox(
            "Apply CLAHE Enhancement", 
            value=self.detector.config['preprocessing']['apply_clahe']
        )
        self.detector.config['preprocessing']['remove_text'] = st.sidebar.checkbox(
            "Remove Text Regions", 
            value=self.detector.config['preprocessing']['remove_text']
        )
        
        # Detection options
        st.sidebar.subheader("Detection Methods")
        self.detector.config['detection']['use_classic'] = st.sidebar.checkbox(
            "Template-based Detection", 
            value=self.detector.config['detection']['use_classic']
        )
        self.detector.config['detection']['use_robust'] = st.sidebar.checkbox(
            "Robust Detection (LSD + Gabor)", 
            value=self.detector.config['detection']['use_robust']
        )
        self.detector.config['detection']['combine_methods'] = st.sidebar.checkbox(
            "Combine Detection Methods", 
            value=self.detector.config['detection']['combine_methods']
        )
        
        if self.detector.config['detection']['combine_methods']:
            self.detector.config['detection']['consensus_threshold'] = st.sidebar.slider(
                "Consensus Threshold", 1, 4, 
                self.detector.config['detection']['consensus_threshold']
            )
        
        # Classification options
        st.sidebar.subheader("Classification")
        self.detector.config['classification']['use_legend'] = st.sidebar.checkbox(
            "Use Legend-based Classification", 
            value=self.detector.config['classification']['use_legend']
        )
        self.detector.config['classification']['min_confidence'] = st.sidebar.slider(
            "Minimum Confidence", 0.0, 1.0, 
            self.detector.config['classification']['min_confidence']
        )
        
        # Output options
        st.sidebar.subheader("Export Options")
        self.detector.config['output']['export_svg'] = st.sidebar.checkbox(
            "Export SVG", value=self.detector.config['output']['export_svg']
        )
        self.detector.config['output']['export_geojson'] = st.sidebar.checkbox(
            "Export GeoJSON", value=self.detector.config['output']['export_geojson']
        )
        self.detector.config['output']['export_dxf'] = st.sidebar.checkbox(
            "Export DXF", value=self.detector.config['output']['export_dxf']
        )
        self.detector.config['output']['export_csv'] = st.sidebar.checkbox(
            "Export CSV", value=self.detector.config['output']['export_csv']
        )
        self.detector.config['output']['export_overlay'] = st.sidebar.checkbox(
            "Export Overlay Image", value=self.detector.config['output']['export_overlay']
        )
    
    def image_upload_section(self):
        """Handle image upload and display."""
        st.header("📁 Image Upload")
        
        uploaded_file = st.file_uploader(
            "Choose a floor plan image",
            type=['png', 'jpg', 'jpeg', 'tiff', 'bmp'],
            help="Upload a floor plan image in common formats"
        )
        
        if uploaded_file is not None:
            # Save uploaded file to temp directory
            temp_path = os.path.join(self.temp_dir, uploaded_file.name)
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Load image
            if self.detector.load_image(temp_path):
                st.session_state.image_loaded = True
                
                # Display image
                st.subheader("📷 Loaded Image")
                
                # Convert to RGB for display
                display_image = cv2.cvtColor(self.detector.original_image, cv2.COLOR_BGR2RGB)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.image(display_image, caption="Original Image", use_column_width=True)
                
                with col2:
                    if self.detector.processed_image is not None:
                        st.image(self.detector.processed_image, caption="Processed Image", 
                               use_column_width=True, clamp=True)
                
                # Image information
                h, w = self.detector.original_image.shape[:2]
                st.info(f"Image Size: {w} × {h} pixels")
            else:
                st.error("Failed to load image. Please check the file format.")
                st.session_state.image_loaded = False
    
    def template_learning_section(self):
        """Handle template learning from user ROI selection."""
        st.header("🎯 Template Learning")
        
        if not st.session_state.image_loaded:
            st.warning("Please upload an image first.")
            return
        
        st.markdown("""
        **Instructions:**
        1. Enter coordinates for a region containing a clean dash + gap sample
        2. The system will learn the dash pattern from this region
        3. This template will be used for classic template-based detection
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            roi_x = st.number_input("ROI X (left)", min_value=0, 
                                   max_value=self.detector.original_image.shape[1]-1, value=100)
            roi_y = st.number_input("ROI Y (top)", min_value=0, 
                                   max_value=self.detector.original_image.shape[0]-1, value=100)
        
        with col2:
            roi_width = st.number_input("ROI Width", min_value=10, 
                                      max_value=self.detector.original_image.shape[1], value=100)
            roi_height = st.number_input("ROI Height", min_value=10, 
                                       max_value=self.detector.original_image.shape[0], value=50)
        
        # Show ROI preview
        if st.button("Preview ROI"):
            self._show_roi_preview(roi_x, roi_y, roi_width, roi_height)
        
        # Learn template button
        if st.button("🔍 Learn Template from ROI", type="primary"):
            if self.detector.learn_template_from_roi(roi_x, roi_y, roi_width, roi_height):
                st.success("✅ Template learned successfully!")
                st.session_state.template_learned = True
                st.session_state.roi_coordinates = (roi_x, roi_y, roi_width, roi_height)
                
                # Show template info
                template = self.detector.classic_detector.dash_template
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Dash Length", f"{template.dash_length} px")
                with col2:
                    st.metric("Gap Length", f"{template.gap_length} px")
                with col3:
                    st.metric("Total Period", f"{template.total_period} px")
            else:
                st.error("❌ Failed to learn template. Please check ROI coordinates.")
    
    def scale_setting_section(self):
        """Handle scale detection and manual setting."""
        st.header("📏 Scale Setting")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Automatic Scale Detection")
            if st.button("🔍 Auto-Detect Scale"):
                if self.detector.detect_scale():
                    scale_info = self.detector.scale_detector.get_scale_info()
                    st.success(f"✅ Scale detected: {scale_info['scale_px_to_m']:.6f} px/m")
                    st.info(f"Detection method: {scale_info['scale_type']}")
                else:
                    st.warning("⚠️ Could not auto-detect scale. Consider manual setting.")
        
        with col2:
            st.subheader("Manual Scale Setting")
            
            # Manual scale input
            point1_x = st.number_input("Point 1 X", value=100, key="p1x")
            point1_y = st.number_input("Point 1 Y", value=100, key="p1y")
            point2_x = st.number_input("Point 2 X", value=200, key="p2x")
            point2_y = st.number_input("Point 2 Y", value=100, key="p2y")
            known_distance = st.number_input("Known Distance (meters)", value=1.0, min_value=0.01)
            
            if st.button("📐 Set Manual Scale"):
                if self.detector.set_manual_scale(
                    (point1_x, point1_y), (point2_x, point2_y), known_distance
                ):
                    st.success("✅ Manual scale set successfully!")
                else:
                    st.error("❌ Failed to set manual scale.")
        
        # Show current scale
        scale_info = self.detector.scale_detector.get_scale_info()
        if scale_info['scale_px_to_m'] != 1.0:
            st.info(f"Current scale: {scale_info['scale_px_to_m']:.6f} px/m")
    
    def detection_section(self):
        """Handle the detection process."""
        st.header("🔎 Detection")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Individual detection steps
            st.subheader("Step-by-Step Detection")
            
            if st.button("🔍 Detect Legend"):
                if self.detector.detect_legend():
                    st.success(f"✅ Legend detected with {len(self.detector.legend_entries)} entries")
                    self._show_legend_entries()
                else:
                    st.warning("⚠️ No legend detected")
            
            if st.button("🎯 Run Classic Detection"):
                if st.session_state.template_learned:
                    if self.detector.detect_classic():
                        st.success(f"✅ Classic detection: {len(self.detector.classic_results)} lines")
                    else:
                        st.error("❌ Classic detection failed")
                else:
                    st.warning("⚠️ Please learn template first")
            
            if st.button("🔬 Run Robust Detection"):
                if self.detector.detect_robust():
                    total = sum(len(results) for results in self.detector.robust_results.values())
                    st.success(f"✅ Robust detection: {total} total detections")
                    self._show_robust_results()
                else:
                    st.error("❌ Robust detection failed")
        
        with col2:
            # Complete detection
            st.subheader("Complete Detection Pipeline")
            
            if st.button("🚀 Run Complete Detection", type="primary"):
                with st.spinner("Running complete detection pipeline..."):
                    results = self.detector.detect_all()
                    st.session_state.detection_results = results
                
                if results:
                    st.success(f"✅ Detection complete! Found {len(results)} dashed lines")
                    self._show_detection_summary()
                else:
                    st.warning("⚠️ No dashed lines detected")
            
            # Export results
            if st.session_state.detection_results:
                st.subheader("📤 Export Results")
                
                export_dir = st.text_input("Export Directory", value="output")
                
                if st.button("💾 Export All Formats"):
                    with st.spinner("Exporting results..."):
                        output_files = self.detector.export_results(export_dir)
                    
                    if output_files:
                        st.success("✅ Export complete!")
                        self._show_export_files(output_files)
                    else:
                        st.error("❌ Export failed")
    
    def results_section(self):
        """Display detection results and statistics."""
        st.header("📊 Results")
        
        if not st.session_state.detection_results:
            st.info("Run detection to see results here.")
            return
        
        results = st.session_state.detection_results
        
        # Summary statistics
        st.subheader("📈 Summary Statistics")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Lines", len(results))
        with col2:
            total_length = sum(line.get('length_m', 0) for line in results)
            st.metric("Total Length", f"{total_length:.2f} m")
        with col3:
            avg_confidence = np.mean([line.get('confidence', 0) for line in results if 'confidence' in line])
            st.metric("Avg Confidence", f"{avg_confidence:.3f}")
        
        # Line type distribution
        st.subheader("📊 Line Type Distribution")
        line_types = {}
        for line in results:
            line_type = line.get('classified_type', line.get('class', 'unknown'))
            line_types[line_type] = line_types.get(line_type, 0) + 1
        
        if line_types:
            df = pd.DataFrame(list(line_types.items()), columns=['Line Type', 'Count'])
            st.bar_chart(df.set_index('Line Type'))
        
        # Detailed results table
        st.subheader("📋 Detailed Results")
        
        if st.checkbox("Show detailed table"):
            df_data = []
            for i, line in enumerate(results):
                df_data.append({
                    'ID': line.get('id', f'line_{i}'),
                    'Type': line.get('classified_type', line.get('class', 'unknown')),
                    'Length (m)': f"{line.get('length_m', 0):.3f}",
                    'Length (px)': f"{line.get('length_px', 0):.1f}",
                    'Angle (°)': f"{line.get('angle_deg', 0):.1f}",
                    'Confidence': f"{line.get('confidence', 0):.3f}",
                    'Method': line.get('detection_method', 'unknown')
                })
            
            df = pd.DataFrame(df_data)
            st.dataframe(df, use_container_width=True)
        
        # Visualization
        st.subheader("🖼️ Visualization")
        if st.button("Show Detection Overlay"):
            self._show_detection_overlay()
    
    def _show_roi_preview(self, x: int, y: int, width: int, height: int):
        """Show preview of selected ROI."""
        if self.detector.original_image is None:
            return
        
        # Create image with ROI rectangle
        display_image = cv2.cvtColor(self.detector.original_image, cv2.COLOR_BGR2RGB)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.imshow(display_image)
        
        # Add ROI rectangle
        rect = patches.Rectangle((x, y), width, height, linewidth=2, 
                               edgecolor='red', facecolor='none')
        ax.add_patch(rect)
        
        ax.set_title("ROI Preview")
        ax.set_xlim(max(0, x-50), min(display_image.shape[1], x+width+50))
        ax.set_ylim(min(display_image.shape[0], y+height+50), max(0, y-50))
        
        st.pyplot(fig)
        plt.close()
        
        # Show ROI crop
        roi_crop = self.detector.original_image[y:y+height, x:x+width]
        if roi_crop.size > 0:
            roi_rgb = cv2.cvtColor(roi_crop, cv2.COLOR_BGR2RGB)
            st.image(roi_rgb, caption="ROI Crop", width=300)
    
    def _show_legend_entries(self):
        """Display detected legend entries."""
        if not self.detector.legend_entries:
            return
        
        st.subheader("📖 Legend Entries")
        for i, entry in enumerate(self.detector.legend_entries):
            with st.expander(f"Entry {i+1}: {entry.get('text', 'N/A')}"):
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Type:** {entry.get('line_type', 'unknown')}")
                    st.write(f"**Confidence:** {entry.get('confidence', 0):.3f}")
                    st.write(f"**Method:** {entry.get('extraction_method', 'N/A')}")
                with col2:
                    if 'keywords' in entry:
                        st.write(f"**Keywords:** {', '.join(entry['keywords'])}")
                    if 'bbox' in entry:
                        st.write(f"**BBox:** {entry['bbox']}")
    
    def _show_robust_results(self):
        """Display robust detection results."""
        if not self.detector.robust_results:
            return
        
        st.subheader("🔬 Robust Detection Results")
        for method, results in self.detector.robust_results.items():
            st.write(f"**{method.upper()}:** {len(results)} detections")
    
    def _show_detection_summary(self):
        """Display detection summary."""
        stats = self.detector.get_detection_statistics()
        
        with st.expander("📊 Detection Statistics"):
            st.json(stats)
    
    def _show_export_files(self, output_files: dict):
        """Display exported files."""
        st.subheader("📁 Exported Files")
        for format_name, file_path in output_files.items():
            st.write(f"**{format_name.upper()}:** `{file_path}`")
    
    def _show_detection_overlay(self):
        """Show detection results overlaid on original image."""
        if (self.detector.original_image is None or 
            not st.session_state.detection_results):
            return
        
        # Create overlay
        overlay = self.detector.original_image.copy()
        
        # Colors for different line types
        colors = {
            'dashed_service_line': (0, 0, 255),     # Red
            'dashed_utility_line': (255, 0, 0),    # Blue
            'dashed_boundary': (0, 255, 0),        # Green
            'unknown': (255, 0, 255),              # Magenta
            'default': (0, 255, 255)               # Yellow
        }
        
        for line in st.session_state.detection_results:
            points = line.get('points', [])
            if len(points) < 2:
                continue
            
            line_type = line.get('classified_type', line.get('class', 'unknown'))
            color = colors.get(line_type, colors['default'])
            
            # Draw line
            pts = np.array(points, dtype=np.int32)
            cv2.polylines(overlay, [pts], False, color, 2)
        
        # Convert to RGB and display
        overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
        st.image(overlay_rgb, caption="Detection Results Overlay", use_column_width=True)


def main():
    """Run the Streamlit GUI."""
    gui = DashDetectionGUI()
    gui.run()


if __name__ == "__main__":
    main()
