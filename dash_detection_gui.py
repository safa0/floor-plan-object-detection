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
import math
from typing import Optional, Tuple, List, Dict, Any
from streamlit_drawable_canvas import st_canvas
from skimage.morphology import skeletonize
from scipy.signal import find_peaks

# Notebook-style pipeline only; no integrated system


class DashDetectionGUI:
    """Streamlit GUI for dashed line detection."""
    
    def __init__(self):
        # Persist temp dir and image across reruns
        if 'temp_dir' not in st.session_state:
            st.session_state.temp_dir = tempfile.mkdtemp()
        self.temp_dir = st.session_state.temp_dir

        # Session state initialization
        if 'original_image' not in st.session_state:
            st.session_state.original_image = None
        if 'current_image_path' not in st.session_state:
            st.session_state.current_image_path = None
        if 'image_loaded' not in st.session_state:
            st.session_state.image_loaded = False
        if 'template_learned' not in st.session_state:
            st.session_state.template_learned = False
        if 'detection_results' not in st.session_state:
            st.session_state.detection_results = []
        if 'roi_coordinates' not in st.session_state:
            st.session_state.roi_coordinates = None
        if 'sample_params' not in st.session_state:
            st.session_state.sample_params = None
        if 'notebook_debug' not in st.session_state:
            st.session_state.notebook_debug = {}

        # Local refs
        self.original_image = st.session_state.original_image
    
    def run(self):
        """Run the Streamlit application."""
        st.set_page_config(
            page_title="Dashed Line Detection",
            page_icon="📐",
            layout="wide"
        )
        
        st.title("🏗️ Floor Plan Dashed Line Detection System")
        st.markdown("---")
        
        # Main content
        col1, col2 = st.columns([2, 1])
        
        with col1:
            self.image_upload_section()
            if st.session_state.image_loaded:
                self.template_learning_section()
                self.detection_section()
        
        with col2:
            self.results_section()
    
    def setup_sidebar(self):
        return  # No sidebar needed for notebook-only pipeline
    
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
            
            # Load image only if new or not yet loaded to avoid resetting state on reruns
            should_load = (not st.session_state.image_loaded) or (st.session_state.current_image_path != temp_path)
            if should_load:
                img = cv2.imread(temp_path)
                if img is None:
                    st.error("Failed to load image. Please check the file format.")
                    st.session_state.image_loaded = False
                else:
                    st.session_state.original_image = img
                    st.session_state.image_loaded = True
                    st.session_state.current_image_path = temp_path
                    self.original_image = img
            
            if st.session_state.image_loaded and st.session_state.original_image is not None:
                # Display image
                st.subheader("📷 Loaded Image")
                
                # Convert to RGB for display
                display_image = cv2.cvtColor(st.session_state.original_image, cv2.COLOR_BGR2RGB)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.image(display_image, caption="Original Image", use_column_width=True)
                
                # Image information
                h, w = st.session_state.original_image.shape[:2]
                st.info(f"Image Size: {w} × {h} pixels")
    
    def template_learning_section(self):
        """Handle sample learning from interactive ROI selection (notebook pipeline)."""
        st.header("🎯 Sample Learning (Notebook Pipeline)")
        
        if not st.session_state.image_loaded:
            st.warning("Please upload an image first.")
            return
        
        # Show learned sample params if available
        if st.session_state.sample_params:
            sp = st.session_state.sample_params
            st.info(
                f"📏 Sample params — stroke≈{sp.get('stroke_width', float('nan')):.1f}px, "
                f"dash≈{sp.get('dash_len', float('nan')):.1f}px, gap≈{sp.get('gap_len', float('nan')):.1f}px, "
                f"period≈{sp.get('period_px', float('nan')):.1f}px"
            )
        
        st.markdown("""
        **Instructions:**
        1. Use the drawing tools below to select a region containing a clean dash + gap sample
        2. Draw a rectangle around 1-2 complete dash cycles
        3. Avoid areas with overlapping lines or text
        4. Click "Learn Sample from Selected ROI" to extract pattern parameters
        """)
        
        self._interactive_roi_selection()
    
    def detection_section(self):
        """Notebook-style detection pipeline (sample-driven)."""
        st.header("🔎 Detection (Notebook Pipeline)")

        if not st.session_state.image_loaded:
            st.warning("Please upload an image first.")
            return

        if not st.session_state.sample_params:
            st.warning("Please learn a sample from ROI above first.")
            return

        if st.button("🚀 Run Detection (Notebook)", type="primary"):
            with st.spinner("Running sample-driven detection..."):
                results, debug = self._run_notebook_pipeline()
                st.session_state.detection_results = results
                st.session_state.notebook_debug = debug

        # Show debug visuals if available
        debug = st.session_state.get('notebook_debug', {})
        if debug:
            st.subheader("Debug Visuals")
            colA, colB = st.columns(2)
            with colA:
                if 'img_eq' in debug:
                    st.image(debug['img_eq'], caption="Equalized (CLAHE)", use_column_width=True, clamp=True)
                if 'mask_width' in debug:
                    st.image(debug['mask_width']*255, caption="Width Mask", use_column_width=True, clamp=True)
                if 'skeleton' in debug:
                    st.image(debug['skeleton']*255, caption="Skeleton", use_column_width=True, clamp=True)
            with colB:
                if 'img_bin' in debug:
                    st.image(debug['img_bin'], caption="Binarized (Otsu, ink=white)", use_column_width=True, clamp=True)
                if 'resp_max' in debug:
                    st.image(debug['resp_max_norm'], caption="Matched-filter response (max over angles)", use_column_width=True, clamp=True)
                if 'match_mask' in debug:
                    st.image(debug['match_mask']*255, caption="Thresholded response", use_column_width=True, clamp=True)

        # Overlay
        if st.session_state.detection_results:
            st.subheader("🖼️ Overlay")
            overlay = self._make_overlay_from_results(st.session_state.detection_results)
            st.image(overlay, caption="Detected dashed-line polylines (red)", use_column_width=True)
    
    def results_section(self):
        """Display detection results and statistics."""
        st.header("📊 Results")
        
        if not st.session_state.detection_results:
            st.info("Run detection to see results here.")
            return
        
        results = st.session_state.detection_results
        
        # Summary statistics
        st.subheader("📈 Summary Statistics")
        
        total_length = sum(line.get('length_m', 0) for line in results)
        avg_confidence = np.mean([line.get('confidence', 0) for line in results if 'confidence' in line])
        
        st.write(f"**Total Lines:** {len(results)}")
        st.write(f"**Total Length:** {total_length:.2f} m")
        st.write(f"**Average Confidence:** {avg_confidence:.3f}")
        
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
    
    def _interactive_roi_selection(self):
        """Interactive ROI selection using drawable canvas."""
        st.subheader("🖱️ Interactive ROI Selection")
        
        # Convert image to RGB for display
        display_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(display_image)
        
        # Calculate display size (limit to reasonable size for web interface)
        h, w = display_image.shape[:2]
        max_width = 800
        if w > max_width:
            scale_factor = max_width / w
            new_width = max_width
            new_height = int(h * scale_factor)
        else:
            scale_factor = 1.0
            new_width = w
            new_height = h
        
        # Resize image for display
        img_pil_resized = img_pil.resize((new_width, new_height))
        
        # Drawing canvas
        canvas_result = st_canvas(
            fill_color="rgba(255, 0, 0, 0.1)",  # Semi-transparent red fill
            stroke_width=2,
            stroke_color="#FF0000",  # Red stroke
            background_image=img_pil_resized,
            update_streamlit=True,
            width=new_width,
            height=new_height,
            drawing_mode="rect",  # Rectangle drawing mode
            point_display_radius=0,
            key="roi_canvas",
        )
        
        # Process canvas data
        if canvas_result.json_data is not None:
            objects = canvas_result.json_data["objects"]
            
            if objects:
                # Get the last drawn rectangle
                last_rect = objects[-1]
                
                if last_rect["type"] == "rect":
                    # Extract rectangle coordinates (in display coordinates)
                    display_x = int(last_rect["left"])
                    display_y = int(last_rect["top"]) 
                    display_width = int(last_rect["width"])
                    display_height = int(last_rect["height"])
                    
                    # Convert back to original image coordinates
                    roi_x = int(display_x / scale_factor)
                    roi_y = int(display_y / scale_factor)
                    roi_width = int(display_width / scale_factor)
                    roi_height = int(display_height / scale_factor)
                    
                    # Ensure coordinates are within image bounds
                    roi_x = max(0, min(roi_x, w - 1))
                    roi_y = max(0, min(roi_y, h - 1))
                    roi_width = max(10, min(roi_width, w - roi_x))
                    roi_height = max(10, min(roi_height, h - roi_y))
                    
                    # Show selected ROI info
                    st.info(f"Selected ROI: ({roi_x}, {roi_y}) - {roi_width}×{roi_height} px")
                    
                    # Show ROI crop preview
                    roi_crop = self.original_image[roi_y:roi_y+roi_height, roi_x:roi_x+roi_width]
                    if roi_crop.size > 0:
                        roi_rgb = cv2.cvtColor(roi_crop, cv2.COLOR_BGR2RGB)
                        st.image(roi_rgb, caption="Selected ROI", width=200)
                    
                    # Learn sample params button (notebook pipeline)
                    if st.button("🔍 Learn Sample from Selected ROI", type="primary"):
                        params = self._learn_sample_from_roi(roi_x, roi_y, roi_width, roi_height)
                        if params:
                            st.success("✅ Sample learned successfully!")
                            st.session_state.template_learned = True
                            st.session_state.roi_coordinates = (roi_x, roi_y, roi_width, roi_height)
                            st.session_state.sample_params = params
                            st.write("**Sample Metrics:**")
                            st.write(f"• Stroke Width: {params.get('stroke_width', float('nan')):.2f} px")
                            st.write(f"• Dash Length: {params.get('dash_len', float('nan')):.2f} px")
                            st.write(f"• Gap Length: {params.get('gap_len', float('nan')):.2f} px")
                            st.write(f"• Period: {params.get('period_px', float('nan')):.2f} px")
                        else:
                            st.error("❌ Failed to learn from ROI. Please select a better ROI.")
            else:
                st.info("👆 Draw a rectangle around a clean dash pattern to select ROI")
        
        # Instructions
        st.markdown("""
        **How to use:**
        - 🖱️ Click and drag to draw a rectangle around a dash sample
        - 🎯 Select 1-2 complete dash cycles for best results  
        - ✨ The red rectangle shows your selection
        - 🔄 Draw a new rectangle to replace the previous selection
        """)

    def _learn_sample_from_roi(self, x: int, y: int, w: int, h: int) -> Optional[Dict[str, float]]:
        """Learn stroke width, dash length, gap and period from selected ROI (notebook logic)."""
        try:
            if self.original_image is None:
                return None
            gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
            roi = gray[y:y+h, x:x+w]
            if roi.size == 0 or roi.shape[0] < 10 or roi.shape[1] < 10:
                return None

            roi_blur = cv2.GaussianBlur(roi, (3, 3), 0)
            _, roi_bin = cv2.threshold(roi_blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

            # Stroke width via distance transform (ink=1)
            dist = cv2.distanceTransform(roi_bin, cv2.DIST_L2, 3)
            ink_mask = roi_bin > 0
            if np.count_nonzero(ink_mask) < 50:
                return None
            stroke_width = float(np.median(dist[ink_mask]) * 2.0)
            stroke_width = max(1.0, stroke_width)

            # Orientation via covariance of ink coordinates
            ys, xs = np.where(ink_mask)
            cov = np.cov(xs, ys)
            eigvals, eigvecs = np.linalg.eig(cov)
            main_axis = eigvecs[:, np.argmax(eigvals)]
            angle_rad = math.atan2(main_axis[1], main_axis[0])
            angle_deg = -np.degrees(angle_rad)  # negative for cv2 rotation

            # Rotate with padding to avoid cropping
            h_roi, w_roi = roi_bin.shape
            center = (w_roi // 2, h_roi // 2)
            M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
            cos = abs(M[0, 0]); sin = abs(M[0, 1])
            new_w = int((h_roi * sin) + (w_roi * cos))
            new_h = int((h_roi * cos) + (w_roi * sin))
            M[0, 2] += (new_w / 2) - center[0]
            M[1, 2] += (new_h / 2) - center[1]
            roi_rot = cv2.warpAffine(roi_bin, M, (new_w, new_h), flags=cv2.INTER_NEAREST, borderValue=0)

            # Projection along dash axis
            proj = np.sum(roi_rot > 0, axis=0).astype(np.float32)
            if proj.max() > proj.min():
                proj = (proj - proj.min()) / (proj.max() - proj.min() + 1e-6)
            else:
                proj = np.zeros_like(proj)

            peaks, _ = find_peaks(proj, height=0.3, distance=max(2, int(stroke_width)))
            period_px = float(np.median(np.diff(peaks))) if len(peaks) > 1 else float('nan')

            dash_widths: List[int] = []
            th = 0.3
            for p in peaks:
                left = int(p)
                while left > 0 and proj[left] > th:
                    left -= 1
                right = int(p)
                while right < len(proj) - 1 and proj[right] > th:
                    right += 1
                dash_widths.append(right - left)

            dash_len = float(np.median(dash_widths)) if len(dash_widths) > 0 else float('nan')
            gap_len = float(period_px - dash_len) if (not math.isnan(period_px) and not math.isnan(dash_len)) else float('nan')

            return {
                'stroke_width': stroke_width,
                'dash_len': dash_len,
                'gap_len': gap_len,
                'period_px': period_px,
                'roi_angle_deg': float(-angle_deg)  # store original orientation if needed
            }
        except Exception:
            return None

    def _run_notebook_pipeline(self) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Run the notebook's sample-driven detection and return result dicts plus debug images."""
        gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)

        # Step 2 — Preprocess full image (CLAHE + Otsu INV)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img_eq = clahe.apply(gray)
        _, img_bin = cv2.threshold(img_eq, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Step 3 — Width gating
        sp = st.session_state.sample_params
        stroke_width = sp.get('stroke_width', 3.0)
        dist_full = cv2.distanceTransform(img_bin, cv2.DIST_L2, 3)
        tol = max(1.0, 0.15 * stroke_width)
        lower = max(0.5, (stroke_width / 2.0) - tol)
        upper = (stroke_width / 2.0) + tol
        mask_width = ((dist_full >= lower) & (dist_full <= upper)).astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask_width = cv2.morphologyEx(mask_width, cv2.MORPH_OPEN, kernel, iterations=1)

        # Step 3b — Oriented matched filtering
        def make_line_kernel(length: float, thickness: float, angle_deg: float) -> np.ndarray:
            size = max(int(length * 1.5), int(thickness * 6), 9)
            if size % 2 == 0:
                size += 1
            kern = np.zeros((size, size), np.float32)
            cv2.line(kern, (size // 10, size // 2), (size - size // 10, size // 2), 1.0, int(max(1, round(thickness))))
            M = cv2.getRotationMatrix2D((size // 2, size // 2), angle_deg, 1.0)
            kern = cv2.warpAffine(kern, M, (size, size))
            s = kern.sum() + 1e-6
            return kern / s

        dash_len = sp.get('dash_len', float('nan'))
        dash_for_kernel = dash_len if (not math.isnan(dash_len) and dash_len > 2) else max(6.0, stroke_width * 3.0)
        angles = np.arange(0, 180, 10.0)
        resp_max = np.zeros_like(dist_full, dtype=np.float32)
        mask_width_f = mask_width.astype(np.float32)
        for ang in angles:
            k = make_line_kernel(dash_for_kernel, stroke_width, float(ang))
            r = cv2.filter2D(mask_width_f, -1, k, borderType=cv2.BORDER_REPLICATE)
            resp_max = np.maximum(resp_max, r)

        thr = 0.4 * float(resp_max.max() if resp_max.size else 0.0)
        match_mask = (resp_max >= thr).astype(np.uint8)
        se = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        match_mask = cv2.morphologyEx(match_mask, cv2.MORPH_CLOSE, se, iterations=1)

        # Step 4 — Skeletonize and trace polylines
        skel = skeletonize(match_mask > 0).astype(np.uint8)
        polylines = self._trace_skeleton_to_polylines(skel)

        # Convert to result dicts
        results = self._polylines_to_results(polylines)

        # Prepare debug visuals
        debug: Dict[str, Any] = {}
        debug['img_eq'] = img_eq
        debug['img_bin'] = img_bin
        debug['mask_width'] = mask_width
        # Normalize resp for display
        rmin, rmax = float(resp_max.min()), float(resp_max.max())
        if rmax > rmin:
            resp_norm = ((resp_max - rmin) / (rmax - rmin + 1e-6) * 255.0).astype(np.uint8)
        else:
            resp_norm = (resp_max * 0).astype(np.uint8)
        debug['resp_max_norm'] = resp_norm
        debug['match_mask'] = match_mask
        debug['skeleton'] = skel

        return results, debug

    def _trace_skeleton_to_polylines(self, skel: np.ndarray) -> List[List[Tuple[int, int]]]:
        """Trace skeleton (uint8) into ordered polylines; returns list of (y, x) points."""
        H, W = skel.shape
        visited = np.zeros_like(skel, dtype=bool)

        def neighbors(y: int, x: int):
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dy == 0 and dx == 0:
                        continue
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < H and 0 <= nx < W and skel[ny, nx]:
                        yield ny, nx

        def degree(y: int, x: int) -> int:
            return sum(1 for _ in neighbors(y, x))

        polylines: List[List[Tuple[int, int]]] = []

        def trace_from(start: Tuple[int, int]) -> List[Tuple[int, int]]:
            path: List[Tuple[int, int]] = [start]
            y, x = start
            visited[y, x] = True
            nbrs = [n for n in neighbors(y, x) if not visited[n]]
            if not nbrs:
                return path
            y, x = nbrs[0]
            path.append((y, x))
            visited[y, x] = True
            while True:
                nbrs = [n for n in neighbors(y, x) if not visited[n]]
                if not nbrs:
                    break
                if len(nbrs) == 1:
                    y, x = nbrs[0]
                    path.append((y, x))
                    visited[y, x] = True
                else:
                    break
            return path

        # Trace from endpoints first
        for sy in range(H):
            for sx in range(W):
                if skel[sy, sx] and not visited[sy, sx] and degree(sy, sx) == 1:
                    poly = trace_from((sy, sx))
                    if len(poly) >= 10:
                        polylines.append(poly)

        # Then any remaining loops
        for sy in range(H):
            for sx in range(W):
                if skel[sy, sx] and not visited[sy, sx]:
                    poly = trace_from((sy, sx))
                    if len(poly) >= 10:
                        polylines.append(poly)

        return polylines

    def _polylines_to_results(self, polylines: List[List[Tuple[int, int]]]) -> List[Dict[str, Any]]:
        """Convert (y,x) polylines to result dicts compatible with results UI."""
        def length_of_poly(points_xy: List[Tuple[int, int]]) -> float:
            total = 0.0
            for i in range(len(points_xy) - 1):
                x1, y1 = points_xy[i]
                x2, y2 = points_xy[i+1]
                total += float(((x2 - x1)**2 + (y2 - y1)**2) ** 0.5)
            return total

        results: List[Dict[str, Any]] = []
        for idx, poly in enumerate(polylines):
            if len(poly) < 2:
                continue
            # Convert to (x,y)
            points_xy: List[Tuple[int, int]] = [(int(x), int(y)) for (y, x) in poly]
            length_px = length_of_poly(points_xy)
            dx = points_xy[-1][0] - points_xy[0][0]
            dy = points_xy[-1][1] - points_xy[0][1]
            angle_deg = math.degrees(math.atan2(dy, dx)) if (dx != 0 or dy != 0) else 0.0
            results.append({
                'id': f'nb_line_{idx}',
                'class': 'dashed_service_line',
                'points': points_xy,
                'angle_deg': angle_deg,
                'length_px': length_px,
                'confidence': 0.0,
                'detection_method': 'notebook'
            })
        return results

    def _make_overlay_from_results(self, results: List[Dict[str, Any]]) -> np.ndarray:
        """Create an RGB overlay image with results drawn in red over the original image."""
        base = self.original_image
        if base is None:
            return np.zeros((100, 100, 3), dtype=np.uint8)
        overlay = base.copy()
        for line in results:
            pts_list = line.get('points', [])
            if len(pts_list) < 2:
                continue
            pts = np.array(pts_list, dtype=np.int32)
            cv2.polylines(overlay, [pts], isClosed=False, color=(0, 0, 255), thickness=2)
        return cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
    
    # Removed manual ROI preview and classic/robust unrelated helpers to keep only notebook pipeline
    
    # Removed legacy legend/classic/robust visualization utilities


def main():
    """Run the Streamlit GUI."""
    gui = DashDetectionGUI()
    gui.run()


if __name__ == "__main__":
    main()
