"""
Demo Script for Dashed Line Detection System

This script demonstrates the complete workflow of the dashed line detection system
with a synthetic floor plan example.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from integrated_dash_detector import IntegratedDashDetector
import os


def create_demo_floor_plan(width=800, height=600):
    """
    Create a synthetic floor plan with various dashed line types for demonstration.
    
    Returns:
        numpy.ndarray: Synthetic floor plan image
    """
    # Create white background
    image = np.ones((height, width, 3), dtype=np.uint8) * 255
    
    # Add title
    cv2.putText(image, "DEMO FLOOR PLAN", (50, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    
    # Add scale text
    cv2.putText(image, "SCALE 1:100", (width-200, height-30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    # Draw room boundaries (solid lines)
    room_points = [
        [(100, 100), (700, 100)],  # Top wall
        [(700, 100), (700, 400)],  # Right wall
        [(700, 400), (100, 400)],  # Bottom wall
        [(100, 400), (100, 100)]   # Left wall
    ]
    
    for start, end in room_points:
        cv2.line(image, start, end, (0, 0, 0), 3)
    
    # Add dashed service lines (HVAC ducts)
    hvac_lines = [
        [(150, 150), (650, 150)],  # Horizontal duct
        [(200, 120), (200, 380)],  # Vertical duct
        [(500, 120), (500, 200)],  # Short vertical duct
    ]
    
    for start, end in hvac_lines:
        draw_dashed_line(image, start, end, (255, 0, 0), dash_length=15, gap_length=10)
    
    # Add utility lines (plumbing)
    utility_lines = [
        [(300, 130), (300, 370)],  # Vertical pipe
        [(130, 250), (370, 250)], # Horizontal pipe
        [(500, 250), (670, 250)], # Another horizontal pipe
    ]
    
    for start, end in utility_lines:
        draw_dashed_line(image, start, end, (0, 0, 255), dash_length=8, gap_length=8)
    
    # Add boundary lines (property lines)
    boundary_lines = [
        [(50, 50), (750, 50)],   # Top boundary
        [(50, 450), (750, 450)], # Bottom boundary
    ]
    
    for start, end in boundary_lines:
        draw_dashed_line(image, start, end, (0, 255, 0), dash_length=20, gap_length=15)
    
    # Add construction lines
    construction_lines = [
        [(400, 130), (400, 370)],  # Future wall
        [(130, 300), (670, 300)],  # Future opening
    ]
    
    for start, end in construction_lines:
        draw_dot_dash_line(image, start, end, (255, 0, 255))
    
    # Add legend
    add_legend(image, width-180, 80)
    
    return image


def draw_dashed_line(image, start, end, color, dash_length=10, gap_length=5, thickness=2):
    """Draw a dashed line on the image."""
    x1, y1 = start
    x2, y2 = end
    
    # Calculate line parameters
    dx = x2 - x1
    dy = y2 - y1
    line_length = np.sqrt(dx*dx + dy*dy)
    
    if line_length == 0:
        return
    
    # Unit direction vector
    ux = dx / line_length
    uy = dy / line_length
    
    # Draw dashes
    current_pos = 0
    while current_pos < line_length:
        # Dash start
        dash_start_x = int(x1 + current_pos * ux)
        dash_start_y = int(y1 + current_pos * uy)
        
        # Dash end
        dash_end_pos = min(current_pos + dash_length, line_length)
        dash_end_x = int(x1 + dash_end_pos * ux)
        dash_end_y = int(y1 + dash_end_pos * uy)
        
        # Draw the dash
        cv2.line(image, (dash_start_x, dash_start_y), (dash_end_x, dash_end_y), color, thickness)
        
        # Move to next dash
        current_pos += dash_length + gap_length


def draw_dot_dash_line(image, start, end, color, dot_length=3, dash_length=15, gap_length=5, thickness=2):
    """Draw a dot-dash line pattern."""
    x1, y1 = start
    x2, y2 = end
    
    # Calculate line parameters
    dx = x2 - x1
    dy = y2 - y1
    line_length = np.sqrt(dx*dx + dy*dy)
    
    if line_length == 0:
        return
    
    # Unit direction vector
    ux = dx / line_length
    uy = dy / line_length
    
    # Draw dot-dash pattern
    current_pos = 0
    is_dot = True  # Start with dot
    
    while current_pos < line_length:
        if is_dot:
            # Draw dot
            segment_length = dot_length
        else:
            # Draw dash
            segment_length = dash_length
        
        # Segment start
        seg_start_x = int(x1 + current_pos * ux)
        seg_start_y = int(y1 + current_pos * uy)
        
        # Segment end
        seg_end_pos = min(current_pos + segment_length, line_length)
        seg_end_x = int(x1 + seg_end_pos * ux)
        seg_end_y = int(y1 + seg_end_pos * uy)
        
        # Draw the segment
        cv2.line(image, (seg_start_x, seg_start_y), (seg_end_x, seg_end_y), color, thickness)
        
        # Move to next segment
        current_pos += segment_length + gap_length
        is_dot = not is_dot  # Alternate between dot and dash


def add_legend(image, x, y):
    """Add a legend to the image."""
    # Legend background
    cv2.rectangle(image, (x-10, y-10), (x+160, y+120), (240, 240, 240), -1)
    cv2.rectangle(image, (x-10, y-10), (x+160, y+120), (0, 0, 0), 1)
    
    # Legend title
    cv2.putText(image, "LEGEND", (x, y+10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    # Legend entries
    entries = [
        ("HVAC Duct", (255, 0, 0), 15, 10),
        ("Plumbing", (0, 0, 255), 8, 8),
        ("Property Line", (0, 255, 0), 20, 15),
        ("Future Wall", (255, 0, 255), None, None)  # Dot-dash
    ]
    
    for i, (label, color, dash_len, gap_len) in enumerate(entries):
        y_pos = y + 25 + i * 20
        
        # Draw sample line
        if dash_len is not None:
            draw_dashed_line(image, (x, y_pos), (x+30, y_pos), color, dash_len//2, gap_len//2, 1)
        else:
            draw_dot_dash_line(image, (x, y_pos), (x+30, y_pos), color, 2, 8, 3, 1)
        
        # Draw label
        cv2.putText(image, label, (x+35, y_pos+3), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1)


def run_demo():
    """Run the complete demonstration."""
    print("🚀 Dashed Line Detection System Demo")
    print("=" * 50)
    
    # Step 1: Create demo floor plan
    print("1. Creating synthetic floor plan...")
    demo_image = create_demo_floor_plan()
    
    # Save demo image
    demo_path = "demo_floor_plan.png"
    cv2.imwrite(demo_path, demo_image)
    print(f"   ✅ Demo floor plan saved: {demo_path}")
    
    # Step 2: Initialize detector
    print("\n2. Initializing detection system...")
    detector = IntegratedDashDetector()
    
    # Configure for demo
    detector.config['preprocessing']['apply_deskew'] = False  # Not needed for synthetic image
    detector.config['detection']['use_robust'] = True
    detector.config['classification']['use_legend'] = True
    
    print("   ✅ Detector initialized")
    
    # Step 3: Load image
    print("\n3. Loading demo image...")
    if detector.load_image(demo_path):
        print("   ✅ Image loaded successfully")
        h, w = detector.original_image.shape[:2]
        print(f"   📐 Image size: {w} × {h} pixels")
    else:
        print("   ❌ Failed to load image")
        return
    
    # Step 4: Learn template from HVAC line
    print("\n4. Learning dash template...")
    # Select ROI from horizontal HVAC line
    roi_x, roi_y = 200, 140  # Position over HVAC line
    roi_width, roi_height = 60, 20
    
    if detector.learn_template_from_roi(roi_x, roi_y, roi_width, roi_height):
        template = detector.classic_detector.dash_template
        print("   ✅ Template learned successfully")
        print(f"   📏 Dash length: {template.dash_length} px")
        print(f"   📏 Gap length: {template.gap_length} px")
        print(f"   📏 Total period: {template.total_period} px")
    else:
        print("   ⚠️ Template learning failed, continuing with robust methods only")
    
    # Step 5: Set scale (manual for demo)
    print("\n5. Setting scale...")
    # Use the scale text as reference (1:100 means 1 unit = 100 units in reality)
    # Assuming the room is about 20m wide, and it's 600px in the image
    scale_px_to_m = 20.0 / 600  # 20 meters / 600 pixels
    detector.scale_detector.scale_px_to_m = scale_px_to_m
    detector.classic_detector.set_scale(scale_px_to_m)
    print(f"   ✅ Manual scale set: {scale_px_to_m:.6f} px/m")
    
    # Step 6: Detect legend
    print("\n6. Detecting legend...")
    if detector.detect_legend():
        print(f"   ✅ Legend detected with {len(detector.legend_entries)} entries")
        for i, entry in enumerate(detector.legend_entries):
            print(f"      {i+1}. {entry.get('text', 'N/A')} - {entry.get('line_type', 'unknown')}")
    else:
        print("   ⚠️ No legend detected")
    
    # Step 7: Run complete detection
    print("\n7. Running complete detection pipeline...")
    results = detector.detect_all()
    
    if results:
        print(f"   ✅ Detection complete! Found {len(results)} dashed lines")
        
        # Show summary
        total_length = sum(line.get('length_m', 0) for line in results)
        print(f"   📏 Total length: {total_length:.2f} meters")
        
        # Count by type
        line_types = {}
        for line in results:
            line_type = line.get('classified_type', line.get('class', 'unknown'))
            line_types[line_type] = line_types.get(line_type, 0) + 1
        
        print("   📊 Line types detected:")
        for line_type, count in line_types.items():
            print(f"      • {line_type}: {count} lines")
            
    else:
        print("   ⚠️ No dashed lines detected")
        return
    
    # Step 8: Export results
    print("\n8. Exporting results...")
    output_dir = "demo_output"
    os.makedirs(output_dir, exist_ok=True)
    
    output_files = detector.export_results(output_dir)
    
    if output_files:
        print("   ✅ Export complete! Generated files:")
        for format_name, file_path in output_files.items():
            print(f"      • {format_name.upper()}: {file_path}")
    else:
        print("   ❌ Export failed")
    
    # Step 9: Show visualization
    print("\n9. Creating visualization...")
    
    try:
        # Create overlay visualization
        overlay = demo_image.copy()
        
        # Colors for different line types
        colors = {
            'dashed': (0, 255, 255),      # Yellow
            'dashed_service_line': (0, 255, 255),
            'dashed_utility_line': (255, 255, 0),  # Cyan
            'unknown': (255, 0, 255),     # Magenta
            'default': (0, 255, 0)        # Green
        }
        
        for line in results:
            points = line.get('points', [])
            if len(points) < 2:
                continue
            
            line_type = line.get('classified_type', line.get('class', 'unknown'))
            color = colors.get(line_type, colors['default'])
            
            # Draw detected line
            pts = np.array(points, dtype=np.int32)
            cv2.polylines(overlay, [pts], False, color, 3)
            
            # Add length label
            if 'length_m' in line and line['length_m'] > 0:
                mid_idx = len(points) // 2
                if mid_idx < len(points):
                    label_pos = (int(points[mid_idx][0]), int(points[mid_idx][1] - 10))
                    label = f"{line['length_m']:.1f}m"
                    cv2.putText(overlay, label, label_pos, 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # Save visualization
        viz_path = os.path.join(output_dir, "detection_visualization.png")
        cv2.imwrite(viz_path, overlay)
        print(f"   ✅ Visualization saved: {viz_path}")
        
    except Exception as e:
        print(f"   ⚠️ Visualization error: {e}")
    
    # Step 10: Display statistics
    print("\n10. Detection Statistics")
    print("-" * 30)
    
    stats = detector.get_detection_statistics()
    
    print(f"Image: {stats['image_info']['size']}")
    print(f"Scale: {stats['scale_info']['scale_px_to_m']:.6f} px/m")
    print(f"Legend entries: {stats['legend_info']['entries_found']}")
    
    if 'detection_summary' in stats:
        summary = stats['detection_summary']
        print(f"Total lines: {summary.get('total_lines', 0)}")
        print(f"Total length: {summary.get('total_length_m', 0):.2f} m")
        
        if 'line_types' in summary:
            print("Line types:")
            for line_type, count in summary['line_types'].items():
                print(f"  • {line_type}: {count}")
    
    print("\n🎉 Demo completed successfully!")
    print(f"📁 Check the '{output_dir}' folder for all output files")
    print("\n💡 Next steps:")
    print("   • Try the GUI: streamlit run dash_detection_gui.py")
    print("   • Test with your own floor plans")
    print("   • Adjust configuration for optimal results")


if __name__ == "__main__":
    run_demo()
