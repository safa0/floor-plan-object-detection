"""
Output Export Module for Dashed Line Detection

This module handles exporting detected dashed lines to various formats:
- SVG (Scalable Vector Graphics)
- GeoJSON (Geographic data format)
- DXF (AutoCAD format)
- CSV (Bill of Materials)
"""

import json
import csv
import xml.etree.ElementTree as ET
from xml.dom import minidom
import numpy as np
from typing import List, Dict, Any, Optional
import os
from datetime import datetime


class OutputExporter:
    """Exports detected dashed lines to various formats."""
    
    def __init__(self, image_width: int, image_height: int):
        self.image_width = image_width
        self.image_height = image_height
        self.coordinate_system = 'image'  # 'image' or 'world'
        
    def export_svg(self, 
                  lines: List[Dict], 
                  output_path: str,
                  include_overlay: bool = True,
                  background_image_path: Optional[str] = None) -> None:
        """
        Export lines to SVG format.
        
        Args:
            lines: List of detected line dictionaries
            output_path: Output SVG file path
            include_overlay: Whether to include original image as background
            background_image_path: Path to background image
        """
        # Create SVG root element
        svg = ET.Element('svg')
        svg.set('width', str(self.image_width))
        svg.set('height', str(self.image_height))
        svg.set('viewBox', f'0 0 {self.image_width} {self.image_height}')
        svg.set('xmlns', 'http://www.w3.org/2000/svg')
        svg.set('xmlns:xlink', 'http://www.w3.org/1999/xlink')
        
        # Add metadata
        title = ET.SubElement(svg, 'title')
        title.text = 'Detected Dashed Lines'
        
        desc = ET.SubElement(svg, 'desc')
        desc.text = f'Generated on {datetime.now().isoformat()}'
        
        # Add definitions for line styles
        defs = ET.SubElement(svg, 'defs')
        
        # Dashed line style
        dash_style = ET.SubElement(defs, 'style')
        dash_style.set('type', 'text/css')
        dash_style.text = '''
        .dashed-line {
            stroke: #ff0000;
            stroke-width: 2;
            stroke-dasharray: 5,5;
            fill: none;
        }
        .dashed-line-highlight {
            stroke: #00ff00;
            stroke-width: 3;
            stroke-dasharray: 8,3;
            fill: none;
            opacity: 0.8;
        }
        '''
        
        # Background image if provided
        if include_overlay and background_image_path:
            bg_image = ET.SubElement(svg, 'image')
            bg_image.set('x', '0')
            bg_image.set('y', '0')
            bg_image.set('width', str(self.image_width))
            bg_image.set('height', str(self.image_height))
            bg_image.set('xlink:href', background_image_path)
            bg_image.set('opacity', '0.7')
        
        # Add lines group
        lines_group = ET.SubElement(svg, 'g')
        lines_group.set('id', 'dashed-lines')
        
        # Color mapping for different line types
        colors = {
            'dashed_service_line': '#ff0000',  # Red
            'dashed_utility_line': '#0000ff',  # Blue
            'dashed_boundary': '#00ff00',      # Green
            'dashed_hidden': '#ffff00',        # Yellow
            'default': '#ff00ff'               # Magenta
        }
        
        for i, line in enumerate(lines):
            points = line.get('points', [])
            if len(points) < 2:
                continue
            
            # Create polyline element
            polyline = ET.SubElement(lines_group, 'polyline')
            
            # Convert points to SVG format
            points_str = ' '.join([f"{p[0]},{p[1]}" for p in points])
            polyline.set('points', points_str)
            
            # Set style based on line class
            line_class = line.get('class', 'default')
            color = colors.get(line_class, colors['default'])
            
            polyline.set('stroke', color)
            polyline.set('stroke-width', '2')
            polyline.set('stroke-dasharray', '5,3')
            polyline.set('fill', 'none')
            polyline.set('id', line.get('id', f'line_{i}'))
            
            # Add metadata as attributes
            if 'length_m' in line:
                polyline.set('data-length-m', f"{line['length_m']:.3f}")
            if 'angle_deg' in line:
                polyline.set('data-angle', f"{line['angle_deg']:.1f}")
            if 'confidence' in line:
                polyline.set('data-confidence', f"{line['confidence']:.3f}")
            
            # Add text label for length
            if 'length_m' in line and line['length_m'] > 0:
                # Calculate midpoint for label
                mid_idx = len(points) // 2
                if mid_idx < len(points):
                    mid_x, mid_y = points[mid_idx]
                    
                    text = ET.SubElement(lines_group, 'text')
                    text.set('x', str(mid_x + 5))
                    text.set('y', str(mid_y - 5))
                    text.set('font-family', 'Arial, sans-serif')
                    text.set('font-size', '12')
                    text.set('fill', color)
                    text.text = f"{line['length_m']:.2f}m"
        
        # Add legend
        legend_group = ET.SubElement(svg, 'g')
        legend_group.set('id', 'legend')
        legend_group.set('transform', f'translate(10, {self.image_height - 100})')
        
        # Legend background
        legend_bg = ET.SubElement(legend_group, 'rect')
        legend_bg.set('x', '0')
        legend_bg.set('y', '0')
        legend_bg.set('width', '200')
        legend_bg.set('height', '80')
        legend_bg.set('fill', 'white')
        legend_bg.set('stroke', 'black')
        legend_bg.set('opacity', '0.9')
        
        # Legend title
        legend_title = ET.SubElement(legend_group, 'text')
        legend_title.set('x', '10')
        legend_title.set('y', '15')
        legend_title.set('font-family', 'Arial, sans-serif')
        legend_title.set('font-size', '14')
        legend_title.set('font-weight', 'bold')
        legend_title.text = 'Detected Lines'
        
        # Legend entries
        y_pos = 30
        unique_classes = list(set(line.get('class', 'default') for line in lines))
        
        for line_class in unique_classes:
            color = colors.get(line_class, colors['default'])
            
            # Legend line
            legend_line = ET.SubElement(legend_group, 'line')
            legend_line.set('x1', '10')
            legend_line.set('y1', str(y_pos))
            legend_line.set('x2', '30')
            legend_line.set('y2', str(y_pos))
            legend_line.set('stroke', color)
            legend_line.set('stroke-width', '2')
            legend_line.set('stroke-dasharray', '5,3')
            
            # Legend text
            legend_text = ET.SubElement(legend_group, 'text')
            legend_text.set('x', '35')
            legend_text.set('y', str(y_pos + 3))
            legend_text.set('font-family', 'Arial, sans-serif')
            legend_text.set('font-size', '10')
            legend_text.text = line_class.replace('_', ' ').title()
            
            y_pos += 15
        
        # Write to file
        tree = ET.ElementTree(svg)
        ET.indent(tree, space="  ", level=0)
        tree.write(output_path, encoding='utf-8', xml_declaration=True)
        
        print(f"SVG exported to: {output_path}")
    
    def export_geojson(self, 
                      lines: List[Dict], 
                      output_path: str,
                      coordinate_reference_system: str = "EPSG:4326") -> None:
        """
        Export lines to GeoJSON format.
        
        Args:
            lines: List of detected line dictionaries
            output_path: Output GeoJSON file path
            coordinate_reference_system: CRS identifier
        """
        # Create GeoJSON structure
        geojson = {
            "type": "FeatureCollection",
            "crs": {
                "type": "name",
                "properties": {
                    "name": coordinate_reference_system
                }
            },
            "metadata": {
                "generated": datetime.now().isoformat(),
                "image_size": [self.image_width, self.image_height],
                "total_lines": len(lines)
            },
            "features": []
        }
        
        for i, line in enumerate(lines):
            points = line.get('points', [])
            if len(points) < 2:
                continue
            
            # Convert image coordinates to geographic coordinates if needed
            # For now, keep as image coordinates (could add projection later)
            coordinates = [[float(p[0]), float(p[1])] for p in points]
            
            # Create feature
            feature = {
                "type": "Feature",
                "id": line.get('id', f'line_{i}'),
                "geometry": {
                    "type": "LineString",
                    "coordinates": coordinates
                },
                "properties": {
                    "class": line.get('class', 'dashed_line'),
                    "length_px": line.get('length_px', 0),
                    "length_m": line.get('length_m', 0),
                    "angle_deg": line.get('angle_deg', 0),
                    "confidence": line.get('confidence', 0),
                    "detection_method": "template_matching"
                }
            }
            
            geojson["features"].append(feature)
        
        # Write to file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(geojson, f, indent=2, ensure_ascii=False)
        
        print(f"GeoJSON exported to: {output_path}")
    
    def export_dxf(self, lines: List[Dict], output_path: str) -> None:
        """
        Export lines to DXF format (basic implementation).
        
        Note: This is a simplified DXF export. For production use,
        consider using a library like ezdxf.
        """
        try:
            import ezdxf
            
            # Create new DXF document
            doc = ezdxf.new('R2010')
            msp = doc.modelspace()
            
            # Define layers for different line types
            layers = {
                'dashed_service_line': {'color': 1, 'linetype': 'DASHED'},  # Red
                'dashed_utility_line': {'color': 5, 'linetype': 'DASHED'},  # Blue
                'dashed_boundary': {'color': 3, 'linetype': 'DASHED'},      # Green
                'dashed_hidden': {'color': 2, 'linetype': 'HIDDEN'},        # Yellow
                'default': {'color': 7, 'linetype': 'DASHED'}               # White
            }
            
            # Create layers
            for layer_name, props in layers.items():
                layer = doc.layers.new(name=layer_name)
                layer.color = props['color']
                # Note: Linetype setup requires more complex DXF handling
            
            # Add lines
            for i, line in enumerate(lines):
                points = line.get('points', [])
                if len(points) < 2:
                    continue
                
                line_class = line.get('class', 'default')
                layer_name = line_class if line_class in layers else 'default'
                
                # Convert points to DXF coordinates (flip Y axis)
                dxf_points = [(p[0], self.image_height - p[1]) for p in points]
                
                # Create polyline
                polyline = msp.add_lwpolyline(dxf_points)
                polyline.layer = layer_name
                
                # Add attributes as extended data if needed
                if 'length_m' in line:
                    # Add text annotation
                    mid_point = dxf_points[len(dxf_points) // 2]
                    text = msp.add_text(f"{line['length_m']:.2f}m")
                    text.dxf.layer = layer_name
                    text.dxf.height = 5
                    text.set_pos(mid_point)
            
            # Save DXF file
            doc.saveas(output_path)
            print(f"DXF exported to: {output_path}")
            
        except ImportError:
            print("Warning: ezdxf library not available. Creating simple DXF.")
            self._export_simple_dxf(lines, output_path)
    
    def _export_simple_dxf(self, lines: List[Dict], output_path: str) -> None:
        """Simple DXF export without external libraries."""
        with open(output_path, 'w') as f:
            # Basic DXF header
            f.write("0\nSECTION\n2\nHEADER\n")
            f.write("9\n$ACADVER\n1\nAC1014\n")
            f.write("0\nENDSEC\n")
            
            # Entities section
            f.write("0\nSECTION\n2\nENTITIES\n")
            
            for i, line in enumerate(lines):
                points = line.get('points', [])
                if len(points) < 2:
                    continue
                
                # Write POLYLINE entity
                f.write("0\nPOLYLINE\n")
                f.write("8\nDASHED_LINES\n")  # Layer name
                f.write("66\n1\n")  # Entities follow flag
                f.write("70\n0\n")  # Polyline flag
                
                # Write vertices
                for point in points:
                    f.write("0\nVERTEX\n")
                    f.write("8\nDASHED_LINES\n")
                    f.write(f"10\n{point[0]}\n")  # X coordinate
                    f.write(f"20\n{self.image_height - point[1]}\n")  # Y coordinate (flipped)
                    f.write("30\n0.0\n")  # Z coordinate
                
                f.write("0\nSEQEND\n")
            
            f.write("0\nENDSEC\n")
            f.write("0\nEOF\n")
        
        print(f"Simple DXF exported to: {output_path}")
    
    def export_csv_bom(self, 
                      lines: List[Dict], 
                      output_path: str,
                      group_by_class: bool = True) -> None:
        """
        Export Bill of Materials (BOM) to CSV format.
        
        Args:
            lines: List of detected line dictionaries
            output_path: Output CSV file path
            group_by_class: Whether to group lines by class
        """
        if group_by_class:
            # Group lines by class and calculate totals
            class_totals = {}
            
            for line in lines:
                line_class = line.get('class', 'unknown')
                length_m = line.get('length_m', 0)
                
                if line_class not in class_totals:
                    class_totals[line_class] = {
                        'count': 0,
                        'total_length_m': 0,
                        'lines': []
                    }
                
                class_totals[line_class]['count'] += 1
                class_totals[line_class]['total_length_m'] += length_m
                class_totals[line_class]['lines'].append(line)
            
            # Write grouped CSV
            with open(output_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                
                # Header
                writer.writerow([
                    'Line Class',
                    'Count',
                    'Total Length (m)',
                    'Average Length (m)',
                    'Description'
                ])
                
                # Summary rows
                for line_class, data in class_totals.items():
                    avg_length = data['total_length_m'] / data['count'] if data['count'] > 0 else 0
                    description = self._get_class_description(line_class)
                    
                    writer.writerow([
                        line_class,
                        data['count'],
                        f"{data['total_length_m']:.3f}",
                        f"{avg_length:.3f}",
                        description
                    ])
                
                # Overall total
                total_count = sum(data['count'] for data in class_totals.values())
                total_length = sum(data['total_length_m'] for data in class_totals.values())
                
                writer.writerow([])  # Empty row
                writer.writerow(['TOTAL', total_count, f"{total_length:.3f}", '', 'All dashed lines'])
        
        else:
            # Individual line listing
            with open(output_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                
                # Header
                writer.writerow([
                    'Line ID',
                    'Class',
                    'Length (m)',
                    'Length (px)',
                    'Angle (deg)',
                    'Confidence',
                    'Start Point',
                    'End Point'
                ])
                
                # Individual lines
                for line in lines:
                    points = line.get('points', [])
                    start_point = f"({points[0][0]:.1f}, {points[0][1]:.1f})" if points else "N/A"
                    end_point = f"({points[-1][0]:.1f}, {points[-1][1]:.1f})" if len(points) > 1 else "N/A"
                    
                    writer.writerow([
                        line.get('id', 'N/A'),
                        line.get('class', 'unknown'),
                        f"{line.get('length_m', 0):.3f}",
                        f"{line.get('length_px', 0):.1f}",
                        f"{line.get('angle_deg', 0):.1f}",
                        f"{line.get('confidence', 0):.3f}",
                        start_point,
                        end_point
                    ])
        
        print(f"CSV BOM exported to: {output_path}")
    
    def _get_class_description(self, line_class: str) -> str:
        """Get human-readable description for line class."""
        descriptions = {
            'dashed_service_line': 'Service lines (utilities, HVAC)',
            'dashed_utility_line': 'Utility connections',
            'dashed_boundary': 'Property or zone boundaries',
            'dashed_hidden': 'Hidden or underground elements',
            'dashed_construction': 'Construction/demolition lines',
            'default': 'Unclassified dashed lines'
        }
        
        return descriptions.get(line_class, 'Unknown line type')
    
    def export_overlay_image(self, 
                           original_image: np.ndarray,
                           lines: List[Dict],
                           output_path: str,
                           line_width: int = 2) -> None:
        """
        Export overlay image with detected lines highlighted.
        
        Args:
            original_image: Original floor plan image
            lines: List of detected line dictionaries
            output_path: Output image path
            line_width: Width of overlay lines
        """
        import cv2
        
        # Create overlay
        overlay = original_image.copy()
        
        # Color mapping
        colors = {
            'dashed_service_line': (0, 0, 255),     # Red (BGR)
            'dashed_utility_line': (255, 0, 0),    # Blue
            'dashed_boundary': (0, 255, 0),        # Green
            'dashed_hidden': (0, 255, 255),        # Yellow
            'default': (255, 0, 255)               # Magenta
        }
        
        for line in lines:
            points = line.get('points', [])
            if len(points) < 2:
                continue
            
            line_class = line.get('class', 'default')
            color = colors.get(line_class, colors['default'])
            
            # Convert points to numpy array
            pts = np.array(points, dtype=np.int32)
            
            # Draw polyline
            cv2.polylines(overlay, [pts], False, color, line_width)
            
            # Add length label
            if 'length_m' in line and line['length_m'] > 0:
                mid_idx = len(points) // 2
                if mid_idx < len(points):
                    label_pos = (int(points[mid_idx][0]), int(points[mid_idx][1] - 10))
                    label = f"{line['length_m']:.2f}m"
                    
                    cv2.putText(overlay, label, label_pos, 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Save overlay image
        cv2.imwrite(output_path, overlay)
        print(f"Overlay image exported to: {output_path}")
    
    def export_all_formats(self, 
                          lines: List[Dict],
                          original_image: np.ndarray,
                          base_filename: str,
                          output_dir: str = "output") -> Dict[str, str]:
        """
        Export to all supported formats.
        
        Args:
            lines: List of detected line dictionaries
            original_image: Original floor plan image
            base_filename: Base filename (without extension)
            output_dir: Output directory
            
        Returns:
            Dictionary mapping format names to output file paths
        """
        os.makedirs(output_dir, exist_ok=True)
        
        output_files = {}
        
        # SVG
        svg_path = os.path.join(output_dir, f"{base_filename}.svg")
        self.export_svg(lines, svg_path)
        output_files['svg'] = svg_path
        
        # GeoJSON
        geojson_path = os.path.join(output_dir, f"{base_filename}.geojson")
        self.export_geojson(lines, geojson_path)
        output_files['geojson'] = geojson_path
        
        # DXF
        dxf_path = os.path.join(output_dir, f"{base_filename}.dxf")
        self.export_dxf(lines, dxf_path)
        output_files['dxf'] = dxf_path
        
        # CSV (both grouped and individual)
        csv_grouped_path = os.path.join(output_dir, f"{base_filename}_bom_grouped.csv")
        self.export_csv_bom(lines, csv_grouped_path, group_by_class=True)
        output_files['csv_grouped'] = csv_grouped_path
        
        csv_individual_path = os.path.join(output_dir, f"{base_filename}_bom_individual.csv")
        self.export_csv_bom(lines, csv_individual_path, group_by_class=False)
        output_files['csv_individual'] = csv_individual_path
        
        # Overlay image
        overlay_path = os.path.join(output_dir, f"{base_filename}_overlay.png")
        self.export_overlay_image(original_image, lines, overlay_path)
        output_files['overlay'] = overlay_path
        
        print(f"All formats exported to: {output_dir}")
        return output_files


# Example usage
def test_export_system():
    """Test the export system with dummy data."""
    # Create dummy data
    lines = [
        {
            'id': 'line_1',
            'class': 'dashed_service_line',
            'points': [(100, 100), (200, 150), (300, 120)],
            'length_px': 224.7,
            'length_m': 2.247,
            'angle_deg': 15.5,
            'confidence': 0.95
        },
        {
            'id': 'line_2', 
            'class': 'dashed_utility_line',
            'points': [(150, 200), (250, 200)],
            'length_px': 100.0,
            'length_m': 1.0,
            'angle_deg': 0.0,
            'confidence': 0.87
        }
    ]
    
    # Create dummy image
    dummy_image = np.ones((400, 500, 3), dtype=np.uint8) * 255
    
    # Test export
    exporter = OutputExporter(500, 400)
    output_files = exporter.export_all_formats(lines, dummy_image, "test_export", "test_output")
    
    print("Export test completed:")
    for format_name, file_path in output_files.items():
        print(f"  {format_name}: {file_path}")


if __name__ == "__main__":
    test_export_system()
