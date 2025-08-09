import streamlit as st
from ultralytics import YOLO
import PIL
from PIL import Image, ImageDraw, ImageFont
import helper
import setting
import numpy as np
import math
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import time
import os
import shutil

# Increase PIL's decompression bomb limit to support large floor plan images
# Current image: 238,861,095 pixels, default limit: 178,956,970 pixels
# Setting to 500M pixels to provide headroom for even larger images
PIL.Image.MAX_IMAGE_PIXELS = 500_000_000

# Configure PIL for high quality PNG handling
from PIL import PngImagePlugin
# Increase the PNG chunk limit to handle large PNG files properly
PngImagePlugin.MAX_TEXT_CHUNK = 10 * (1024**2)  # 10MB chunks

# Allowlist Ultralytics model class for PyTorch 2.6+ safe-loading
try:
    import torch
    from ultralytics.nn.tasks import DetectionModel as _UltralyticsDetectionModel
    if hasattr(torch, "serialization") and hasattr(torch.serialization, "add_safe_globals"):
        torch.serialization.add_safe_globals([_UltralyticsDetectionModel])
    # Force torch.load to default to weights_only=False (trust only if checkpoint source is trusted)
    if not hasattr(torch, '_ultralytics_patched'):
        _orig_torch_load = torch.load
        def _torch_load_compat(*args, **kwargs):
            if "weights_only" not in kwargs:
                kwargs["weights_only"] = False
            return _orig_torch_load(*args, **kwargs)
        torch.load = _torch_load_compat
        torch._ultralytics_patched = True
except Exception:
    st.warning("Unable to configure PyTorch safe-loading for Ultralytics model")
    print("Warning: Unable to configure PyTorch safe-loading for Ultralytics model") 


def load_image_with_quality_preservation(source_img):
    """
    Load image while preserving maximum quality, especially for PNG files.
    Enhanced preprocessing for better detection accuracy.
    """
    # Ensure file pointer is at the beginning
    source_img.seek(0)
    
    # Open the image
    img = PIL.Image.open(source_img)
    
    # For PNG files, ensure we maintain quality and handle transparency properly
    if hasattr(source_img, 'name') and source_img.name.lower().endswith('.png'):
        # Keep original PNG properties
        if img.mode == 'P':  # Palette mode
            img = img.convert('RGBA')
        elif img.mode not in ['RGB', 'RGBA']:
            img = img.convert('RGBA')
    
    # Ensure consistent RGB format for YOLO (no alpha channel issues)
    if img.mode == 'RGBA':
        # Create white background for transparency
        background = PIL.Image.new('RGB', img.size, (255, 255, 255))
        background.paste(img, mask=img.split()[-1])  # Use alpha channel as mask
        img = background
    elif img.mode != 'RGB':
        img = img.convert('RGB')
    
    return img


def draw_annotations(image, filtered_boxes, model, label_font_size, show_confidence=True):
    """
    Draw annotations on image with specified font size and optional confidence scores.
    """
    annotated_image = image.copy()
    draw = ImageDraw.Draw(annotated_image)

    # Attempt to load a common TrueType font; fallback to default
    font = None
    for font_path in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
        "DejaVuSans.ttf",
    ]:
        try:
            font = ImageFont.truetype(font_path, label_font_size)
            break
        except Exception:
            continue
    if font is None:
        try:
            font = ImageFont.load_default()
        except Exception:
            font = None

    # Simple color palette and improved thickness scaling
    palette = [
        (255, 56, 56), (255, 157, 151), (255, 112, 31), (255, 178, 29),
        (207, 210, 49), (72, 249, 10), (146, 204, 23), (61, 219, 134),
        (26, 147, 52), (0, 212, 187), (44, 153, 168), (0, 194, 255),
        (52, 69, 147), (100, 115, 255), (142, 140, 255), (204, 182, 142),
        (255, 173, 203), (255, 0, 0), (255, 165, 0), (255, 255, 0),
        (0, 255, 0), (0, 255, 255), (0, 0, 255), (255, 0, 255)
    ]
    # Better thickness scaling that works for both small and large images
    box_thickness = max(1, label_font_size // 6)  # More proportional to font size
    pad = max(2, label_font_size // 8)  # Padding scales with font size

    for box in filtered_boxes:
        xyxy = box.xyxy[0]
        
        # Handle both tensor and numpy array formats
        if hasattr(xyxy, 'cpu'):  # PyTorch tensor
            xyxy = xyxy.cpu().numpy()
        
        x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
        
        # Handle cls attribute (could be tensor or scalar)
        if hasattr(box.cls, 'item'):  # PyTorch tensor
            cls_id = int(box.cls.item())
        else:
            cls_id = int(box.cls)
            
        # Handle conf attribute (could be tensor or scalar)
        if hasattr(box.conf[0], 'item'):  # PyTorch tensor
            confidence = float(box.conf[0].item())
        else:
            confidence = float(box.conf[0])
        if show_confidence:
            label = f"{model.names[cls_id]} ({confidence:.2f})"
        else:
            label = model.names[cls_id]
        color = palette[cls_id % len(palette)]

        # Draw rectangle
        draw.rectangle([(x1, y1), (x2, y2)], outline=color, width=box_thickness)

        # Measure text size
        if font is not None:
            try:
                tw, th = draw.textbbox((0, 0), label, font=font)[2:4]
            except Exception:
                tw, th = draw.textsize(label, font=font)
        else:
            tw, th = draw.textsize(label)

        # Position text above box; if it goes out of frame, place inside the box
        text_x = x1 + box_thickness
        text_y = y1 - th - pad
        if text_y < 0:
            text_y = y1 + box_thickness

        # Background for text for readability
        bg_rect = [(text_x - pad, text_y - pad), (text_x + tw + pad, text_y + th + pad)]
        draw.rectangle(bg_rect, fill=color)
        # Text in white
        if font is not None:
            draw.text((text_x, text_y), label, fill=(255, 255, 255), font=font)
        else:
            draw.text((text_x, text_y), label, fill=(255, 255, 255))

    return annotated_image


def create_tile_overlay_visualization(image, tile_size=640, overlap=64):
    """
    Create a visualization showing tile boundaries overlaid on the original image.
    
    Args:
        image: PIL Image object
        tile_size: Size of each tile
        overlap: Overlap between tiles
    
    Returns:
        PIL Image with tile grid overlay
    """
    # Create a copy of the image for overlay
    overlay_image = image.copy()
    draw = ImageDraw.Draw(overlay_image)
    
    width, height = image.size
    step_size = tile_size - overlap
    
    # Colors for different elements
    tile_border_color = (255, 0, 0)  # Red for tile borders
    overlap_color = (255, 255, 0, 128)  # Semi-transparent yellow for overlap areas
    grid_color = (0, 255, 0)  # Green for grid lines
    
    # Draw tile boundaries and overlaps
    tile_count = 0
    for y in range(0, height, step_size):
        for x in range(0, width, step_size):
            # Calculate tile boundaries
            x_end = min(x + tile_size, width)
            y_end = min(y + tile_size, height)
            
            # Adjust starting position if we're at the edge
            x_start = max(0, x_end - tile_size)
            y_start = max(0, y_end - tile_size)
            
            # Draw tile border
            draw.rectangle([(x_start, y_start), (x_end, y_end)], 
                         outline=tile_border_color, width=3)
            
            # Draw tile number
            tile_count += 1
            tile_center_x = x_start + (x_end - x_start) // 2
            tile_center_y = y_start + (y_end - y_start) // 2
            
            # Try to get a font for the tile numbers
            font = None
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
            except:
                try:
                    font = ImageFont.load_default()
                except:
                    pass
            
            # Draw tile number with background
            tile_text = str(tile_count)
            if font:
                bbox = draw.textbbox((0, 0), tile_text, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
            else:
                text_width = len(tile_text) * 6
                text_height = 11
            
            text_x = tile_center_x - text_width // 2
            text_y = tile_center_y - text_height // 2
            
            # Background for text
            draw.rectangle([(text_x - 5, text_y - 3), (text_x + text_width + 5, text_y + text_height + 3)], 
                         fill=(255, 255, 255, 200))
            
            # Draw text
            if font:
                draw.text((text_x, text_y), tile_text, fill=(0, 0, 0), font=font)
            else:
                draw.text((text_x, text_y), tile_text, fill=(0, 0, 0))
    
    # Draw overlap regions if there's overlap
    if overlap > 0:
        # Create semi-transparent overlay for overlap areas
        overlay = Image.new('RGBA', image.size, (0, 0, 0, 0))
        overlay_draw = ImageDraw.Draw(overlay)
        
        # Vertical overlap lines
        for x in range(step_size, width, step_size):
            if x < width:
                overlap_start = max(0, x - overlap // 2)
                overlap_end = min(width, x + overlap // 2)
                overlay_draw.rectangle([(overlap_start, 0), (overlap_end, height)], 
                                     fill=(255, 255, 0, 50))
        
        # Horizontal overlap lines
        for y in range(step_size, height, step_size):
            if y < height:
                overlap_start = max(0, y - overlap // 2)
                overlap_end = min(height, y + overlap // 2)
                overlay_draw.rectangle([(0, overlap_start), (width, overlap_end)], 
                                     fill=(255, 255, 0, 50))
        
        # Composite the overlay
        overlay_image = Image.alpha_composite(overlay_image.convert('RGBA'), overlay).convert('RGB')
    
    return overlay_image


def create_image_tiles(image, tile_size=640, overlap=64, save_debug_tiles=True, debug_folder="debug_tiles"):
    """
    Create tiles from a large image with specified overlap.
    
    Args:
        image: PIL Image object
        tile_size: Size of each tile (default 640x640)
        overlap: Overlap between tiles in pixels (default 64)
        save_debug_tiles: Whether to save individual tile images for debugging
        debug_folder: Folder name to save debug tiles
    
    Returns:
        List of tuples: (tile_image, x_offset, y_offset, tile_width, tile_height)
    """
    width, height = image.size
    step_size = tile_size - overlap
    tiles = []
    
    # Create debug folder if saving debug tiles
    if save_debug_tiles:
        if os.path.exists(debug_folder):
            shutil.rmtree(debug_folder)  # Clean up existing folder
        os.makedirs(debug_folder, exist_ok=True)
        
        # Save original image info
        info_file = os.path.join(debug_folder, "tile_info.txt")
        with open(info_file, 'w') as f:
            f.write(f"Original image size: {width} x {height}\n")
            f.write(f"Tile size: {tile_size} x {tile_size}\n")
            f.write(f"Overlap: {overlap} pixels\n")
            f.write(f"Step size: {step_size} pixels\n")
            f.write(f"Estimated tiles: {math.ceil(width / step_size)} x {math.ceil(height / step_size)}\n\n")
    
    tile_count = 0
    for y in range(0, height, step_size):
        for x in range(0, width, step_size):
            # Calculate tile boundaries
            x_end = min(x + tile_size, width)
            y_end = min(y + tile_size, height)
            
            # Adjust starting position if we're at the edge
            x_start = max(0, x_end - tile_size)
            y_start = max(0, y_end - tile_size)
            
            # Create tile
            tile = image.crop((x_start, y_start, x_end, y_end))
            
            # Save debug tile if requested
            if save_debug_tiles:
                tile_filename = f"tile_{tile_count:03d}_x{x_start}-{x_end}_y{y_start}-{y_end}_{x_end-x_start}x{y_end-y_start}.png"
                tile_path = os.path.join(debug_folder, tile_filename)
                tile.save(tile_path, "PNG", optimize=True)
                
                # Append tile info to the info file
                with open(info_file, 'a') as f:
                    f.write(f"Tile {tile_count}: {tile_filename}\n")
                    f.write(f"  Position: ({x_start}, {y_start}) to ({x_end}, {y_end})\n")
                    f.write(f"  Size: {x_end-x_start} x {y_end-y_start}\n\n")
            
            # Store tile with its position information
            tiles.append((tile, x_start, y_start, x_end - x_start, y_end - y_start))
            tile_count += 1
    
    if save_debug_tiles:
        print(f"Debug: Saved {tile_count} tiles to '{debug_folder}' folder")
        st.sidebar.info(f"🐛 Debug: Saved {tile_count} tiles to '{debug_folder}' folder")
    
    return tiles


def process_single_tile(model_path, tile_data, conf, iou, device, tile_size):
    """
    Process a single tile for object detection (designed for parallel execution).
    
    Args:
        model_path: Path to the YOLO model (to avoid sharing model objects across processes)
        tile_data: Tuple of (tile_image, x_offset, y_offset, tile_width, tile_height, tile_id)
        conf: Confidence threshold
        iou: IoU threshold
        device: Device to run inference on
        tile_size: Size of each tile
    
    Returns:
        List of detections for this tile in original image coordinates
    """
    tile, x_offset, y_offset, tile_width, tile_height, tile_id = tile_data
    
    # Load model for this worker (each thread gets its own model instance)
    from ultralytics import YOLO
    worker_model = YOLO(model_path)
    if device == 'cuda':
        worker_model.to(device)
    
    detections = []
    
    try:
        # Run detection on tile using actual tile dimensions to avoid letterboxing
        results = worker_model.predict(
            tile,
            conf=conf,
            iou=iou,
            device=device,
            verbose=False,
            imgsz=(tile_width, tile_height),
            half=device == 'cuda'
        )
        
        # Convert detections to original image coordinates
        if results[0].boxes is not None:
            for box in results[0].boxes:
                # Get box coordinates relative to tile
                xyxy = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = xyxy
                
                # Convert to original image coordinates
                x1_orig = x1 + x_offset
                y1_orig = y1 + y_offset
                x2_orig = x2 + x_offset
                y2_orig = y2 + y_offset
                
                # Create detection object with original coordinates
                detection = {
                    'xyxy': [x1_orig, y1_orig, x2_orig, y2_orig],
                    'conf': float(box.conf[0]),
                    'cls': int(box.cls[0]),
                    'tile_id': tile_id
                }
                detections.append(detection)
    
    except Exception as e:
        print(f"Error processing tile {tile_id}: {e}")
    
    return detections


def detect_objects_tiled_parallel(model, image, tile_size=640, overlap=64, conf=0.4, iou=0.45, device='cpu', max_workers=None, save_debug_tiles=True):
    """
    Perform parallel object detection on a large image using tiling approach.
    
    Args:
        model: YOLO model
        image: PIL Image object
        tile_size: Size of each tile
        overlap: Overlap between tiles
        conf: Confidence threshold
        iou: IoU threshold for NMS
        device: Device to run inference on
        max_workers: Maximum number of parallel workers (None for auto-detection)
        save_debug_tiles: Whether to save individual tile images for debugging
    
    Returns:
        Generator yielding (progress, all_detections)
    """
    # Create tiles
    tiles = create_image_tiles(image, tile_size, overlap, save_debug_tiles=save_debug_tiles)
    total_tiles = len(tiles)
    
    # Prepare tile data with IDs
    tile_data_list = []
    for i, (tile, x_offset, y_offset, tile_width, tile_height) in enumerate(tiles):
        tile_data_list.append((tile, x_offset, y_offset, tile_width, tile_height, i))
    
    # Get model path for worker processes
    model_path = getattr(model, 'ckpt_path', 'best.pt')
    
    # Determine optimal number of workers
    if max_workers is None:
        if device == 'cuda':
            # For GPU, limit workers to avoid memory issues
            max_workers = min(4, len(tiles))
        else:
            # For CPU, use more workers
            import os
            max_workers = min(os.cpu_count() or 4, len(tiles))
    
    all_detections = []
    completed_tiles = 0
    
    # Use ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tiles for processing
        future_to_tile = {
            executor.submit(process_single_tile, model_path, tile_data, conf, iou, device, tile_size): tile_data[5]
            for tile_data in tile_data_list
        }
        
        # Process completed tiles as they finish
        for future in as_completed(future_to_tile):
            tile_id = future_to_tile[future]
            try:
                tile_detections = future.result()
                all_detections.extend(tile_detections)
                completed_tiles += 1
                
                # Yield progress update
                progress = completed_tiles / total_tiles
                yield progress, all_detections
                
            except Exception as e:
                print(f"Tile {tile_id} generated an exception: {e}")
                completed_tiles += 1
                progress = completed_tiles / total_tiles
                yield progress, all_detections
    
    # Final result
    yield 1.0, all_detections


def detect_objects_tiled(model, image, tile_size=640, overlap=64, conf=0.4, iou=0.45, device='cpu', save_debug_tiles=True):
    """
    Perform object detection on a large image using tiling approach (sequential version for fallback).
    
    Args:
        model: YOLO model
        image: PIL Image object
        tile_size: Size of each tile
        overlap: Overlap between tiles
        conf: Confidence threshold
        iou: IoU threshold for NMS
        device: Device to run inference on
        save_debug_tiles: Whether to save individual tile images for debugging
    
    Returns:
        List of detection boxes in original image coordinates
    """
    # Create tiles
    tiles = create_image_tiles(image, tile_size, overlap, save_debug_tiles=save_debug_tiles)
    all_detections = []
    
    # Process each tile
    for i, (tile, x_offset, y_offset, tile_width, tile_height) in enumerate(tiles):
        # Run detection on tile using actual tile dimensions to avoid letterboxing
        results = model.predict(
            tile,
            conf=conf,
            iou=iou,
            device=device,
            verbose=False,
            imgsz=(tile_width, tile_height),
            half=device == 'cuda'
        )
        
        # Convert detections to original image coordinates
        if results[0].boxes is not None:
            for box in results[0].boxes:
                # Get box coordinates relative to tile
                xyxy = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = xyxy
                
                # Convert to original image coordinates
                x1_orig = x1 + x_offset
                y1_orig = y1 + y_offset
                x2_orig = x2 + x_offset
                y2_orig = y2 + y_offset
                
                # Create detection object with original coordinates
                detection = {
                    'xyxy': [x1_orig, y1_orig, x2_orig, y2_orig],
                    'conf': float(box.conf[0]),
                    'cls': int(box.cls[0]),
                    'tile_id': i
                }
                all_detections.append(detection)
        
        # Update progress
        progress = (i + 1) / len(tiles)
        yield progress, all_detections
    
    # Final result
    yield 1.0, all_detections


def apply_global_nms(detections, iou_threshold=0.45):
    """
    Apply Non-Maximum Suppression across all tiles to remove duplicate detections.
    
    Args:
        detections: List of detection dictionaries
        iou_threshold: IoU threshold for NMS
    
    Returns:
        List of filtered detections
    """
    if not detections:
        return []
    
    # Convert to numpy arrays for easier processing
    boxes = np.array([det['xyxy'] for det in detections])
    scores = np.array([det['conf'] for det in detections])
    classes = np.array([det['cls'] for det in detections])
    
    # Apply NMS for each class separately
    keep_indices = []
    unique_classes = np.unique(classes)
    
    for cls in unique_classes:
        cls_mask = classes == cls
        cls_boxes = boxes[cls_mask]
        cls_scores = scores[cls_mask]
        cls_indices = np.where(cls_mask)[0]
        
        # Calculate IoU matrix
        if len(cls_boxes) > 1:
            cls_keep = nms_python(cls_boxes, cls_scores, iou_threshold)
            keep_indices.extend(cls_indices[cls_keep])
        else:
            keep_indices.extend(cls_indices)
    
    # Return filtered detections
    return [detections[i] for i in keep_indices]


def nms_python(boxes, scores, iou_threshold):
    """
    Pure Python implementation of Non-Maximum Suppression.
    
    Args:
        boxes: numpy array of shape (N, 4) containing bounding boxes
        scores: numpy array of shape (N,) containing confidence scores
        iou_threshold: IoU threshold for suppression
    
    Returns:
        List of indices to keep
    """
    # Calculate areas
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    
    # Sort by confidence scores in descending order
    order = scores.argsort()[::-1]
    
    keep = []
    while order.size > 0:
        # Keep the box with highest confidence
        i = order[0]
        keep.append(i)
        
        # Calculate IoU with remaining boxes
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        intersection = w * h
        
        union = areas[i] + areas[order[1:]] - intersection
        iou = intersection / union
        
        # Keep boxes with IoU less than threshold
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]
    
    return keep


def convert_detections_to_yolo_format(detections, model):
    """
    Convert our detection format back to YOLO-compatible format for annotation.
    """
    if not detections:
        return []
    
    # Create mock YOLO detection objects
    class MockBox:
        def __init__(self, xyxy, conf, cls):
            self.xyxy = [torch.tensor(xyxy, dtype=torch.float32)]
            self.conf = [torch.tensor([conf], dtype=torch.float32)]
            self.cls = torch.tensor(cls, dtype=torch.long)
    
    try:
        import torch
        mock_boxes = []
        for det in detections:
            box = MockBox(det['xyxy'], det['conf'], det['cls'])
            mock_boxes.append(box)
        return mock_boxes
    except ImportError:
        # Fallback if torch is not available
        return []


def main():
    """
    Main function for the Streamlit app.
    """
    setting.configure_page()

    # Creating sidebar
    with st.sidebar:
        st.header("Image Configuration")     # Adding header to sidebar
        # Adding file uploader to sidebar for selecting images
        source_img = st.sidebar.file_uploader(
            "Choose an image...", type=("jpg", "jpeg", "png"))

        # Model Options
        st.subheader("Detection Settings")
        
        # Model selection (keeping structure for future expansion)
        model_option = st.selectbox(
            "Detection Model",
            ["YOLOv8 (Current)"],
            index=0,
            help="Detection model: YOLOv8 with your custom floor plan classes"
        )
        
        confidence = setting.get_model_confidence()
        
        # Get current image size if available (needed for tiling calculations)
        current_image_size = None
        if source_img:
            # Get actual image dimensions from uploaded file
            # Reset file pointer to beginning for safe reading
            source_img.seek(0)
            temp_img = PIL.Image.open(source_img)
            current_image_size = temp_img.size
            # Reset file pointer again for later use
            source_img.seek(0)
        
        # Advanced detection options
        with st.expander("🔧 Advanced Settings"):
            iou_threshold = st.slider(
                "IoU Threshold", 0.1, 0.9, 0.45, 0.05,
                help="Non-Maximum Suppression threshold - higher values reduce duplicate detections"
            )
            show_confidence = st.checkbox(
                "Show Confidence Scores", 
                value=True,
                help="Display confidence scores in labels (e.g., 'Door (0.85)')"
            )
        
        # Initialize default values for tiling and parallel processing
        use_tiling = True
        tile_size = 640
        tile_overlap = 64
        use_parallel = True
        max_workers = 1
        save_debug_tiles = True
        
        # Tiling options for large images
        with st.expander("🏗️ Image Tiling (Recommended for Large Images)"):
            use_tiling = st.checkbox(
                "Enable Image Tiling", 
                value=True,
                help="Split large images into smaller tiles for better YOLO performance. Recommended for images larger than 2000x2000 pixels."
            )
            
            save_debug_tiles = st.checkbox(
                "Save Debug Tiles", 
                value=True,
                help="Save individual tile images to 'debug_tiles' folder for debugging purposes. Helps visualize how the image is being split."
            )
            
            if use_tiling:
                tile_size = st.selectbox(
                    "Tile Size", 
                    [256, 384, 512, 640, 768, 1024, 2048],
                    index=3,  # Default to 640
                    help="Size of each tile. 640px matches model training resolution for optimal accuracy. Smaller = faster, Larger = better for large objects"
                )
                
                tile_overlap = st.slider(
                    "Tile Overlap (pixels)", 
                    32, 256, 64, 32,
                    help="Overlap between adjacent tiles. More overlap = better edge detection but slower processing"
                )
                
                # Parallel processing options
                st.subheader("🚀 Parallel Processing")
                use_parallel = st.checkbox(
                    "Enable Parallel Processing", 
                    value=True,
                    help="Process multiple tiles simultaneously for faster inference. Recommended for multi-core systems."
                )
                
                if use_parallel:
                    # Auto-detect optimal workers or let user override
                    import os
                    cpu_count = os.cpu_count() or 4
                    gpu_available = torch.cuda.is_available() if 'torch' in globals() else False
                    
                    if gpu_available:
                        default_workers = min(4, cpu_count)  # Conservative for GPU to avoid memory issues
                        max_recommended = 8
                    else:
                        default_workers = min(cpu_count, 8)  # More aggressive for CPU
                        max_recommended = cpu_count * 2 if cpu_count else 8
                    
                    max_workers = st.slider(
                        "Parallel Workers", 
                        1, max_recommended, default_workers,
                        help=f"Number of parallel workers. Auto-detected: {default_workers} (CPU cores: {cpu_count}, GPU: {'Yes' if gpu_available else 'No'})"
                    )
                else:
                    max_workers = 1
                
                # Calculate estimated tiles for current image
                if source_img and current_image_size:
                    width, height = current_image_size
                    step_size = tile_size - tile_overlap
                    num_tiles_x = math.ceil(width / step_size)
                    num_tiles_y = math.ceil(height / step_size)
                    total_tiles = num_tiles_x * num_tiles_y
                    
                    if total_tiles > 1:
                        st.info(f"📊 Image will be split into {total_tiles} tiles ({num_tiles_x}×{num_tiles_y})")
                        
                        # Show processing time estimate
                        base_time_per_tile = 0.2  # Rough estimate: 0.2s per tile
                        if use_parallel and max_workers > 1:
                            # Account for parallel processing speedup
                            parallel_efficiency = 0.7  # Account for overhead
                            effective_workers = min(max_workers, total_tiles)
                            estimated_time = (total_tiles * base_time_per_tile) / (effective_workers * parallel_efficiency)
                            processing_note = f" (parallel with {effective_workers} workers)"
                        else:
                            estimated_time = total_tiles * base_time_per_tile
                            processing_note = " (sequential)"
                        
                        if estimated_time > 60:
                            st.warning(f"⏱️ Estimated processing time: {estimated_time/60:.1f} minutes{processing_note}")
                        else:
                            st.info(f"⏱️ Estimated processing time: {estimated_time:.1f} seconds{processing_note}")
                    else:
                        st.info("📊 Image is small enough for single-tile processing")
            else:
                tile_overlap = 0
                use_parallel = False  # No parallel processing when tiling is disabled
                max_workers = 1
                save_debug_tiles = False  # No debug tiles when tiling is disabled
        
        # Initialize session state for dynamic scaling
        if 'dynamic_font_calculated' not in st.session_state:
            st.session_state.dynamic_font_calculated = False
        if 'last_image_size' not in st.session_state:
            st.session_state.last_image_size = None
        if 'calculated_font_size' not in st.session_state:
            st.session_state.calculated_font_size = 20
        
        # Initialize session state for detection results
        if 'detection_results' not in st.session_state:
            st.session_state.detection_results = None
        if 'original_image' not in st.session_state:
            st.session_state.original_image = None
        if 'last_confidence' not in st.session_state:
            st.session_state.last_confidence = None
        if 'last_selected_labels' not in st.session_state:
            st.session_state.last_selected_labels = None
            
        # Dynamic scaling option
        use_dynamic_scaling = st.checkbox("Use Dynamic Text Scaling", value=True, 
                                        help="Automatically scale text size based on actual image dimensions (100% view)")
        
        # Check if we need to recalculate dynamic font size
        should_calculate_dynamic = (use_dynamic_scaling and 
                                  current_image_size and 
                                  (not st.session_state.dynamic_font_calculated or 
                                   st.session_state.last_image_size != current_image_size))
        
        if should_calculate_dynamic:
            # Calculate dynamic font size based on actual image dimensions
            base_size = 20  # Base reference size
            dynamic_size = setting.calculate_dynamic_font_size(current_image_size[0], current_image_size[1], base_size)
            st.session_state.calculated_font_size = dynamic_size
            st.session_state.last_image_size = current_image_size
            st.session_state.dynamic_font_calculated = True
            
        # Show current calculated size and allow manual override
        if use_dynamic_scaling and current_image_size:
            st.info(f"Dynamic font size for {current_image_size[0]}×{current_image_size[1]} image: {st.session_state.calculated_font_size}px")
            
        # Get font size from slider (will use calculated value as default if dynamic scaling was used)
        if use_dynamic_scaling and st.session_state.dynamic_font_calculated:
            # Use calculated size as the slider value, but allow user to override
            label_font_size = setting.get_label_font_size_with_default(st.session_state.calculated_font_size)
        else:
            label_font_size = setting.get_label_font_size()
            
        # If user changed the slider value, disable dynamic calculation for this session
        if (use_dynamic_scaling and st.session_state.dynamic_font_calculated and 
            label_font_size != st.session_state.calculated_font_size):
            st.session_state.dynamic_font_calculated = False
            st.caption("Manual override - dynamic scaling disabled for current session")

        # Label selection for floor plan objects
        available_labels = ['Column', 'Curtain Wall', 'Dimension', 'Door', 'Railing', 'Sliding Door', 'Stair Case', 'Wall', 'Window']
        selected_labels = setting.select_labels(available_labels)

    # Creating main page heading
    st.title("Floor Plan Object Detection using YOLOv8")

    # Creating columns on the main page
    if source_img and use_tiling:
        # Show three columns when tiling is enabled: original, tile overlay, detection results
        col1, col2, col3 = st.columns(3)
    else:
        # Show two columns for standard mode: original and detection results
        col1, col2 = st.columns(2)
        col3 = None

    # Adding image to the first column if image is uploaded
    with col1:
        if source_img:
            # Opening the uploaded image with quality preservation
            uploaded_image = load_image_with_quality_preservation(source_img)
            
            # Store the original image in session state
            st.session_state.original_image = uploaded_image
            
            # Adding the uploaded image to the page with a caption
            st.image(source_img, caption="Original Image", use_column_width=True)
        else:
            st.warning("Please upload an image.")
    
    # Show tile visualization in the second column when tiling is enabled
    if col3 is not None and source_img and use_tiling:  # Three-column mode
        with col2:
            if source_img:
                # Create and display tile overlay visualization
                tile_overlay = create_tile_overlay_visualization(uploaded_image, tile_size, tile_overlap)
                st.image(tile_overlay, caption=f"Tile Overlay ({tile_size}×{tile_size}, {tile_overlap}px overlap)", use_column_width=True)
                
                # Add tile information
                width, height = uploaded_image.size
                step_size = tile_size - tile_overlap
                num_tiles_x = math.ceil(width / step_size)
                num_tiles_y = math.ceil(height / step_size)
                total_tiles = num_tiles_x * num_tiles_y
                
                st.info(f"""
                📊 **Tile Configuration:**
                - Grid: {num_tiles_x} × {num_tiles_y} = {total_tiles} tiles
                - Tile size: {tile_size}×{tile_size} pixels
                - Overlap: {tile_overlap} pixels
                - Step size: {step_size} pixels
                """)
            else:
                st.info("Upload an image to see tile visualization")

    # Load model with explicit GPU usage for better performance
    model = YOLO('best.pt')
    
    # Explicitly move model to GPU if available for faster inference
    if torch.cuda.is_available():
        model.to('cuda')
        st.sidebar.success(f"🚀 GPU Acceleration: ENABLED ({torch.cuda.get_device_name(0)})")
    else:
        st.sidebar.warning("⚠️ GPU Acceleration: DISABLED (CPU inference)")
    
    # Display model information
    st.sidebar.info(f"**Selected Model:** {model_option}\n\n🔄 Custom trained YOLOv8 for floor plan detection")
    
    # Check if we need to run new detection (considering new parameters)
    need_new_detection = (st.session_state.detection_results is None or
                         st.session_state.last_confidence != confidence or
                         st.session_state.last_selected_labels != selected_labels or
                         getattr(st.session_state, 'last_iou_threshold', None) != iou_threshold or
                         getattr(st.session_state, 'last_use_tiling', None) != use_tiling or
                         getattr(st.session_state, 'last_tile_size', None) != tile_size or
                         getattr(st.session_state, 'last_tile_overlap', None) != tile_overlap or
                         getattr(st.session_state, 'last_use_parallel', None) != use_parallel or
                         getattr(st.session_state, 'last_max_workers', None) != max_workers or
                         getattr(st.session_state, 'last_save_debug_tiles', None) != save_debug_tiles)
    
    # Check if we need to re-render annotations (confidence display change doesn't require re-detection)
    need_rerender = (need_new_detection or 
                    getattr(st.session_state, 'last_show_confidence', None) != show_confidence)

    # Run detection only when needed (button click or parameters changed)
    if st.sidebar.button('Detect Objects') and source_img:
        if not source_img:
            st.warning("Please upload an image before detecting objects.")
        else:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            
            if use_tiling and current_image_size and (current_image_size[0] > tile_size or current_image_size[1] > tile_size):
                # Use tiled detection for large images
                with st.spinner('🔍 Running tiled object detection...'):
                    import time
                    start_time = time.time()
                    
                    # Create progress bar
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    all_detections = []
                    total_tiles = 0
                    
                    # Choose parallel or sequential processing
                    if use_parallel and max_workers > 1:
                        # Use parallel processing
                        detection_generator = detect_objects_tiled_parallel(
                            model, uploaded_image, tile_size, tile_overlap, confidence, iou_threshold, device, max_workers, save_debug_tiles
                        )
                        processing_method = f"parallel ({max_workers} workers)"
                    else:
                        # Use sequential processing
                        detection_generator = detect_objects_tiled(
                            model, uploaded_image, tile_size, tile_overlap, confidence, iou_threshold, device, save_debug_tiles
                        )
                        processing_method = "sequential"
                    
                    # Run tiled detection with progress updates
                    for progress, detections in detection_generator:
                        progress_bar.progress(progress)
                        all_detections = detections
                        if progress < 1.0:
                            current_tile = int(progress * len(create_image_tiles(uploaded_image, tile_size, tile_overlap, save_debug_tiles=False)))
                            total_tiles = len(create_image_tiles(uploaded_image, tile_size, tile_overlap, save_debug_tiles=False))
                            status_text.text(f"Processing tile {current_tile}/{total_tiles}")
                    
                    status_text.text("Applying global Non-Maximum Suppression...")
                    
                    # Apply global NMS to remove duplicates across tiles
                    filtered_detections = apply_global_nms(all_detections, iou_threshold)
                    
                    # Filter by selected labels
                    filtered_detections = [det for det in filtered_detections if model.names[det['cls']] in selected_labels]
                    
                    # Convert back to YOLO format for compatibility
                    filtered_boxes = convert_detections_to_yolo_format(filtered_detections, model)
                    
                    inference_time = time.time() - start_time
                    progress_bar.empty()
                    status_text.empty()
                    
                    # Display performance metrics
                    device_used = "GPU" if torch.cuda.is_available() else "CPU"
                    st.sidebar.info(f"⚡ Tiled Inference: {inference_time:.2f}s on {device_used}")
                    st.sidebar.info(f"🧩 Processed {total_tiles} tiles ({processing_method})")
                    st.sidebar.info(f"📊 Objects detected: {len(filtered_boxes)} (after global NMS)")
                    
                    # Model-specific performance insights
                    if use_parallel and max_workers > 1:
                        speedup_estimate = min(max_workers, total_tiles) * 0.7  # Rough estimate accounting for overhead
                        st.sidebar.success(f"✨ Using parallel tiled YOLOv8: ~{speedup_estimate:.1f}x speedup potential")
                    else:
                        st.sidebar.success("✨ Using tiled YOLOv8: Optimized for large floor plans")
            else:
                # Use standard detection for small images or when tiling is disabled
                with st.spinner('🔍 Running object detection...'):
                    import time
                    start_time = time.time()
                    
                    # Enhanced prediction with user-configurable parameters for accuracy
                    res = model.predict(
                        uploaded_image, 
                        conf=confidence,
                        iou=iou_threshold,  # User-configurable IoU threshold for NMS
                        device=device,  # Explicit device
                        verbose=False,  # Reduce console output
                        half=torch.cuda.is_available()  # Use half precision on GPU for speed
                    )
                    
                    inference_time = time.time() - start_time
                    filtered_boxes = [box for box in res[0].boxes if model.names[int(box.cls)] in selected_labels] if res[0].boxes is not None else []
                    
                    # Display performance metrics
                    device_used = "GPU" if torch.cuda.is_available() else "CPU"
                    st.sidebar.info(f"⚡ Inference: {inference_time:.2f}s on {device_used}")
                    st.sidebar.info(f"📊 Objects detected: {len(filtered_boxes)}/{len(res[0].boxes) if res[0].boxes is not None else 0}")
                    
                    # Model-specific performance insights
                    st.sidebar.success("✨ Using standard YOLOv8: Optimized for floor plan elements")
            
            # Store detection results in session state
            st.session_state.detection_results = filtered_boxes
            st.session_state.last_confidence = confidence
            st.session_state.last_selected_labels = selected_labels.copy()
            st.session_state.last_iou_threshold = iou_threshold
            st.session_state.last_show_confidence = show_confidence
            st.session_state.last_use_tiling = use_tiling
            st.session_state.last_tile_size = tile_size
            st.session_state.last_tile_overlap = tile_overlap
            st.session_state.last_use_parallel = use_parallel
            st.session_state.last_max_workers = max_workers
            st.session_state.last_save_debug_tiles = save_debug_tiles

    # Display results if we have detection data
    if st.session_state.detection_results is not None and st.session_state.original_image is not None:
        # Draw annotations with current font size
        annotated_image = draw_annotations(
            st.session_state.original_image, 
            st.session_state.detection_results, 
            model, 
            label_font_size,
            show_confidence
        )
        
        # Use appropriate column for detection results
        results_column = col3 if col3 is not None else col2
        
        with results_column:
            st.image(annotated_image, caption='Detection Results', use_column_width=True)
            # Count detected objects and display counts
            object_counts = helper.count_detected_objects(model, st.session_state.detection_results)
            st.write("\n\nDetected Objects and their Counts:")
            for label, count in object_counts.items():
                st.write(f"{label}: {count}")

            # Generate and provide download link for CSV
            csv_file = helper.generate_csv(object_counts)
            st.download_button(
                label="Download CSV",
                data=csv_file,
                file_name='detected_objects.csv',
                mime='text/csv'
            )

if __name__ == "__main__":
    main()
