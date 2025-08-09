import streamlit as st
from ultralytics import YOLO
import PIL
from PIL import Image, ImageDraw, ImageFont
import helper
import setting

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
    _orig_torch_load = torch.load
    def _torch_load_compat(*args, **kwargs):
        if "weights_only" not in kwargs:
            kwargs["weights_only"] = False
        return _orig_torch_load(*args, **kwargs)
    torch.load = _torch_load_compat
except Exception:
    st.warning("Unable to configure PyTorch safe-loading for Ultralytics model")
    print("Warning: Unable to configure PyTorch safe-loading for Ultralytics model") 


def load_image_with_quality_preservation(source_img):
    """
    Load image while preserving maximum quality, especially for PNG files.
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
    
    return img


def draw_annotations(image, filtered_boxes, model, label_font_size):
    """
    Draw annotations on image with specified font size.
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
        x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
        cls_id = int(box.cls)
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
        confidence = setting.get_model_confidence()
        
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
        
        # Get current image size if available
        current_image_size = None
        if source_img:
            # Get actual image dimensions from uploaded file
            # Reset file pointer to beginning for safe reading
            source_img.seek(0)
            temp_img = PIL.Image.open(source_img)
            current_image_size = temp_img.size
            # Reset file pointer again for later use
            source_img.seek(0)
        
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

        # Multiselect for selecting labels
        available_labels = ['Column', 'Curtain Wall', 'Dimension', 'Door', 'Railing', 'Sliding Door', 'Stair Case', 'Wall', 'Window']
        selected_labels = setting.select_labels(available_labels)

    # Creating main page heading
    st.title("Floor Plan Object Detection using YOLOv8")

    # Creating two columns on the main page
    col1, col2 = st.columns(2)

    # Adding image to the first column if image is uploaded
    with col1:
        if source_img:
            # Opening the uploaded image with quality preservation
            uploaded_image = load_image_with_quality_preservation(source_img)
            
            # Store the original image in session state
            st.session_state.original_image = uploaded_image
            
            # Adding the uploaded image to the page with a caption
            st.image(source_img, caption="Uploaded Image", use_column_width=True)
        else:
            st.warning("Please upload an image.")

    model = YOLO('best.pt')
    
    # Check if we need to run new detection
    need_new_detection = (st.session_state.detection_results is None or
                         st.session_state.last_confidence != confidence or
                         st.session_state.last_selected_labels != selected_labels)

    # Run detection only when needed (button click or parameters changed)
    if st.sidebar.button('Detect Objects') and source_img:
        if not source_img:
            st.warning("Please upload an image before detecting objects.")
        else:
            # Run new detection
            res = model.predict(uploaded_image, conf=confidence)
            filtered_boxes = [box for box in res[0].boxes if model.names[int(box.cls)] in selected_labels]
            
            # Store detection results in session state
            st.session_state.detection_results = filtered_boxes
            st.session_state.last_confidence = confidence
            st.session_state.last_selected_labels = selected_labels.copy()

    # Display results if we have detection data
    if st.session_state.detection_results is not None and st.session_state.original_image is not None:
        # Draw annotations with current font size
        annotated_image = draw_annotations(
            st.session_state.original_image, 
            st.session_state.detection_results, 
            model, 
            label_font_size
        )
        
        with col2:
            st.image(annotated_image, caption='Detected Image', use_column_width=True)
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
