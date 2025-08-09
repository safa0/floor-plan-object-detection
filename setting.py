import streamlit as st

def configure_page():
    """
    Configure Streamlit page settings.
    """
    st.set_page_config(
        page_title="Object Detection using YOLOv8",  # Setting page title
        page_icon="🏡",     # Setting page icon
        layout="wide",      # Setting layout to wide
        initial_sidebar_state="expanded"    # Expanding sidebar by default
    )

def get_model_confidence():
    """
    Get model confidence from user input.
    """
    confidence = float(st.slider(
        "Select Model Confidence", 10, 100, 40,
        help="Higher values = more certain detections (fewer false positives), Lower values = catch more objects (more false positives)")) / 100
    return confidence

def get_label_font_size():
    """
    Get annotation label text size from user input.
    """
    label_font_size = int(st.slider(
        "Annotation Label Text Size (px)", 10, 100, 20))
    return label_font_size

def get_label_font_size_with_default(default_value):
    """
    Get annotation label text size from user input with a custom default value.
    """
    # Ensure default value is within slider bounds
    default_value = max(10, min(default_value, 100))
    label_font_size = int(st.slider(
        "Annotation Label Text Size (px)", 10, 100, default_value,
        help="Dynamically calculated value shown - adjust as needed"))
    return label_font_size

def calculate_dynamic_font_size(image_width, image_height, base_size=20):
    """
    Calculate dynamic font size based on image dimensions.
    Uses image diagonal to scale appropriately for both small and large images.
    """
    # Calculate image diagonal in pixels
    diagonal = (image_width ** 2 + image_height ** 2) ** 0.5
    
    # Reference diagonal (for a ~1000x1000 image)
    reference_diagonal = 1414  # sqrt(1000^2 + 1000^2)
    
    # Scale factor based on image size
    scale_factor = diagonal / reference_diagonal
    
    # Calculate dynamic font size with reasonable bounds
    dynamic_size = int(base_size * scale_factor)
    
    # Ensure font size is within reasonable bounds
    return max(10, min(dynamic_size, 200))

def select_labels(available_labels):
    """
    Select labels from available options.
    """
    selected_labels = st.multiselect(
        "Select Labels",
        available_labels
    )
    if not selected_labels:
        selected_labels = available_labels
    return selected_labels
