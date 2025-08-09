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
        "Select Model Confidence", 10, 100, 40)) / 100
    return confidence

def get_label_font_size():
    """
    Get annotation label text size from user input.
    """
    label_font_size = int(st.slider(
        "Annotation Label Text Size (px)", 10, 30, 12))
    return label_font_size

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
