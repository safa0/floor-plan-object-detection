import streamlit as st
from ultralytics import YOLO
import PIL
from PIL import ImageDraw, ImageFont
import helper
import setting

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
        # Annotation label text size
        label_font_size = setting.get_label_font_size()

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
            # Opening the uploaded image
            uploaded_image = PIL.Image.open(source_img)
            # Adding the uploaded image to the page with a caption
            st.image(source_img,caption="Uploaded Image",use_column_width=True)
        else:
            st.warning("Please upload an image.")

    model = YOLO('best.pt')

    if st.sidebar.button('Detect Objects'):
        if not source_img:
            st.warning("Please upload an image before detecting objects.")
        else:
            res = model.predict(uploaded_image, conf=confidence)
            filtered_boxes = [box for box in res[0].boxes if model.names[int(box.cls)] in selected_labels]
            res[0].boxes = filtered_boxes

            # Manually render boxes and labels with configurable text size
            annotated_image = uploaded_image.copy()
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

            # Simple color palette and thickness scaling
            palette = [
                (255, 56, 56), (255, 157, 151), (255, 112, 31), (255, 178, 29),
                (207, 210, 49), (72, 249, 10), (146, 204, 23), (61, 219, 134),
                (26, 147, 52), (0, 212, 187), (44, 153, 168), (0, 194, 255),
                (52, 69, 147), (100, 115, 255), (142, 140, 255), (204, 182, 142),
                (255, 173, 203), (255, 0, 0), (255, 165, 0), (255, 255, 0),
                (0, 255, 0), (0, 255, 255), (0, 0, 255), (255, 0, 255)
            ]
            box_thickness = max(2, label_font_size // 8)
            pad = max(2, label_font_size // 6)

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

            with col2:
                st.image(annotated_image, caption='Detected Image', use_column_width=True)
                # Count detected objects and display counts
                object_counts = helper.count_detected_objects(model, filtered_boxes)
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
