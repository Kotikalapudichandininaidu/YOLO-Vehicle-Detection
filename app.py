import streamlit as st
import numpy as np
from PIL import Image
import cv2
from ultralytics import YOLO
import os

# --- Page Configuration ---
st.set_page_config(
    page_title="Object Detection App",
    page_icon="🔍",
    layout="centered",
    initial_sidebar_state="auto",
)

# --- Title and Description ---
st.title("Object Detection with YOLO")
st.write("Upload an image and let the model detect objects!")

# --- Model Loading ---
@st.cache_resource
def load_model():
    """
    Loads the YOLO model.
    Please ensure 'best.pt' is in the same directory as this script,
    or update the path to its correct location.
    """
    try:
        # Update this path if your 'best.pt' is in a different location
        model = YOLO("best.pt")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.info("Please make sure 'best.pt' is in the correct path.")
        return None

model = load_model()

if model:
    # --- Image Upload ---
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display uploaded image
        st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
        st.write("")
        st.write("Detecting objects...")

        # Process the image for detection
        image = Image.open(uploaded_file).convert('RGB')
        image_np = np.array(image)
        # Convert RGB to BGR for OpenCV and YOLO compatibility
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

        # --- Perform Prediction ---
        try:
            # You can adjust conf (confidence threshold) and iou (IoU threshold) as needed
            results = model.predict(source=image_bgr, save=False, imgsz=416, conf=0.5, iou=0.7)

            # Get the annotated image from results
            # The plot() method returns a BGR image with detections drawn
            annotated_image_bgr = results[0].plot()

            # Convert BGR to RGB for Streamlit display
            annotated_image_rgb = cv2.cvtColor(annotated_image_bgr, cv2.COLOR_BGR2RGB)

            st.image(annotated_image_rgb, caption="Detected Objects", use_container_width=True)

            # Display detection details (optional)
            if results[0].boxes:
                st.subheader("Detection Details:")
                for i, box in enumerate(results[0].boxes):
                    cls_id = int(box.cls)
                    conf = float(box.conf)
                    label = model.names[cls_id]
                    st.write(f"Object {i+1}: **{label}** (Confidence: {conf:.2f})")
            else:
                st.write("No objects detected.")

        except Exception as e:
            st.error(f"Error during object detection: {e}")
            st.info("Please check the model's compatibility with the image and its configuration.")
else:
    st.warning("Model could not be loaded. Please check the error message above.")

# --- Footer ---
st.sidebar.header("About")
st.sidebar.info("This app uses a YOLO model for object detection. "
                "Upload an image to see it in action.")
