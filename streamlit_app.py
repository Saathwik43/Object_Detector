import streamlit as st
import cv2
import numpy as np
import os
import json
import time
from datetime import datetime
import tempfile
import io
from PIL import Image
import pandas as pd
import logging

# Configure page
st.set_page_config(
    page_title="Object Detection System",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
.main-header {
    text-align: center;
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    padding: 2rem;
    border-radius: 10px;
    color: white;
    margin-bottom: 2rem;
}
</style>
""", unsafe_allow_html=True)

class ObjectDetectionSystem:
    def __init__(self):
        """Initialize the object detection system"""
        self.config = {
            "confidence_threshold": 0.5,
            "nms_threshold": 0.4,
            "input_width": 608,
            "input_height": 608
        }
        self.net = None
        self.classes = []
        self.colors = []
        self.use_haar_cascade = True

        # Initialize face cascade (built into OpenCV)
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        # Basic classes for demo
        self.classes = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'face']
        self.colors = np.random.uniform(0, 255, size=(len(self.classes), 3))

    def detect_objects_haar(self, frame):
        """Detect faces using Haar cascades"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)

        detections = []
        for (x, y, w, h) in faces:
            detections.append({
                'label': 'face',
                'confidence': 0.85,
                'bbox': [x, y, w, h],
                'color': [0, 255, 0]
            })

        return detections

    def detect_objects(self, frame):
        """Main detection method"""
        return self.detect_objects_haar(frame)

    def draw_detections(self, frame, detections):
        """Draw bounding boxes and labels on frame"""
        for detection in detections:
            x, y, w, h = detection['bbox']
            label = detection['label']
            confidence = detection['confidence']
            color = detection['color']

            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

            # Draw label
            label_text = f"{label}: {confidence:.2f}"
            cv2.putText(frame, label_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return frame

    def process_image(self, image):
        """Process a single image"""
        # Convert PIL image to OpenCV format
        if isinstance(image, Image.Image):
            frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        else:
            frame = image

        # Detect objects
        detections = self.detect_objects(frame)

        # Draw detections
        result_frame = self.draw_detections(frame.copy(), detections)

        # Convert back to RGB for Streamlit
        result_frame_rgb = cv2.cvtColor(result_frame, cv2.COLOR_BGR2RGB)

        return result_frame_rgb, detections

@st.cache_resource
def load_detector():
    return ObjectDetectionSystem()

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🎯 Object Detection System</h1>
        <p>Upload images to detect faces and objects!</p>
    </div>
    """, unsafe_allow_html=True)

    # Initialize detector
    detector = load_detector()

    # Sidebar
    st.sidebar.header("🔧 Settings")
    confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.5, 0.1)
    detector.config["confidence_threshold"] = confidence_threshold

    # Main content
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=['jpg', 'jpeg', 'png', 'bmp'],
        help="Upload an image to detect objects"
    )

    if uploaded_file is not None:
        # Display original image
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Original Image")
            image = Image.open(uploaded_file)
            st.image(image, use_column_width=True)

        # Process button
        if st.button("🔍 Detect Objects", type="primary"):
            with st.spinner("Processing..."):
                result_image, detections = detector.process_image(image)

                with col2:
                    st.subheader("Detection Results")
                    st.image(result_image, use_column_width=True)

                # Statistics
                st.success(f"Found {len(detections)} objects!")

                if detections:
                    for i, detection in enumerate(detections):
                        st.write(f"Object {i+1}: {detection['label']} (confidence: {detection['confidence']:.2f})")

if __name__ == "__main__":
    main()
