import streamlit as st
import cv2
import numpy as np
from PIL import Image
import pandas as pd

st.set_page_config(
    page_title="Object Detection System",
    page_icon="🎯",
    layout="wide"
)

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

class FaceDetector:
    def __init__(self):
        try:
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            self.initialized = True
        except Exception as e:
            st.error(f"Failed to load face detector: {e}")
            self.initialized = False
    
    def detect_faces(self, image):
        if not self.initialized:
            return []
        
        try:
            # Convert PIL to OpenCV
            frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
            
            # Draw rectangles
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, 'Face', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Convert back to RGB
            result = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return result, len(faces)
            
        except Exception as e:
            st.error(f"Detection failed: {e}")
            return np.array(image), 0

@st.cache_resource
def load_detector():
    return FaceDetector()

def main():
    st.markdown("""
    <div class="main-header">
        <h1>🎯 Face Detection System</h1>
        <p>Upload an image to detect faces!</p>
    </div>
    """, unsafe_allow_html=True)
    
    detector = load_detector()
    
    if not detector.initialized:
        st.error("Face detector failed to initialize. Please try refreshing the page.")
        return
    
    st.sidebar.header("🔧 Settings")
    st.sidebar.info("Face detection using OpenCV Haar Cascades")
    
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=['jpg', 'jpeg', 'png', 'bmp'],
        help="Upload an image to detect faces"
    )
    
    if uploaded_file is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original Image")
            image = Image.open(uploaded_file)
            st.image(image, use_column_width=True)
        
        if st.button("🔍 Detect Faces", type="primary"):
            with st.spinner("Processing..."):
                result_image, face_count = detector.detect_faces(image)
                
                with col2:
                    st.subheader("Detection Results")
                    st.image(result_image, use_column_width=True)
                
                if face_count > 0:
                    st.success(f"✅ Found {face_count} face(s)!")
                else:
                    st.info("No faces detected in this image.")

if __name__ == "__main__":
    main()
