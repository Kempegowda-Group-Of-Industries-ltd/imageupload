import streamlit as st
import mediapipe as mp
import cv2
import numpy as np
from io import BytesIO

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
face_detection = mp_face_detection.FaceDetection()

def detect_faces(image):
    # Convert the image to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Detect faces
    results = face_detection.process(image_rgb)
    
    # Draw detection results on the image
    if results.detections:
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = image.shape
            x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                         int(bboxC.width * iw), int(bboxC.height * ih)
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return image

# Streamlit app
st.title('Face Detection with MediaPipe')

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])

if uploaded_file is not None:
    # Read image
    image = np.array(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    
    # Detect faces
    image_with_faces = detect_faces(image)
    
    # Convert image for Streamlit
    st.image(image_with_faces, channels="BGR", caption="Processed Image")
