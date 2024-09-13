import streamlit as st
import mediapipe as mp
import cv2
import numpy as np
import requests
from io import BytesIO

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection()

# Streamlit interface
st.title("MediaPipe Face Detection")

# Upload or provide image URL
uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])
url = st.text_input("Or enter an image URL (optional)")

if uploaded_file is not None:
    image_data = uploaded_file.read()
elif url:
    response = requests.get(url)
    image_data = BytesIO(response.content)
else:
    st.error("Please upload an image file or provide a URL.")
    st.stop()

# Read image using OpenCV
image = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Detect faces
results = face_detection.process(image_rgb)

# Draw detections
if results.detections:
    for detection in results.detections:
        bboxC = detection.location_data.relative_bounding_box
        ih, iw, _ = image.shape
        x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Convert image to RGB for Streamlit
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
st.image(image_rgb, caption="Detected Faces", use_column_width=True)
