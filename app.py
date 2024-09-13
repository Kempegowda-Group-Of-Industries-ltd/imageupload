import streamlit as st
import mediapipe as mp
from PIL import Image, ImageDraw
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

# Read image using PIL
image = Image.open(BytesIO(image_data))
image_rgb = np.array(image.convert('RGB'))

# Convert to RGB and process with MediaPipe
image_rgb = np.array(image)
results = face_detection.process(image_rgb)

# Draw detections manually using PIL
draw = ImageDraw.Draw(image)
if results.detections:
    ih, iw, _ = image.size
    for detection in results.detections:
        bboxC = detection.location_data.relative_bounding_box
        x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
        draw.rectangle([x, y, x + w, y + h], outline="green", width=3)

# Display the image with detected faces
st.image(image, caption="Detected Faces", use_column_width=True)
