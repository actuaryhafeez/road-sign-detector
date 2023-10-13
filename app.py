import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image

st.title('Road Sign Detection App')

# Load model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5/runs/train/exp/weights/last.pt', force_reload=True)

# Upload image
uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "png", "gif"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    image_np = np.array(image)

    # Detect
    results = model(image_np)

    # Display raw rendered output without conversion
    processed_img = np.squeeze(results.render())
    st.image(processed_img, caption='Raw Processed Image', use_column_width='auto', width=500)
