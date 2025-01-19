# Importing Libraries :
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as preprocessing
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import cv2
import cvzone
import math
from ultralytics import YOLO

# Setting Icon Image :
img = Image.open("Icon.png")
st.set_page_config(page_title="Vegetable Image Detection & Classification Using CNN", page_icon=img, layout="wide")

# Hide Menu_Bar & Footer :
hide_menu_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
"""
st.markdown(hide_menu_style, unsafe_allow_html=True)

# Set the background image :
Background_image = """
<style>
[data-testid="stAppViewContainer"] > .main {
background-image: url("https://img.freepik.com/free-vector/abstract-watercolor-design_1055-7990.jpg?w=740&t=st=1709652911~exp=1709653511~hmac=d7d05c8a11f19a9bf91f7e4534342dbabf0aa64dbad0b89e51a4fdc8ce4e8f68");
background-size: 100%;
background-position: top left;
background-position: center;
background-size: cover;
background-repeat: repeat;
background-repeat: round;
background-attachment: local;
background-image: url("https://img.freepik.com/free-vector/abstract-watercolor-design_1055-7990.jpg?w=740&t=st=1709652911~exp=1709653511~hmac=d7d05c8a11f19a9bf91f7e4534342dbabf0aa64dbad0b89e51a4fdc8ce4e8f68");
background-position: right bottom;
background-repeat: no-repeat;
}  
[data-testid="stHeader"] {
background-color: rgba(0,0,0,0);
}
</style>
"""
st.markdown(Background_image, unsafe_allow_html=True)

title_html = f'<h1 style="color:#219ebc; text-align:center;font-family:Edwardian Script ITC;font-size:96px;">Vegetable Classification & Detection</h1>'
st.markdown(title_html, unsafe_allow_html=True)

# Load the image using PIL 
image_path = "C:\\Users\\lenovo\\Downloads\\Vegetable_Classification_And_Detection\\CNN Architecture.jpg" 
image = Image.open(image_path) 

# Display the image with Streamlit 
st.image(image, use_container_width=True)

# Creating Columns for Audio and Image :
col1, col2 = st.columns([1, 4])
with col1:
    st.markdown("""
                <h6 style="color:Red;font-family:Hobo Std;"> Audio File </h6>
                """, unsafe_allow_html=True)
    btn = st.button("Click")
    if btn:
        st.audio("VC.mp3")
with col2:
    st.markdown("""
                <marquee width="100%" direction="left" scrollamount="6">
                <img src="https://th.bing.com/th/id/OIP.CQSH56oBGHTHZp3rhT1K6wHaE8?pid=ImgDet&w=200&h=200&c=7&dpr=1.5" alt="VC Image 1">
                <img src="https://th.bing.com/th/id/OIP.r36GrsYEAfYo7Rq2Y-yDUQHaHa?pid=ImgDet&w=200&h=200&c=7&dpr=1.5" alt="VC Image 2">
                <img src="https://th.bing.com/th/id/OIP.ufdlWnNT41uyoRckj5FB_wHaE7?pid=ImgDet&w=200&h=200&c=7&dpr=1.5" alt="VC Image 3">
                <img src="https://th.bing.com/th/id/OIP.WSasQlY5xSx_fRFPDRKfPQHaE7?pid=ImgDet&w=200&h=200&c=7&dpr=1.5" alt="VC Image 4">
                <img src="https://th.bing.com/th/id/OIP.nUHgV9LcaPmHD7jUECGDlgHaE2?pid=ImgDet&w=200&h=200&c=7&dpr=1.5" alt="VC Image 5">
                </marquee>
                """, unsafe_allow_html=True)

# Marquee Tag - About VC :
st.markdown("""
    <marquee width="100%" direction="left" height="100px" scrollamount="6" style="color:white;font-family:Maiandra GD;">
    🚀 Disclaimer : 
            🌟 Vegetable classification is a crucial process in Agriculture and Food Processing. 
            🌟 Supervised Learning technique is used to train the model. 
            🌟 Convolutional Neural Network (CNN) is a deep learning technology that is mainly used in Image Recognition and Classification tasks. 
            🌟 Image Classification is the process of categorizing an image into a predefined classes based on the tasks. 
            🌟 The model will learn based on the pixels and the model will recognize the pattern and extract the feature and training with the predefined classes/labels. 
            🌟 Vegetables Detection employs the YOLO (You Only Look Once) model for object detection, which identifies vegetables in the live feed with bounding boxes and labels.
    </marquee>
""", unsafe_allow_html=True)

# Define vegetable names and class map
vegetable_names = ['Bean', 'Bitter_Gourd', 'Bottle_Gourd', 'Brinjal', 'Broccoli', 'Cabbage', 'Capsicum',
                   'Cauliflower', 'Cucumber', 'Potato', 'Radish', 'Tomato']
class_map = {i: veg for i, veg in enumerate(vegetable_names)}

# Load the model
model = load_model('C:\\Users\\lenovo\\Downloads\\Vegetable_Classification_And_Detection\\Save_Model\\my_model.keras')
# Define the function to generate predictions and plot the image
def generate_predictions(test_image_path, actual_label):
    # Load and preprocess the image
    test_img = load_img(test_image_path, target_size=(150, 150))  # Adjust target size
    test_img_arr = img_to_array(test_img) / 255.0
    test_img_input = np.expand_dims(test_img_arr, axis=0)

    # Make Predictions
    predictions = model.predict(test_img_input)
    predicted_label = np.argmax(predictions)
    predicted_vegetable = class_map[predicted_label]
    prediction_score = np.max(predictions) * 100

    # Display the image with the predicted and actual labels
    plt.figure(figsize=(4, 4))
    plt.imshow(test_img_arr)
    plt.title(f"Predicted Label: {predicted_vegetable}, Actual Label: {actual_label}\nConfidence Score: {prediction_score:.2f}%")
    plt.grid(False)
    plt.axis('off')
    plt.show()

    return f"Image belongs to [ {predicted_vegetable} ] with Confidence Score [ {prediction_score:.2f}% ]"

# Streamlit app main part
Options = st.selectbox("Select your choice", ["Vegetable Classification", "Vegetable Detection"])

if Options == "Vegetable Classification":
    col_1, col_2 = st.columns([5, 5])

    with col_1:
        uploaded_file = st.file_uploader('Upload an Image', type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            image_path = os.path.join("upload", uploaded_file.name)
            os.makedirs(os.path.dirname(image_path), exist_ok=True)
            with open(image_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            st.image(image, caption='Uploaded Image', use_container_width=True)
            result = generate_predictions(image_path, actual_label='Unknown')
            st.markdown(result)

    with col_2:  
        captured_image = st.camera_input("Capture an image", help="This is just a basic example")
        if captured_image is not None:
            capture_folder = "capture" 
            os.makedirs(capture_folder, exist_ok=True) # Ensure the folder exists 
            image_path = os.path.join(capture_folder, "captured_image.jpg")
            with open(image_path, 'wb') as imagefile:
                imagefile.write(captured_image.getbuffer())

            image = Image.open(image_path)
            st.image(image, caption='Captured Image', use_container_width=True)
            result = generate_predictions(image_path, actual_label='Unknown')
            st.markdown(result)

elif Options == "Vegetable Detection":
    
    # Function to perform object detection and draw bounding boxes
    def perform_object_detection(image, model, classNames):
        results = model(image, stream=True)
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0]) # Convert coordinates to integers
                x1, y1, x2, y2 = max(0, x1), max(0, y1), max(0, x2), max(0, y2)  # Ensure non-negative coordinates
                cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 255), thickness=3) # Specify thickness explicitly
                w, h = x2 - x1, y2 - y1
                cvzone.cornerRect(image, (x1, y1, w, h))
                conf = math.ceil((box.conf[0] * 100)) / 100
                cls = int(box.cls[0])
                cvzone.putTextRect(image, f'{classNames[cls]} {conf}', (max(1, x1), max(0, y1))) # Adjusted y coordinate
        return image

    # Load YOLO model
    model = YOLO("best.pt")

    # Define class names
    classNames = ['Bean', 'Bitter_Gourd', 'Bottle_Gourd', 'Brinjal', 'Broccoli', 'Cabbage', 'Capsicum',
                   'Cauliflower', 'Cucumber', 'Potato', 'Radish', 'Tomato']

    # Webcam capture function
    def capture_image(cap, is_detection_started):
        while is_detection_started:
            ret, frame = cap.read()
            if not ret:
                break

            frame = perform_object_detection(frame, model, classNames)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            yield frame

    # Main Streamlit code
    if __name__ == "__main__":
        cap = cv2.VideoCapture(0)
        cap.set(3, 1500)
        cap.set(4, 720)

        is_detection_started = False
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            start_button = st.button("Start")
        with col2:
            stop_button = st.button("Stop")

        if start_button:
            is_detection_started = True

        if stop_button:
            is_detection_started = False

        if is_detection_started:
            frames = capture_image(cap, is_detection_started)
            frame = next(frames)

            stframe = st.empty()
            stframe.image(frame, channels="RGB")

            for frame in frames:
                stframe.image(frame, channels="RGB", use_container_width=True)

        cap.release()

#Define your footer content
footer_text = """ <p><br>< Final Year Project Developed By Group No. - 16 /> <br> Under the guidance of Mr. Tapas Paul <br> (Assistant Professor) <br> Dept. Of Information Technology <br> Asansol Engineering College <br> Asansol, West Bengal, India <br> © 2025 . All Rights Reserved.</p> """

# Create a container for the footer
footer = st.container()

# Add footer content to the container
with footer:
    st.markdown(f"<div style='text-align: center; color: gray; margin-top: 20px'>{footer_text}</div>", unsafe_allow_html=True)

