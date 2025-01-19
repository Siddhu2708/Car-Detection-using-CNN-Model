import streamlit as st
import tensorflow
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Define the classes
CLASSES = ['Audi', 'Hyundai Creta', 'Mahindra Scorpio', 'Maruti Swift',
           'Rolls Royce', 'Tata Safari', 'Toyota Innova']

# Load the pre-trained model
model = load_model ( 'car_model.h5' )

# Function to preprocess image
def preprocess_image(img):
    img = img.resize((256, 256))  # Assuming the model expects 224x224 images
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize pixel values
    return img_array

# Streamlit app
st.title("Car Image Prediction App 🚗")
st.write("Upload an image of a car to predict its class.")

# File uploader
uploaded_file = st.file_uploader("Choose a car image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display uploaded image
    st.image(uploaded_file, caption='Uploaded Image', width=400)

    # Load image
    img = Image.open(uploaded_file)
    processed_img = preprocess_image(img)

    # Predict
    prediction = model.predict(processed_img)
    predicted_class = CLASSES[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    # Display results
    st.write(f"### Predicted Class: {predicted_class}")
    st.write(f"Confidence: {confidence:.2f}%")
