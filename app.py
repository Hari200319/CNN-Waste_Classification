import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image

url = "https://drive.google.com/file/d/1z_ngAxR-_82N5UvT2vXwy6UL5ig7iqZN/view?usp=drive_link"

model_path = "waste_management_cnn.h5"


if not os.path.exists(model_path):
    gdown.download(url, model_path, quiet=False)


model = load_model(model_path)



class_labels = ["Organic", "Recyclable"]  


st.title("Waste Classification Using CNN")
st.write("Upload an image to classify it as **Organic** or **Recyclable**.")


uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
   
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    
    img = image.resize((150, 150))  
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0  

    
    prediction = model.predict(img)
    class_idx = np.argmax(prediction)  
    predicted_label = class_labels[class_idx]  
    
    st.write(f"**Prediction:** {predicted_label}")
    st.success(f"The model classifies this as: **{predicted_label}**")
