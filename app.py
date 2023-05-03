#importing neccesary libraries
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
import numpy as np
from PIL import Image

#actual sign language on sidebar
if st.sidebar.button(':red[Click here to show  actual digit sign language]'):
    st.sidebar.image(r"desc.png", use_column_width = True)       

    #loading already trained model
model = load_model('https://github.com/OmololaOkebiorun/Sign_language_classification/blob/main/digit_sign_language_class.h5')

#header and subheader to describe app
st.header(":red[Sign Language (Digits) Classification]")
st.subheader("Sign languages (also known as signed languages) are languages that use the visual-manual modality to convey meaning, instead of spoken words.")
st.subheader("This app will predict the digit the sign language depicts.")

#upload image to be predicted
file = st.file_uploader(":red[Upload sign language image]", type=["jpg", "png"])

#to read file as string:
if file is not None:
    
    file_bytes = np.asarray(bytearray(file.read()), dtype = np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #converts image to grayscale
    img = cv2.resize(img, (64, 64)) #resizing image
    st.image(img) #displays image to be predicted
       
#function to predict class of image
def predict_class(img_array):
    digit_class = np.argmax(model.predict(img.reshape(1, 64, 64,1)))
    return digit_class


if st.button('Predict class'):
    st.subheader(f"{predict_class(img)}")


            
