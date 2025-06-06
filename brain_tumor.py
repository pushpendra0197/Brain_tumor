import numpy  as np
import pandas as pd
import streamlit as st
from tensorflow.python.keras.models import load_model
import matplotlib.pyplot as plt
import tensorflow
from PIL import Image
import base64
st.set_page_config(page_title="Brain Tumor Detector",page_icon=r"images.png")
path=(r"image.png")
with open(path,"rb") as file:
    image_back=base64.b64encode(file.read()).decode()
page_element=f"""
<style>
[data-testid="stAppViewContainer"]
{{
background-image:url("data:image;base64,{image_back}");
background-size:2000px 1000px;
background-repeat:no-repeat;
background-position:center;
}}
<style>
"""
st.markdown(page_element,unsafe_allow_html=True)
upload=st.file_uploader("Upload Image of Mri")
button=st.button("Predict")
if button:
    if upload is not None:
      image=Image.open(upload)
      resize_image=image.resize((256,256))
      final_image=np.expand_dims(resize_image,axis=0)
      Model= tensorflow.keras.models.load_model(r"Braintumor.keras")
      prediction=Model.predict(final_image)
      class_name=['Glioma', 'Healthy', 'Meningioma', 'Pituitary']
      output=class_name[np.argmax(prediction)]
      st.image(image)
      st.header(output,divider="rainbow")
