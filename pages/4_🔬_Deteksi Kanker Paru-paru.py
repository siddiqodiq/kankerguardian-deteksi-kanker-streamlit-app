import pickle
import requests
import streamlit as st
import numpy as np
import pyparsing
import pandas as pd
import matplotlib as plt
import seaborn as sns
from PIL import Image 
import tensorflow
from tensorflow.keras.models import load_model
import tensorflow as tf
from tempfile import NamedTemporaryFile
from tensorflow.keras.preprocessing import image 
from streamlit_option_menu import option_menu
from streamlit_lottie import st_lottie
st.set_page_config(page_title='Lung Cancer Detection')

def loading_model():
    fp = "models/keras_model.h5"
    model_loader = load_model(fp)
    return model_loader

cnn = loading_model()
st.write("""
    # Deteksi Kanker Paru-Paru dengan CT-Scan dada
    """)

temp = st.file_uploader("Upload Gambar CT-Scan", type = ['png', 'jpeg', 'jpg'])
if temp is not None:
    file_details = {
        "FileName": temp.name,
        "FileType": temp.type,
        "FileSize": temp.size
    }
    # Convert file size to kilobytes
    file_details["FileSizeKB"] = file_details["FileSize"] / 1024

buffer = temp
temp_file = NamedTemporaryFile(delete = False)
if buffer:
    temp_file.write(buffer.getvalue())
    st.write(image.load_img(temp_file.name))

if buffer is None:
    st.text("Upload Gambar yang sesuai")

else:
    ved_img = image.load_img(temp_file.name, target_size=(224, 224))

    # Preprocessing the image
    pp_ved_img = image.img_to_array(ved_img)
    pp_ved_img = pp_ved_img/255
    pp_ved_img = np.expand_dims(pp_ved_img, axis=0)

    #predict
    hardik_preds= cnn.predict(pp_ved_img)
    print(hardik_preds[0])

    if hardik_preds[0][0]>= 0.5:
      out = ('Gambar ini {:.2%} persen kondisi normal'.format(hardik_preds[0][0]))
      st.balloons()
      st.success(out)
    
    else: 
      out = ('Gambar ini {:.2%} persen kanker paru-paru'.format(1-hardik_preds[0][0]))
      st.error(out)

    image = Image.open(temp)
    st.image(image,use_column_width=True)

hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)