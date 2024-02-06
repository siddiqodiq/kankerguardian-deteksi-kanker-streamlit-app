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


# Define or import custom objects
gm_exp = tf.Variable(3., dtype=tf.float32)

def loading_model():
    fp = "models/model_cnn.h5"
    model_loader = load_model(fp)
    return model_loader

cnn = loading_model()
st.write("""
    # Deteksi kanker kulit dengan foto
    """)

temp = st.file_uploader("Upload foto kulit", type=['png', 'jpeg', 'jpg'])

if temp is not None:
    file_details = {
        "FileName": temp.name,
        "FileType": temp.type,
        "FileSize": temp.size
    }
    # Convert file size to kilobytes
    file_details["FileSizeKB"] = file_details["FileSize"] / 1024

    buffer = temp
    temp_file = NamedTemporaryFile(delete=False)

    if buffer:
        temp_file.write(buffer.getvalue())
        st.image(Image.open(temp_file.name), use_column_width=True)

        ved_img = image.load_img(temp_file.name, target_size=(299, 299))

        # Preprocessing the image
        pp_ved_img = image.img_to_array(ved_img)
        pp_ved_img = pp_ved_img / 255
        pp_ved_img = np.expand_dims(pp_ved_img, axis=0)

        # predict
        hardik_preds = cnn.predict(pp_ved_img)

        threshold = 0.5  # You can adjust the threshold based on your preference

        if hardik_preds[0][0] <= threshold:
            out = ('Foto masuk kriteria kanker kulit'.format(hardik_preds[0][0]))
            st.error(out)
        else:
            out = ('Foto masuk kriteria normal'.format(1 - hardik_preds[0][0]))
            st.success(out)
            st.balloons()

else:
    st.text("Upload foto yang sesuai!")