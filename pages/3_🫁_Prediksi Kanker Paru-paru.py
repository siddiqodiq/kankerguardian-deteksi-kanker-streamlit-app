import pickle
import requests
import streamlit as st
import numpy as np
import pyparsing
import pandas as pd
import matplotlib as plt
import seaborn as sns
from PIL import Image 
from streamlit_option_menu import option_menu
from streamlit_lottie import st_lottie
st.set_page_config(page_title='Lung Cancer Detection')

cancer_model = pickle.load(open('models/prediksi.sav', 'rb'))

  ##this code is reduce effort of manuallu entering values
testx=pd.read_csv("datasets/testx.csv",index_col=0)
testy=pd.read_csv("datasets/testy.csv",index_col=0)
testx.reset_index(drop=True, inplace=True)
testy.reset_index(drop=True, inplace=True)
    
concate_data = pd.concat([testx,testy],axis=1)
st.title('Prediksi Kanker Paru-Paru')
st.write("Geser slider pada setiap bagian dengan rentang 1-10 untuk memprediksi apakah kamu berpotensi mengalami kanker paru-paru. Nilai 1-10 berarti seberapa sering kamu mengalami atau menerima keadaan tersebut. Untuk Gender 1=Male dan 2=Female")

col1, col2, col3 = st.columns(3)
gender_mapping = {"Male": 1, "Female": 2}
with col1:
    Age = st.slider('Umur', min_value=0, max_value=120, key="1")

with col2:
    Gender = st.slider('jenis kelamin, pria =1 | wanita= 2', min_value=0, max_value=2, key="2")
with col3:
  AirPollution = st.slider('Polusi Udara', min_value=0, max_value=10, key="3")

with col1:
  Alcoholuse = st.slider('Minum Miras/Alkohol', min_value=0, max_value=10, key="4")

with col2:
  BalancedDiet = st.slider('Diet/nutrisi seimbang', min_value=0, max_value=10, key="5")

with col3:
  Obesity = st.slider('Obesitas (BB berlebih)', min_value=0, max_value=10, key="6")

with col1:
  Smoking = st.slider('Merokok', min_value=0, max_value=10, key="7")
with col2:
  PassiveSmoker = st.slider('Perokok Pasif', min_value=0, max_value=10, key="8")

with col3:
  Fatigue = st.slider('Mudah Lelah', min_value=0, max_value=10, key="9")

with col1:
  WeightLoss = st.slider('Berat Badan Turun', min_value=0, max_value=10, key="10")

with col2:
  ShortnessofBreath = st.slider('Susah Bernafas', min_value=0, max_value=10, key="11")
with col3:
  Wheezing = st.slider('Nafas Mengi', min_value=0, max_value=10, key="12")

with col1:
  SwallowingDifficulty = st.slider('Susah Bernafas', min_value=0, max_value=10, key="13")

with col2:
  ClubbingofFingerNails = st.slider('Jari tangan membengkak (clubbing)', min_value=0, max_value=10, key="14")

with col3:
  FrequentCold = st.slider('Sering Kedinginan', min_value=0, max_value=10, key="15")
with col1:
  DryCough = st.slider('Batuk Kering', min_value=0, max_value=10, key="16")

with col2:
  Snoring = st.slider('Mendengkur', min_value=0, max_value=10, key="17")


heart_diagnosis = ''

if st.button('Cek Hasil'):
    # Collect input values into a list
    input_data = [
        Age, Gender, AirPollution, Alcoholuse, BalancedDiet, Obesity, Smoking,
        PassiveSmoker, Fatigue, WeightLoss, ShortnessofBreath, Wheezing,
        SwallowingDifficulty, ClubbingofFingerNails, FrequentCold, DryCough, Snoring
    ]

    # Make prediction using the model
    heart_prediction = cancer_model.predict([input_data])

    if heart_prediction[0] == 'High':
        heart_diagnosis = 'Kamu masuk kriteria penderita kanker paru-paru'
        st.error(heart_diagnosis)

    elif heart_prediction[0] == 'Medium':
        heart_diagnosis = 'kamu masuk kriteria berisiko menderita kanker paru-paru'
        st.warning(heart_diagnosis)
    else:
        heart_diagnosis = 'Kamu masuk kriteria normal'
        st.balloons()
        st.success(heart_diagnosis)

expander = st.expander("Beberapa data random dari tabel test")
expander.write(concate_data.head(16))