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

def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_url_lung = "https://lottie.host/206c4af2-39e9-4c1f-aa59-1e3b5f74e6bc/NquWZIUHCz.json"
lottie_lung = load_lottieurl(lottie_url_lung)
_left, mid, _right = st.columns([0.2,0.4,0.2])
with mid:
        st_lottie(
            lottie_lung
        )# page title
st.title('Apa itu kanker paru-paru?')

st.write("Kanker paru merupakan kanker yang terjadi pada sel-sel dinding saluran udara organ paru. Sel tidak normal ini membelah secara tidak terkontrol dan membentuk suatu masa padat atau yang kita kenal tumor. Seiring dengan perkembangan sel ini, fungsi normal paru dapat terganggu. Sel kanker ini juga dapat menyebar ke jaringan getah bening, organ paru lain, tulang, otak, dan hati. Kanker paru dapat berkembang dalam waktu bulan hingga menahun tanpa menunjukkan tanda dan gejala yang jelas.")
st.write("Di Indonesia, berdasarkan data Globocan 2020, jumlah kasus baru kanker paru menempati urutan ke-3 (8,8%), setelah kanker payudara (16,6%), dan kanker serviks (9,2%). Kanker paru merupakan jenis kanker yang paling banyak yang terjadi pada laki-laki (14,1%).")
st.write("")
st.title("Apakah Merokok adalah penyebabnya?")
lottie_url_smoking = "https://lottie.host/4eae6ce8-2f05-4c3a-ab4d-5d845a976c5d/thEMfD0bTr.json"
lottie_smoking = load_lottieurl(lottie_url_smoking)
_left, mid, _right = st.columns([0.2,0.4,0.2])
with mid:
        st_lottie(lottie_smoking, key = "smoking")

st.markdown(
        """
        Seperti yang sudah dijelaskan sebelumnya, kebiasaan yang dapat memicu penyakit kanker paru paru adalah merokok. Bahkan, sekitar 90% kasus kanker paru-paru pada pria dan sekitar 70% pada wanita disebabkan oleh asap rokok. Jika Anda merupakan perokok aktif yang kuat, maka semakin besar risiko untuk menderita kanker paru-paru. Hanya sebagian kecil kanker paru-paru (sekitar 10-15% pada pria dan 5% pada wanita), disebabkan oleh zat yang ditemui atau terhirup di tempat bekerja. Penyebab lainnya yaitu paparan asbes, radiasi, arsen, kromat, nikel, klorometil eter, gas mustard dan pancaran oven arang di lingkungan kerja, meskipun biasanya hanya terjadi pada pekerja yang juga merokok.

Sementara itu, paparan polusi udara sebagai penyebab kanker paru-paru, masih belum ditemukan secara pasti.  Kemudian, sejumlah pasien kanker paru-paru juga memiliki riwayat  adanya pemaparan oleh gas radon di rumah tangga. Terkadang, kanker paru (terutama adenokarsinoma) terjadi pada orang yang paru-parunya telah memiliki jaringan parut karena penyakit paru-paru lainnya, seperti tuberkulosis dan fibrosis.
        """
)