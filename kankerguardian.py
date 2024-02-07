import requests
import streamlit as st
import plotly.figure_factory as ff
from streamlit_extras.colored_header import colored_header
import numpy as np
from streamlit_option_menu import option_menu
from streamlit_lottie import st_lottie
st.set_page_config(page_title="KankerGuardian", page_icon="images\logo.png")
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_url_kanker = "https://lottie.host/f9fb8e7c-1c04-4506-937a-52df427d9a53/xFVsKBo1Fd.json"

def example():
    colored_header(
        label="🦠KankerGuardian🦠",
        description="Kankerguardian adalah aplikasi inovatif yang menghadirkan solusi lengkap untuk edukasi, deteksi, dan prediksi kanker paru-paru. Dengan fokus pada memberikan pemahaman mendalam tentang kedua jenis kanker tersebut, aplikasi ini menjadi penjaga terpercaya dalam upaya pencegahan dan pemantauan kesehatan.",
        color_name="green-70",
    )
example()
lottie_kanker = load_lottieurl(lottie_url_kanker)
st_lottie(lottie_kanker)