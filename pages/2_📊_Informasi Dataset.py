import pickle
import requests
import streamlit as st
import numpy as np
import pyparsing
import pandas as pd
import matplotlib as plt
import seaborn as sns
from PIL import Image 
from streamlit_extras.metric_cards import style_metric_cards
from streamlit_option_menu import option_menu
from streamlit_lottie import st_lottie
st.set_page_config(page_title='Lung Cancer Detection')

def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

st.sidebar.title("Informasi Dataset📊")
tabs = ["Prediksi paru-paru", "Deteksi kanker paru-paru"]
selected_tab = st.sidebar.selectbox("Daftar dataset:", tabs)
if selected_tab == "Prediksi paru-paru":
    tab1, tab2, tab3 , tab4= st.tabs(["Analisis Dataset", "Training Data", "Test Data","Algoritma"])
    with tab1:
            
            st.header("Lung Cancer Dataset")
            data=pd.read_csv("datasets/data.csv")
            data = data.rename(columns={
        'Age': 'Usia',
        'Gender': 'Jenis Kelamin',
        'Air Pollution': 'Polusi Udara',
        'Alcohol use': 'Konsumsi Alkohol',
        'Dust Allergy': 'Alergi Debu',
        'Genetic Risk': 'Resiko Genetik',
        'chronic Lung Disease': 'Penyakit Paru-paru Kronis',    
        'Obesity': 'Obesitas',
        'OccuPational Hazards': 'Lingkungan kerja tidak sehat',
        'Smoking': 'Merokok',
        'Passive Smoker': 'Perokok Pasif',
        'Balanced Diet': 'Diet/nutrisi Seimbang',
        'Chest Pain': 'Nyeri Dada',
        'Coughing of Blood': 'Batuk berdarah',
        'Fatigue': 'Kelelahan',
        'Weight Loss': 'Penurunan Berat Badan',
        'Shortness of Breath': 'Kesulitan Bernapas',
        'Wheezing': 'Bunyi Mengi',
        'Swallowing Difficulty': 'Kesulitan Menelan',
        'Clubbing of Finger Nails': 'Clubbing Kuku Jari',
        'Frequent Cold': 'Sering Flu',
        'Dry Cough': 'Batuk Kering',
        'Snoring': 'Sering Mendengkur',
        'Level': 'Tingkat'
    })
            st.write(data.head(10))
            code = '''#daftar Kolom
Index('Usia', 'Jenis Kelamin', 'Pencemaran Udara', 'Penggunaan Alkohol', 'Alergi Debu',
'Lingkungan kerja tidak sehat', 'Risiko Genetik', 'Penyakit Paru Kronis',
'Diet Seimbang', 'Obesitas', 'Merokok', 'Perokok Pasif', 'Nyeri Dada',
'Batuk Darah', 'Kelelahan', 'Kehilangan Berat Badan', 'Sesak Nafas',
'Bersin-bersin', 'Kesulitan Menelan', 'Clubbing Kuku Jari',
'Sering Kedinginan', 'Batuk Kering', 'Mendengkur', 'Level'],
dtype='object')'''
            st.code(code, language='python')
            # Streamlit App
            # Display the total number of patients for each level using st.metric in columns
            st.write('Patient Level Summary:')
            level_counts = data['Tingkat'].value_counts()

            # Create columns for metrics
            col1, col2, col3 = st.columns(3)

            # Display metrics for each level in columns
            with col1:
                st.metric(label='Tingkat Rendah', value=level_counts.get("Low", 0))
                
            with col2:
                st.metric(label='Tingkat Menengah', value=level_counts.get("Medium", 0))

            with col3:
                st.metric(label='Tingkat Kronis', value=level_counts.get("High", 0))
            
            style_metric_cards()
            
            st.header("Pearson Correlation Matrix")
            st.write("Pearson Correlation Matrix adalah sebuah matriks yang mengukur sejauh mana hubungan linier antara dua variabel. Dalam konteks analisis data, ini sering digunakan untuk mengidentifikasi tingkat korelasi atau hubungan antara setiap pasang variabel. Pearson Correlation Matrix mengukur seberapa kuat dan arah hubungan linier antara dua variabel, dengan nilai berkisar antara -1 hingga 1.")
            coors = Image.open("images/coors.png")

            st.image(coors, caption='Pearson Correlation Matrix',width=800)
            st.write("Dari matriks korelasi di atas, diterrapkan sebuah fungsi yang memilih nilai berdasarkan korelasi tinggi dengan atribut ataua kolomtertentu yang dapat dihilangkan untuk meningkatkan kinerja model Machine Learning.")
            st.markdown( """
                
                - Kolom/atribut yang dihilangkan :-
                """)
            code = '''{'Nyeri Dada',
'Batuk Darah',
'Alergi Debu',
'Resiko Genetik',
'Lingkungan pekerjaan tidak sehat',
'Penyakit Paru-paru Kronis'}'''
            st.code(code, language='python')

    with tab2:
            st.header("Dataset Gambar CT-Scan dada")
            st.subheader("X_Train Data")
            data=pd.read_csv("datasets/train.csv", index_col=0)
            data = data.rename(columns={
        'Age': 'Usia',
        'Gender': 'Jenis Kelamin',
        'Air Pollution': 'Polusi Udara',
        'Alcohol use': 'Konsumsi Alkohol',
        'Dust Allergy': 'Alergi Debu',
        'Genetic Risk': 'Resiko Genetik',
        'chronic Lung Disease': 'Penyakit Paru-paru Kronis',    
        'Obesity': 'Obesitas',
        'OccuPational Hazards': 'Lingkungan kerja tidak sehat',
        'Smoking': 'Merokok',
        'Passive Smoker': 'Perokok Pasif',
        'Balanced Diet': 'Diet/nutrisi Seimbang',
        'Chest Pain': 'Nyeri Dada',
        'Coughing of Blood': 'Batuk berdarah',
        'Fatigue': 'Kelelahan',
        'Weight Loss': 'Penurunan Berat Badan',
        'Shortness of Breath': 'Kesulitan Bernapas',
        'Wheezing': 'Bunyi Mengi',
        'Swallowing Difficulty': 'Kesulitan Menelan',
        'Clubbing of Finger Nails': 'Clubbing Kuku Jari',
        'Frequent Cold': 'Sering Flu',
        'Dry Cough': 'Batuk Kering',
        'Snoring': 'Sering Mendengkur',
        'Level': 'Tingkat'
    })
            st.write(data)
            code = '''#daftar Kolom
Index('Usia', 'Jenis Kelamin', 'Pencemaran Udara', 'Penggunaan Alkohol', 'Alergi Debu',
'Lingkungan kerja tidak sehat', 'Risiko Genetik', 'Penyakit Paru Kronis',
'Diet Seimbang', 'Obesitas', 'Merokok', 'Perokok Pasif', 'Nyeri Dada',
'Batuk Darah', 'Kelelahan', 'Kehilangan Berat Badan', 'Sesak Nafas',
'Bersin-bersin', 'Kesulitan Menelan', 'Clubbing Kuku Jari',
'Sering Kedinginan', 'Batuk Kering', 'Mendengkur'],
dtype='object')'''
            st.code(code, language='python')
            data=pd.read_csv("datasets/trainy.csv", index_col=0)
            st.subheader("Y_Train Data")
            st.dataframe(data, use_container_width=True)
        
    with tab3:
            st.header("Lung Cancer Training Dataset")
            st.subheader("X_Test Data")
            data=pd.read_csv("datasets/testx.csv", index_col=0)
            data = data.rename(columns={
        'Age': 'Usia',
        'Gender': 'Jenis Kelamin',
        'Air Pollution': 'Polusi Udara',
        'Alcohol use': 'Konsumsi Alkohol',
        'Dust Allergy': 'Alergi Debu',
        'Genetic Risk': 'Resiko Genetik',
        'chronic Lung Disease': 'Penyakit Paru-paru Kronis',    
        'Obesity': 'Obesitas',
        'OccuPational Hazards': 'Lingkungan kerja tidak sehat',
        'Smoking': 'Merokok',
        'Passive Smoker': 'Perokok Pasif',
        'Balanced Diet': 'Diet/nutrisi Seimbang',
        'Chest Pain': 'Nyeri Dada',
        'Coughing of Blood': 'Batuk berdarah',
        'Fatigue': 'Kelelahan',
        'Weight Loss': 'Penurunan Berat Badan',
        'Shortness of Breath': 'Kesulitan Bernapas',
        'Wheezing': 'Bunyi Mengi',
        'Swallowing Difficulty': 'Kesulitan Menelan',
        'Clubbing of Finger Nails': 'Clubbing Kuku Jari',
        'Frequent Cold': 'Sering Flu',
        'Dry Cough': 'Batuk Kering',
        'Snoring': 'Sering Mendengkur',
        'Level': 'Tingkat'
    })
            st.write(data)
            code ='''#daftar Kolom
Index('Usia', 'Jenis Kelamin', 'Pencemaran Udara', 'Penggunaan Alkohol', 'Alergi Debu',
'Lingkungan kerja tidak sehat', 'Risiko Genetik', 'Penyakit Paru Kronis',
'Diet Seimbang', 'Obesitas', 'Merokok', 'Perokok Pasif', 'Nyeri Dada',
'Batuk Darah', 'Kelelahan', 'Kehilangan Berat Badan', 'Sesak Nafas',
'Bersin-bersin', 'Kesulitan Menelan', 'Clubbing Kuku Jari',
'Sering Kedinginan', 'Batuk Kering', 'Mendengkur', 'Level'],
dtype='object')'''
            st.code(code, language='python')
            data=pd.read_csv("datasets/testy.csv", index_col=0)
            st.subheader("Y_Test Data")
            st.dataframe(data, use_container_width=True)
            
    with tab4:
            st.header("Algoritma yang digunakan:")
            algo = Image.open("images/algo.png")

            st.image(algo, caption='ML Algorithms',width=500)

            st.write("Karena ini adalah multi-class classification, kami menggunakan Algoritma yang umumnya digunakan untuk Supervised learning")

            st.markdown(
                """
                Supervised Learning Algorithms:
                - Linear Regression
                - Support Vector Machine
                - K-Nearest Neighbours (KNN)
                - Decision Tree Classifier
                """
                )
            
            st.write("Akurasi algoritma: ")
            code = '''Akurasi dari Logistic Regression : 95 %
Akurasi dari SVM : 100 %
Akurasi dari Decision Tree : 100 %
Akurasi dari KNN : 100 %'''
            st.code(code, language='python')

            st.header("Confusion Matrix")

            col1, col2 = st.columns(2)

            with col1:
                algo = Image.open("images/lg.png")

                st.image(algo, caption='LG Confusion Matrix',width=350)

            with col2:
                algo = Image.open("images/svm.png")

                st.image(algo, caption='SVM Confusion Matrix',width=390)

            

elif selected_tab == "Deteksi kanker paru-paru":
    tab1, tab2= st.tabs(["Analisis Dataset",'Algoritma'])
    with tab1:
        st.header("Dataset Gambar CT-Scan")
        st.image("images/dada.png", caption="Gambar pada Dataset CT-Scan dada")
        st.write("Jumlah gambar pada dataset yang digunakan: ")
        code = '''Total training gambar paru-paru kanker: 486
Total training gambar paru-paru normal: 151
Total test gambar paru-paru kanker: 486
Total test gambar paru-paru normal: 151'''
        st.code(code, language='python')
        url = "https://www.kaggle.com/datasets/mohamedhanyyy/chest-ctscan-images"
        st.write("Berikut dataset yang digunakan [Dataset CT-Scan Dada](%s)" % url)
    with tab2:
        st.header("Convolutional Neural Network Model")
        st.write("CNN memiliki kemampuan bawaan untuk mengekstraksi fitur-fitur hierarki dari data citra. Ini memungkinkan model untuk secara otomatis mempelajari fitur-fitur penting pada berbagai tingkat abstraksi, mulai dari fitur sederhana seperti tepi hingga fitur kompleks seperti pola kanker.")
        

        st.subheader("Pendekatan yang digunakan: ")
        st.markdown(
                """
                - Model dilatih menggunakan API Keras
                - Menggunakan Lapisan Konvolusi 2D bersama dengan Lapisan MaxPooling yang berurutan untuk meningkatkan kinerja model.
                - Klasifikasi dua kelas, yaitu masalah klasifikasi biner, kita akan mengakhiri jaringan dengan aktivasi sigmoid. Output dari jaringan akan berupa skalar tunggal antara 0 dan 1, yang mengkodekan probabilitas bahwa gambar saat ini adalah kelas 1 (berlawanan dengan kelas 0).
                """
                )
        st.subheader("Model Summary()")
        code='''
        Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 conv2d (Conv2D)             (None, 148, 148, 16)      448       
                                                                 
 max_pooling2d (MaxPooling2  (None, 74, 74, 16)        0         
 D)                                                              
                                                                 
 conv2d_1 (Conv2D)           (None, 72, 72, 32)        4640      
                                                                 
 max_pooling2d_1 (MaxPoolin  (None, 36, 36, 32)        0         
 g2D)                                                            
                                                                 
 conv2d_2 (Conv2D)           (None, 34, 34, 64)        18496     
                                                                 
 max_pooling2d_2 (MaxPoolin  (None, 17, 17, 64)        0         
 g2D)                                                            
                                                                 
 flatten (Flatten)           (None, 18496)             0         
                                                                 
 dense (Dense)               (None, 512)               9470464   
                                                                 
 dense_1 (Dense)             (None, 1)                 513       
                                                                 
=================================================================
Total params: 9494561 (36.22 MB)
Trainable params: 9494561 (36.22 MB)
Non-trainable params: 0 (0.00 Byte)
_________________________________________________________________
        '''
        st.code(code, language='python')
        st.subheader("Model Compile ")
        st.write("Model dilatih dengan loss binary_crossentropy, karena ini adalah masalah klasifikasi biner, maka aktivasi akhir  adalah sigmoid. Kita akan menggunakan pengoptimal rmsprop dengan learning rate 0.001.")
        code = '''from tensorflow.keras.optimizers import RMSprop
            model.compile(optimizer=RMSprop(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics = ['accuracy'])'''
        st.code(code, language='python')
        code = '''model.fit(
            train_generator,
            epochs=15,
            validation_data=validation_generator,
            verbose=2
                )'''
        st.code(code, language='python')
        st.caption("Hasil akurasi :")
        code='''
        Epoch 5/15
32/32 - 27s - loss: 0.0833 - accuracy: 0.9796 - val_loss: 0.5608 - val_accuracy: 0.7859 - 27s/epoch - 834ms/step
Epoch 6/15
32/32 - 31s - loss: 0.0149 - accuracy: 0.9953 - val_loss: 0.5376 - val_accuracy: 0.8196 - 31s/epoch - 960ms/step
Epoch 7/15
32/32 - 27s - loss: 0.0141 - accuracy: 0.9937 - val_loss: 1.4226 - val_accuracy: 0.6177 - 27s/epoch - 841ms/step
Epoch 8/15
32/32 - 31s - loss: 0.0087 - accuracy: 0.9984 - val_loss: 0.5090 - val_accuracy: 0.8349 - 31s/epoch - 964ms/step
Epoch 9/15
32/32 - 27s - loss: 9.8805e-04 - accuracy: 1.0000 - val_loss: 0.6309 - val_accuracy: 0.8410 - 27s/epoch - 844ms/step
Epoch 10/15
32/32 - 26s - loss: 2.8183e-04 - accuracy: 1.0000 - val_loss: 0.5964 - val_accuracy: 0.8777 - 26s/epoch - 825ms/step
        '''
        st.code(code, language='python')
        st.subheader("Plotting the Traning vs Validation (Accuracy dan Loss)")
        col1, col2 = st.columns(2)
        
        with col1:
            acc = Image.open("images/acc.png")
            st.image(acc, caption='Model Akurasi',width=350)

        with col2:
                loss = Image.open("images/loss.png")

                st.image(loss, caption='Model Loss',width=350)

        st.write("Seperti yang dapat kita lihat dari diagram di atas bahwa Model  berkerja baik pada Data Pelatihan dan juga Data Validasi")


