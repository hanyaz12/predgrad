from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image
df = pd.read_csv("./data/data_lulus_tepat_waktu.csv")

# hapus kolom ‘tepat’ dalam dataset lalu masukan ke variabel x
x = df.drop(["tepat"], axis=1)

# mengambil kolom ‘tepat’ dalam dataset lalu masukan ke variabel y
y = df["tepat"]

# split train dan test

x_train, x_test, y_train, y_test = train_test_split(x, y)

# latih model dengan Gaussian Naïve Bayes

modelNB = GaussianNB()

nbtrain = modelNB.fit(x_train, y_train)

# streamlit
im = Image.open("./images/logoupb.png")
st.set_page_config(
    page_title="Prediksi Kelulusan",
    page_icon=im
)

st.image(im, width=120)
st.title('Prediksi Kelulusan')
ip1 = st.number_input('Masukkan IP Semester 1', 0.0, 4.0)
ip2 = st.number_input('Masukkan IP Semester 2', 0.0, 4.0)
ip3 = st.number_input('Masukkan IP Semester 3', 0.0, 4.0)
ip4 = st.number_input('Masukkan IP Semester 4', 0.0, 4.0)

prediksi = st.button("Prediksi")

if prediksi:
    hasil = nbtrain.predict([[ip1, ip2, ip3, ip4]])
    st.success(f"Lulus tepat waktu : {hasil[0]}")


st.markdown("<br>",
            unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'><b>Muhamad Ridwan</b></p>",
            unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'><b>312010026</b></p>",
            unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center;'><b>Universitas Pelita Bangsa</b></h4>",
            unsafe_allow_html=True)
st.markdown("<h5 style='text-align: center;'><b>2023</b></h5>",
            unsafe_allow_html=True)
