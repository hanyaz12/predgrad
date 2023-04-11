from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

df = pd.read_csv("data_lulus_tepat_waktu.csv")

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
st.title('Prediksi Kelulusan')
ip1 = st.number_input('Masukkan IP Semester 1', 0.0)
ip2 = st.number_input('Masukkan IP Semester 2', 0.0)
ip3 = st.number_input('Masukkan IP Semester 3', 0.0)
ip4 = st.number_input('Masukkan IP Semester 4', 0.0)

prediksi = st.button("Prediksi")

if prediksi:
    hasil = nbtrain.predict([[ip1, ip2, ip3, ip4]])
    st.success(f"Lulus tepat waktu : {hasil[0]}")

st.write('Muhamad Ridwan')
st.write('312010026')
st.write('Universitas Pelita Bangsa')
st.write('2023')
