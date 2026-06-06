import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title="Deteksi Serangan IoT", page_icon="🛡️")

# 1. Memuat Pipeline
model_path = os.path.join(os.path.dirname(__file__), 'pipeline_terbaik.pkl')
pipeline = joblib.load(model_path)

# Dapatkan daftar fitur yang HANYA diketahui model
model_features = pipeline.feature_names_in_

st.title("🛡️ Sistem Deteksi Kerentanan IoT")
st.write("Aplikasi pendeteksi serangan IoT.")
st.markdown("---")

# 2. Fungsi memuat data
def load_data():
    df = pd.read_csv('Preprocessed Balanced dataset.csv')
    
    # Buang target
    kolom_dibuang = ['Label', 'Attack_Category', 'Attack_sub_category', 'Attack_type']
    X = df.drop(columns=[col for col in kolom_dibuang if col in df.columns])
    
    # Bikin DataFrame baru yang KOLOMNYA HANYA sesuai daftar fitur model
    # Jika kolom di model ada tapi di CSV tidak ada, isi dengan 0
    X_final = pd.DataFrame(index=X.index)
    for col in model_features:
        if col in X.columns:
            X_final[col] = X[col]
        else:
            X_final[col] = 0
            
    return X_final

X = load_data()

# 3. Tombol Eksekusi
if st.button("🔍 Simulasi Deteksi Paket Data Jaringan"):
    # Ambil 1 sampel saja
    sampel = X.sample(1)
    
    st.write("### 📡 Data Jaringan yang Masuk:")
    st.dataframe(sampel)
    
    # Prediksi
    prediksi = pipeline.predict(sampel)
    
    if prediksi[0] == 1:
        st.error("🚨 PERINGATAN: Terdeteksi sebagai aktivitas SERANGAN!")
    else:
        st.success("✅ AMAN: Lalu lintas jaringan Normal.")
