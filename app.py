import streamlit as st
import pandas as pd
import joblib
import os

# Konfigurasi Halaman
st.set_page_config(page_title="Deteksi Serangan IoT", page_icon="🛡️")

# Memuat Model
model_path = 'pipeline_terbaik.pkl'

@st.cache_resource
def load_model():
    if os.path.exists(model_path):
        return joblib.load(model_path)
    else:
        return None

pipeline = load_model()

st.title("🛡️ Sistem Deteksi Kerentanan IoT")
st.write("Aplikasi simulasi deteksi serangan jaringan IoT menggunakan model Machine Learning.")
st.markdown("---")

# Memuat Data
@st.cache_data
def load_data():
    # Pastikan nama file CSV sesuai dengan yang ada di GitHub
    df = pd.read_csv('Preprocessed Balanced dataset.csv')
    target_cols = ['Label', 'Attack_Category', 'Attack_sub_category']
    X = df.drop(columns=[c for c in target_cols if c in df.columns])
    return X

if pipeline is None:
    st.error(f"File {model_path} tidak ditemukan! Pastikan file tersebut sudah di-upload ke folder yang sama.")
else:
    X = load_data()

    # Tombol Prediksi
    if st.button("🔍 Simulasi Deteksi Paket Data Jaringan"):
        # Ambil 1 sampel acak untuk simulasi
        sampel_data = X.sample(1)
        
        # Penyelarasan Fitur (Menggunakan nama kolom dari data asli)
        expected_features = X.columns.tolist()
        
        # Memastikan struktur kolom sesuai
        for col in expected_features:
            if col not in sampel_data.columns:
                sampel_data[col] = 0
        
        sampel_data = sampel_data[expected_features]
        
        st.write("### 📡 Data Jaringan yang Masuk (Fitur):")
        st.dataframe(sampel_data)
        
        # Prediksi
        prediksi = pipeline.predict(sampel_data)
        
        st.write("### 🤖 Hasil Analisis Model:")
        if prediksi[0] == 1:
            st.error("🚨 PERINGATAN: Terdeteksi sebagai aktivitas SERANGAN (Attack)!")
        else:
            st.success("✅ AMAN: Lalu lintas jaringan Normal.")
