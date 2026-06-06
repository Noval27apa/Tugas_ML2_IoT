import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title="Deteksi Serangan IoT", page_icon="🛡️")

# 1. Memuat Pipeline
# Pastikan file pipeline_terbaik.pkl ada di folder yang sama di GitHub
model_path = 'pipeline_terbaik.pkl'
pipeline = joblib.load(model_path)

st.title("🛡️ Sistem Deteksi Kerentanan IoT")
st.write("Aplikasi simulasi deteksi serangan jaringan IoT menggunakan model Machine Learning.")
st.markdown("---")

# 2. Fungsi Load Data
@st.cache_data
def load_data():
    df = pd.read_csv('Preprocessed Balanced dataset.csv')
    # Hapus kolom target
    target_cols = ['Label', 'Attack_Category', 'Attack_sub_category']
    X = df.drop(columns=[c for c in target_cols if c in df.columns])
    return X

X = load_data()

# 3. Tombol Prediksi
if st.button("🔍 Simulasi Deteksi Paket Data Jaringan"):
    sampel_data = X.sample(1)
    
    # --- JURUS PENYELARAS MUTLAK ---
    # Memastikan fitur yang masuk ke model SAMA PERSIS dengan saat training
    expected_features = pipeline.feature_names_in_
    
    # Jika ada fitur yang kurang, tambahkan dengan nilai 0
    for col in expected_features:
        if col not in sampel_data.columns:
            sampel_data[col] = 0
            
    # Pastikan urutan kolom sama dan hanya ambil yang dibutuhkan
    sampel_data = sampel_data[expected_features]
    # -------------------------------
    
    st.write("### 📡 Data Jaringan yang Masuk (Fitur):")
    st.dataframe(sampel_data)
    
    # Prediksi
    prediksi = pipeline.predict(sampel_data)
    
    st.write("### 🤖 Hasil Analisis Model:")
    if prediksi[0] == 1:
        st.error("🚨 PERINGATAN: Terdeteksi sebagai aktivitas SERANGAN (Attack)!")
    else:
        st.success("✅ AMAN: Lalu lintas jaringan Normal.")
