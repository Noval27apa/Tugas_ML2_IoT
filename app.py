import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title="Deteksi Serangan IoT", page_icon="🛡️")

# 1. Memuat Pipeline
model_path = os.path.join(os.path.dirname(__file__), 'pipeline_terbaik.pkl')
pipeline = joblib.load(model_path)

# Dapatkan daftar fitur yang diharapkan model dari pipeline
expected_features = pipeline.feature_names_in_

st.title("🛡️ Sistem Deteksi Kerentanan IoT")
st.write("Aplikasi antarmuka untuk mendemonstrasikan model Machine Learning.")
st.markdown("---")

# 2. Fungsi memuat data
@st.cache_data
def load_data():
    df = pd.read_csv('Preprocessed Balanced dataset.csv')
    
    # Buang label jika ada
    kolom_dibuang = ['Label', 'Attack_Category', 'Attack_sub_category', 'Attack_type']
    X = df.drop(columns=[col for col in kolom_dibuang if col in df.columns])
    
    # Lakukan penyesuaian otomatis:
    # 1. Pastikan semua fitur yang ada di model ada di dataset (isi 0 jika tidak ada)
    # 2. Pastikan urutan kolom SAMA PERSIS dengan model
    for col in expected_features:
        if col not in X.columns:
            X[col] = 0
            
    # HANYA ambil kolom yang model inginkan, dan urutkan sesuai model
    X = X[expected_features]
    
    return X

X = load_data()

# 3. Tombol Eksekusi
if st.button("🔍 Simulasi Deteksi Paket Data Jaringan"):
    sampel_data = X.sample(1)
    
    st.write("### 📡 Data Jaringan yang Masuk (Fitur):")
    st.dataframe(sampel_data)
    
    # Predict
    prediksi = pipeline.predict(sampel_data)
    
    st.write("### 🤖 Hasil Analisis Model:")
    if prediksi[0] == 1:
        st.error("🚨 PERINGATAN: Terdeteksi sebagai aktivitas SERANGAN (Attack)!")
    else:
        st.success("✅ AMAN: Lalu lintas jaringan Normal.")
