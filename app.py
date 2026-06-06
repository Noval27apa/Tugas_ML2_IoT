import streamlit as st
import pandas as pd
import joblib
import os

# Konfigurasi Halaman
st.set_page_config(page_title="Deteksi Serangan IoT", page_icon="🛡️")

# 1. Memuat Model
model_path = 'pipeline_terbaik.pkl'

@st.cache_resource
def load_model():
    if os.path.exists(model_path):
        return joblib.load(model_path)
    else:
        return None

pipeline = load_model()

# 2. Memuat Data Sampel (Yang berukuran kecil)
@st.cache_data
def load_data():
    # Menggunakan file sampel agar tidak terkena limit ukuran GitHub
    file_path = 'data_sampel.csv'
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        target_cols = ['Label', 'Attack_Category', 'Attack_sub_category']
        X = df.drop(columns=[c for c in target_cols if c in df.columns])
        return X
    else:
        return None

st.title("🛡️ Sistem Deteksi Kerentanan IoT")
st.write("Aplikasi simulasi deteksi serangan jaringan IoT menggunakan model Machine Learning.")
st.markdown("---")

# 3. Logika Utama
if pipeline is None:
    st.error(f"File {model_path} tidak ditemukan di GitHub!")
else:
    X = load_data()
    
    if X is None:
        st.error("File data_sampel.csv tidak ditemukan di GitHub!")
    else:
        if st.button("🔍 Simulasi Deteksi Paket Data Jaringan"):
            # Ambil 1 sampel acak
            sampel_data = X.sample(1)
            
            # Penyelarasan Fitur
            expected_features = X.columns.tolist()
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
