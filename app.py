import streamlit as st
import pandas as pd
import joblib

# Mengatur tampilan halaman
st.set_page_config(page_title="Deteksi Serangan IoT", page_icon="🛡️")

# 1. Memuat (load) Pipeline Utuh dari Tahap 8/9
# Karena ini Pipeline, data akan otomatis di-scale sebelum diprediksi
import os
model_path = os.path.join(os.path.dirname(__file__), 'pipeline_terbaik.pkl')
pipeline = joblib.load(model_path)

st.title("🛡️ Sistem Deteksi Kerentanan IoT")
st.write("Aplikasi antarmuka untuk mendemonstrasikan model Machine Learning dalam mendeteksi serangan jaringan IoT.")
st.markdown("---")

# 2. Fungsi untuk memuat dataset sebagai simulasi aliran data jaringan
@st.cache_data
def load_data():
    df = pd.read_csv('Preprocessed Balanced dataset.csv')
    X = df.drop(columns=['Label', 'Attack_Category', 'Attack_sub_category'])
    y = df['Label']
    return X, y

X, y = load_data()

st.write("Klik tombol di bawah untuk mengambil sampel paket data jaringan secara acak dan melihat hasil deteksinya.")

# 3. Tombol untuk melakukan prediksi simulasi
if st.button("🔍 Simulasi Deteksi Paket Data Jaringan"):
    # Mengambil 1 baris data secara acak
    sampel_data = X.sample(1)
    label_asli = y[sampel_data.index].values[0]
    
    st.write("### 📡 Data Jaringan yang Masuk (Fitur):")
    st.dataframe(sampel_data)
    
    # Melakukan prediksi dengan Pipeline
    prediksi = pipeline.predict(sampel_data)
    
    st.write("### 🤖 Hasil Analisis Model:")
    if prediksi[0] == 1:
        st.error("🚨 PERINGATAN: Terdeteksi sebagai aktivitas SERANGAN (Attack)!")
    else:
        st.success("✅ AMAN: Lalu lintas jaringan Normal.")
        
    st.write(f"*Bocoran Kunci Jawaban dari Dataset: **{'Serangan (1)' if label_asli == 1 else 'Normal (0)'}***")
