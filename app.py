import streamlit as st
import pandas as pd
import joblib
import os

# Mengatur tampilan halaman
st.set_page_config(page_title="Deteksi Serangan IoT", page_icon="🛡️")

# 1. Memuat (load) Pipeline
model_path = os.path.join(os.path.dirname(__file__), 'pipeline_terbaik.pkl')
pipeline = joblib.load(model_path)

st.title("🛡️ Sistem Deteksi Kerentanan IoT")
st.write("Aplikasi antarmuka untuk mendemonstrasikan model Machine Learning dalam mendeteksi serangan jaringan IoT.")
st.markdown("---")

# 2. Fungsi memuat data dengan penyelarasan PAKSA
@st.cache_data
def load_data():
    df = pd.read_csv('Preprocessed Balanced dataset.csv')
    
    y = df['Label'] if 'Label' in df.columns else None
    X = df.drop(columns=['Label']) if 'Label' in df.columns else df.copy()
    
    X = X.rename(columns={
        'fwd_pkts_s': 'fwd_pkts/s', 
        'bwd_pkts_s': 'bwd_pkts/s'
    })
    
    if hasattr(pipeline, "feature_names_in_"):
        expected_features = pipeline.feature_names_in_
        for col in expected_features:
            if col not in X.columns:
                X[col] = 0
        X = X[expected_features]
        
    return X, y

X, y = load_data()

st.write("Klik tombol di bawah untuk mengambil sampel paket data jaringan secara acak dan melihat hasil deteksinya.")

# 3. Tombol Eksekusi
if st.button("🔍 Simulasi Deteksi Paket Data Jaringan"):
    sampel_data = X.sample(1)
    
    st.write("### 📡 Data Jaringan yang Masuk (Fitur):")
    st.dataframe(sampel_data)
    
    # --- PERBAIKAN ERROR 1D ARRAY ---
    # Memaksa data menjadi 2 Dimensi menggunakan .reshape(1, -1) lalu dibungkus kembali menjadi DataFrame
    data_2d = sampel_data.values.reshape(1, -1)
    data_prediksi = pd.DataFrame(data_2d, columns=sampel_data.columns)
    
    # Melakukan prediksi dengan Pipeline menggunakan data yang sudah di-reshape
    prediksi = pipeline.predict(data_prediksi)
    
    st.write("### 🤖 Hasil Analisis Model:")
    if prediksi[0] == 1:
        st.error("🚨 PERINGATAN: Terdeteksi sebagai aktivitas SERANGAN (Attack)!")
    else:
        st.success("✅ AMAN: Lalu lintas jaringan Normal.")
        
    if y is not None:
        label_asli = y[sampel_data.index].values[0]
        st.write(f"*Bocoran Kunci Jawaban dari Dataset: **{'Serangan (1)' if label_asli == 1 else 'Normal (0)'}***")
