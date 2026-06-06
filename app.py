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
    
    # Ambil label target
    y = df['Label'] if 'Label' in df.columns else None
    
    # Buang label dari data fitur
    X = df.drop(columns=['Label']) if 'Label' in df.columns else df.copy()
    
    # --- 1. Perbaiki Masalah Garis Miring vs Garis Bawah ---
    X = X.rename(columns={
        'fwd_pkts_s': 'fwd_pkts/s', 
        'bwd_pkts_s': 'bwd_pkts/s'
    })
    
    # --- 2. Penyelarasan Anti-Gagal dengan Model ---
    if hasattr(pipeline, "feature_names_in_"):
        expected_features = pipeline.feature_names_in_
        
        # Jika model mencari kolom yang tidak ada (seperti Attack_type), tambahkan kolom palsu berisi 0
        for col in expected_features:
            if col not in X.columns:
                X[col] = 0
                
        # Urutkan kolom sama persis seperti saat model dilatih, buang sisanya
        X = X[expected_features]
        
    return X, y

X, y = load_data()

st.write("Klik tombol di bawah untuk mengambil sampel paket data jaringan secara acak dan melihat hasil deteksinya.")

# 3. Tombol Eksekusi
if st.button("🔍 Simulasi Deteksi Paket Data Jaringan"):
    # Mengambil 1 baris data secara acak
    sampel_data = X.sample(1)
    
    st.write("### 📡 Data Jaringan yang Masuk (Fitur):")
    st.dataframe(sampel_data)
    
    # Melakukan prediksi dengan Pipeline
    prediksi = pipeline.predict(sampel_data)
    
    st.write("### 🤖 Hasil Analisis Model:")
    if prediksi[0] == 1:
        st.error("🚨 PERINGATAN: Terdeteksi sebagai aktivitas SERANGAN (Attack)!")
    else:
        st.success("✅ AMAN: Lalu lintas jaringan Normal.")
        
    if y is not None:
        label_asli = y[sampel_data.index].values[0]
        st.write(f"*Bocoran Kunci Jawaban dari Dataset: **{'Serangan (1)' if label_asli == 1 else 'Normal (0)'}***")
