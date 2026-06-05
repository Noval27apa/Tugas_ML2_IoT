import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
import joblib

# 1. Mengatur nama eksperimen di MLflow
mlflow.set_experiment("Eksperimen_Deteksi_IoT_Kelompok1")

print("=== MEMULAI SCRIPT TRACKING MLFLOW ===")
print("1. Membaca dataset...")
df = pd.read_csv('Preprocessed Balanced dataset.csv')

# Memisahkan Fitur (X) dan Target (y)
X = df.drop(columns=['Label', 'Attack_Category', 'Attack_sub_category'])
y = df['Label']

# Membagi data (80% Training, 20% Testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 2. Menentukan parameter terbaik hasil tuning secara statis
best_params = {
    'criterion': 'entropy',
    'max_depth': 10,
    'min_samples_split': 5,
    'min_samples_leaf': 1,
    'random_state': 42
}

# MEMBUAT PIPELINE UTUH (Scaler + Model) sesuai instruksi soal
pipeline_final = Pipeline([
    ('scaler', StandardScaler()), # Menjawab instruksi: "diproses dengan scaler"
    ('classifier', DecisionTreeClassifier(**best_params))
])

# 3. Menjalankan blok mlflow.start_run()
with mlflow.start_run(run_name="Pipeline_Decision_Tree_Final"):
    print("\n[MLFLOW] Memulai pencatatan run...")
    
    # Mencatat parameter secara statis menggunakan log_params()
    print("[MLFLOW] Mencatat hyperparameter terbaik...")
    mlflow.log_params(best_params)
    
    # Latih ulang PIPELINE
    print("2. Melatih ulang PIPELINE dengan parameter terbaik...")
    pipeline_final.fit(X_train, y_train)
    
    # Melakukan prediksi untuk mengambil metrik evaluasi
    print("3. Mengevaluasi Pipeline pada data testing...")
    y_pred = pipeline_final.predict(X_test)
    
    # Menghitung akurasi dan F1-Score
    akurasi = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    
    # Mencatat metrik menggunakan log_metrics()
    print("[MLFLOW] Mencatat metrik final (Akurasi & F1-Score)...")
    metrik_final = {
        'accuracy': akurasi,
        'f1_score_macro': f1
    }
    mlflow.log_metrics(metrik_final)
    
    # Menyimpan PIPELINE sebagai artifact ke dalam MLflow
    print("[MLFLOW] Menyimpan artifact Pipeline...")
    mlflow.sklearn.log_model(pipeline_final, "pipeline_decision_tree_iot")
    
    # KITA OVERRIDE FILE .PKL TAHAP 8 AGAR MENJADI PIPELINE UTUH JUGA
    joblib.dump(pipeline_final, 'pipeline_terbaik.pkl')
    print("[INFO] File pipeline_terbaik.pkl telah diperbarui menjadi wujud Pipeline utuh.")
    
    print("\n=== SCRIPT SELESAI: SEMUA DATA BERHASIL DICATAT KE MLFLOW ===")