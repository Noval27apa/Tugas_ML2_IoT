import mlflow
import mlflow.sklearn
import joblib
from sklearn.metrics import f1_score

# Load model pipeline yang sudah kita simpan di Tahap 8
model = joblib.load('pipeline_terbaik.pkl')

# Memulai eksperimen MLflow
mlflow.set_experiment("Deteksi_IoT_Vulnerability")

with mlflow.start_run():
    # Log model ke MLflow
    mlflow.sklearn.log_model(model, "model_decision_tree")
    
    # Log parameter jika ada (opsional)
    mlflow.log_param("model_type", "DecisionTreeClassifier")
    
    print("Model berhasil dicatat ke MLflow!")