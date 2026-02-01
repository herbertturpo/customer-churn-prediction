from pathlib import Path

# --- 1. Project Structure ---
ROOT_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = ROOT_DIR / "data" / "raw"
PROCESSED_DATA_DIR = ROOT_DIR / "data" / "processed"
MODELS_DIR = ROOT_DIR / "models"

DATASET_NAME = "WA_Fn-UseC_-Telco-Customer-Churn.csv"
RAW_DATA_PATH = RAW_DATA_DIR / DATASET_NAME
MODEL_PATH = MODELS_DIR / "churn_model.joblib"

# --- 2. Model Contract & Hyperparameters ---
TARGET = "Churn"
SEED = 42
# Threshold optimized for business costs (Prioritize Recall: FN >> FP)
OPERATING_THRESHOLD = 0.35

# --- 3. Feature Segmentation ---
# Used by the ColumnTransformer in preprocessing.py
NUMERIC_FEATURES = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges"
]

CATEGORICAL_FEATURES = [
    "gender", "SeniorCitizen", "Partner", "Dependents",
    "PhoneService", "MultipleLines", "InternetService",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaperlessBilling", "PaymentMethod"
]

# --- 4. Deployment Packaging Contract ---
# Explicit order and types for production consistency
EXPECTED_FEATURES = [
    "gender", "SeniorCitizen", "Partner", "Dependents",
    "tenure", "PhoneService", "MultipleLines", "InternetService",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
    "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling",
    "PaymentMethod", "MonthlyCharges", "TotalCharges"
]

EXPECTED_DTYPES = {
    "gender": "object", "SeniorCitizen": "object", "Partner": "object",
    "Dependents": "object", "tenure": "numeric", "PhoneService": "object",
    "MultipleLines": "object", "InternetService": "object",
    "OnlineSecurity": "object", "OnlineBackup": "object",
    "DeviceProtection": "object", "TechSupport": "object",
    "StreamingTV": "object", "StreamingMovies": "object",
    "Contract": "object", "PaperlessBilling": "object",
    "PaymentMethod": "object", "MonthlyCharges": "numeric",
    "TotalCharges": "numeric"
}

EXAMPLE_INPUT = {
    "gender": "Female", "SeniorCitizen": "No", "Partner": "Yes",
    "Dependents": "No", "tenure": 1, "PhoneService": "No",
    "MultipleLines": "No phone service", "InternetService": "DSL",
    "OnlineSecurity": "No", "OnlineBackup": "Yes", "DeviceProtection": "No",
    "TechSupport": "No", "StreamingTV": "No", "StreamingMovies": "No",
    "Contract": "Month-to-month", "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check", "MonthlyCharges": 29.85,
    "TotalCharges": 0.0
}