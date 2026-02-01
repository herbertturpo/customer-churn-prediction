from pathlib import Path

# Project root directory
ROOT_DIR = Path(__file__).resolve().parent.parent

# Data directory paths
RAW_DATA_DIR = ROOT_DIR / "data" / "raw"
PROCESSED_DATA_DIR = ROOT_DIR / "data" / "processed"

# Dataset configuration
DATASET_NAME = "WA_Fn-UseC_-Telco-Customer-Churn.csv"
RAW_DATA_PATH = RAW_DATA_DIR / DATASET_NAME


# Model configuration
MODEL_VERSION = "1.0.0"
OPERATING_THRESHOLD = 0.35
TARGET = "Churn"
SEED = 42

# Model threshold optimized for business costs (FN >> FP)
OPERATING_THRESHOLD = 0.35

# Feature Contract
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

EXPECTED_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES