import joblib
from datetime import datetime
from sklearn.model_selection import train_test_split

# Absolute imports from src package
from src.config import (
    SEED, MODEL_PATH, EXPECTED_FEATURES, TARGET, 
    OPERATING_THRESHOLD, EXPECTED_DTYPES, EXAMPLE_INPUT
)
from src.data_loader import load_raw_data
from src.train import train_production_pipeline
from src.evaluate import evaluate_production_model

def save_model_package(production_pipeline):
    """
    Consolidates the trained pipeline with its metadata and contract.
    """
    model_metadata = {
        "model_name": "LogisticRegression + RobustScaler + SMOTE (train only)",
        "model_version": "1.0.0",
        "trained_at": datetime.utcnow().isoformat(),
        "target": TARGET,
        "positive_class": 1,
        "threshold": OPERATING_THRESHOLD,
        "notes": "Pipeline trained with SMOTE only for fitting. Inference is pure Scikit-learn."
    }

    model_package = {
        "pipeline": production_pipeline,    
        "threshold": OPERATING_THRESHOLD,
        "expected_features": EXPECTED_FEATURES,
        "expected_dtypes": EXPECTED_DTYPES,
        "example_input": EXAMPLE_INPUT,
        "metadata": model_metadata
    }

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model_package, MODEL_PATH)
    print(f"Model package saved at: {MODEL_PATH}")

def run_experiment():
    # 1. Load data
    df = load_raw_data()
    X = df[EXPECTED_FEATURES]
    y = df[TARGET].apply(lambda x: 1 if x == 'Yes' else 0)
    
    # 2. Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )

    # 3. Train Production Pipeline (Extracting clean pipeline)
    # This uses: fit(SMOTE) -> return Pipeline(No SMOTE)
    production_pipeline = train_production_pipeline(X_train, y_train)

    # 4. Evaluate using the clean pipeline
    results = evaluate_production_model(production_pipeline, X_test, y_test)
    print(f"\nModel AUC: {results['auc']:.4f}")
    print(f"Classification Report:\n{results['report']}")

    # 5. Save the complete package
    save_model_package(production_pipeline)

if __name__ == "__main__":
    run_experiment()