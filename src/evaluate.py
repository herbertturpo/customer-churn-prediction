from sklearn.metrics import classification_report, roc_auc_score
from src.config import OPERATING_THRESHOLD

def evaluate_production_model(pipeline, X_test, y_test):
    """Generates evaluation metrics for the production-ready pipeline."""
    probs = pipeline.predict_proba(X_test)[:, 1]
    preds = (probs >= OPERATING_THRESHOLD).astype(int)

    report = classification_report(y_test, preds)
    auc = roc_auc_score(y_test, probs)
    
    return {"report": report, "auc": auc}