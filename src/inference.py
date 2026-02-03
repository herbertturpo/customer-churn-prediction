import pandas as pd
import numpy as np
from src.config import OPERATING_THRESHOLD
from src.preprocessing import validate_input_dataframe

def predict_churn_batch(
    df: pd.DataFrame,
    pipeline,
    threshold: float = OPERATING_THRESHOLD,
    id_col: str = "customerID"
) -> pd.DataFrame:
    """
    Performs batch churn inference using the production pipeline.
    
    Args:
        df: Input DataFrame containing raw features.
        pipeline: Trained Scikit-learn or Imbalanced-learn Pipeline.
        threshold: Decision threshold for the positive class.
        id_col: Column name to use as identifier.
        
    Returns:
        pd.DataFrame: Results including IDs (if present), probabilities, and predictions.
    """
    # 1. Validation & Contract Enforcement
    # We use your existing function to ensure columns and types are correct before inference
    df_validated = validate_input_dataframe(df)

    # 2. Identifier Preservation
    # We extract the ID from the ORIGINAL dataframe to avoid issues if it wasn't in EXPECTED_FEATURES
    customer_ids = df[id_col].values if id_col in df.columns else None

    # 3. Inference
    # The pipeline already contains 'cleaner_step' and 'preprocessor', 
    # so we pass the validated dataframe directly.
    churn_proba = pipeline.predict_proba(df_validated)[:, 1]
    churn_pred = (churn_proba >= threshold).astype(int)

    # 4. Result Construction
    results = pd.DataFrame({
        "churn_probability": churn_proba,
        "churn_prediction": churn_pred
    })

    if customer_ids is not None:
        results.insert(0, id_col, customer_ids)

    return results