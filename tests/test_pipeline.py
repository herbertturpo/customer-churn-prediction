import pytest
import pandas as pd
import joblib
from src.config import MODEL_PATH, OPERATING_THRESHOLD

@pytest.fixture
def model_package():
    """Load the full model package for testing."""
    if not MODEL_PATH.exists():
        pytest.skip("Model artifact missing. Run main.py first.")
    return joblib.load(MODEL_PATH)

def test_pipeline_inference_flow(model_package):
    """Checks if the pipeline predicts correctly using the example input."""
    pipeline = model_package["pipeline"]
    # Create DataFrame from the example input in the package
    test_df = pd.DataFrame([model_package["example_input"]])
    
    from src.inference import predict_churn_batch
    results = predict_churn_batch(test_df, pipeline, model_package["threshold"])
    
    assert "churn_probability" in results.columns
    assert results.iloc[0]["churn_prediction"] in [0, 1]

def test_pipeline_with_missing_values(model_package):
    """Verifies that the imputer handles NaNs without crashing."""
    pipeline = model_package["pipeline"]
    test_data = pd.DataFrame([model_package["example_input"]])
    
    # Force a NaN in a numeric feature
    test_data["TotalCharges"] = None
    
    from src.inference import predict_churn_batch
    try:
        predict_churn_batch(test_data, pipeline)
    except Exception as e:
        pytest.fail(f"Pipeline crashed with NaN input: {e}")