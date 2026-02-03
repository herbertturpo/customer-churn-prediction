import pandas as pd
from src.preprocessing import clean_categories

def test_clean_categories_logic():
    """Verify that 'No internet service' and empty strings are handled."""
    df_raw = pd.DataFrame({
        "MultipleLines": ["No phone service"],
        "OnlineSecurity": ["No internet service"],
        "TotalCharges": [" "]  # Your specific case from the notebook
    })
    
    df_cleaned = clean_categories(df_raw)
    
    assert df_cleaned["MultipleLines"].iloc[0] == "No"
    assert df_cleaned["OnlineSecurity"].iloc[0] == "No"
    assert df_cleaned["TotalCharges"].iloc[0] == 0