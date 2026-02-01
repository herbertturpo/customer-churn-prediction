import pandas as pd
from sklearn.preprocessing import FunctionTransformer, RobustScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from src.config import NUMERIC_FEATURES, CATEGORICAL_FEATURES, EXPECTED_FEATURES

def validate_input_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validates that the input meets the model contract.
    Ensures required columns exist and types are correct.
    """
    # 1. Check for missing columns
    missing_cols = set(EXPECTED_FEATURES) - set(df.columns) 
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # 2. Select only expected columns
    df_validated = df[EXPECTED_FEATURES].copy()

    # 3. Validate numeric types
    for col in NUMERIC_FEATURES:
        df_validated[col] = pd.to_numeric(df_validated[col], errors="coerce")

    return df_validated




def clean_categories(df: pd.DataFrame) -> pd.DataFrame:
    """
    Semantic cleaning and critical type conversion for production.
    Standardizes Telco-specific categories.
    """
    df_c = df.copy()

    if 'TotalCharges' in df_c.columns:
        df_c['TotalCharges'] = pd.to_numeric(df_c['TotalCharges'], errors='coerce')
        df_c['TotalCharges'] = df_c['TotalCharges'].fillna(0)

    cols_to_fix = [
        'MultipleLines', 'OnlineSecurity', 'OnlineBackup',
        'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies'
    ]

    for col in cols_to_fix:
        if col in df_c.columns:
            df_c[col] = df_c[col].replace({
                'No internet service': 'No',
                'No phone service': 'No'
            })

    return df_c



# Define the steps for the Scikit-learn Pipeline
cleaner_step = FunctionTransformer(clean_categories)

numeric_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', RobustScaler())
])

categorical_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_pipeline, NUMERIC_FEATURES),
        ('cat', categorical_pipeline, CATEGORICAL_FEATURES)
    ],
    remainder='drop'
)