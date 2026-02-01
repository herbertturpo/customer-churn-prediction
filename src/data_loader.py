import pandas as pd
import sys
from pathlib import Path

# Add project root to sys.path to allow absolute imports from 'src'
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.config import RAW_DATA_PATH

def load_raw_data():
    """
    Loads the raw Telco Churn dataset from the data/raw folder.
    
    Returns:
        pd.DataFrame: The loaded dataset.
    """
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at: {RAW_DATA_PATH}")
    
    return pd.read_csv(RAW_DATA_PATH)

if __name__ == "__main__":
    # Internal test to verify the loader works independently
    try:
        df = load_raw_data()
        print(f"Data loaded successfully. Rows: {df.shape[0]}, Columns: {df.shape[1]}")
    except Exception as e:
        print(f"Loading failed: {e}")