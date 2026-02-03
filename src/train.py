from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from src.preprocessing import cleaner_step, preprocessor

def train_production_pipeline(X_train, y_train):
    """
    Trains with SMOTE but returns a clean Scikit-learn Pipeline for production.
    """
    # 1. Training setup (with SMOTE)
    training_pipe = ImbPipeline(steps=[
        ("cleaning", cleaner_step),
        ("preprocessing", preprocessor),
        ("smote", SMOTE(random_state=42)),
        ("model", LogisticRegression(
            C=0.1, penalty="l2", solver="lbfgs", max_iter=1000, random_state=42
        ))
    ])

    print("Fitting training pipeline (including SMOTE)...")
    training_pipe.fit(X_train, y_train)

    # 2. Extract trained components (Production logic)
    trained_preprocessor = training_pipe.named_steps["preprocessing"]
    trained_model = training_pipe.named_steps["model"]

    # 3. Create the Inference Pipeline (No SMOTE)
    production_pipe = Pipeline(steps=[
        ("cleaning", cleaner_step),
        ("preprocessing", trained_preprocessor),
        ("model", trained_model)
    ])

    return production_pipe