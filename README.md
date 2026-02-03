# 📡 Telco Customer Churn — End-to-End Production ML Pipeline

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![Scikit-Learn](https://img.shields.io/badge/sklearn-1.8.0-orange.svg)](https://scikit-learn.org/)
[![Tests-Passed](https://img.shields.io/badge/tests-passed-brightgreen.svg)](https://docs.pytest.org/)

An **end-to-end Machine Learning solution** designed to identify high-risk churn customers for a telecommunications provider.  
This project transitions from **experimental analysis** to a **modular, production-ready pipeline**, optimized for **real business decision-making**.

---

## 🎯 Business Context

In the telecommunications industry, **customer acquisition costs are significantly higher than retention costs**.  
Accurately identifying customers at risk of churn enables marketing and retention teams to act **before cancellation occurs**, improving ROI and customer lifetime value.

### Optimization Strategy

- **Operating Threshold:** `0.35`
- **Primary Objective:** Maximize **Recall**
- **Business Rationale:**  
  Prioritizing Recall minimizes **False Negatives**, ensuring that the majority of potential churners are captured—even at the expense of additional false positives.  
  This aligns with **high-value retention strategies**, where missing a churner is more costly than contacting a loyal customer.

---

## 🏗️ System Architecture

The project follows a **clean, decoupled, and scalable architecture**, designed for maintainability and future deployment:

## 🏗️ Project Structure & Architecture

The repository follows a **clean separation between experimentation and production**, ensuring scalability, reproducibility, and maintainability.

```plaintext
├── data/                 # Raw and processed datasets (not versioned)
├── models/               # Serialized model artifacts (.joblib) + metadata
├── notebooks/            # Exploratory analysis & research notebooks
├── reports/              # Metrics, figures, and executive outputs
├── src/                  # Production-ready core logic
│   ├── __init__.py
│   ├── config.py         # Data contracts, feature schemas, constants
│   ├── data_loader.py    # Robust data ingestion & validation
│   ├── preprocessing.py # Cleaning logic & sklearn transformers
│   ├── train.py          # Model training (SMOTE applied here only)
│   ├── evaluate.py       # Model evaluation (Recall, ROC-AUC)
│   └── inference.py      # Production-grade batch inference
├── tests/                # Automated tests & robustness checks
│   ├── test_pipeline.py
│   └── test_preprocessing.py
├── main.py               # Pipeline orchestrator (single entry point)
├── requirements.txt      # Dependency management
├── README.md             # Project documentation
└── .gitignore


## 🛠️ Key Engineering Features

### ✅ Clean Inference Pipeline
- Separates training-only components (e.g., **SMOTE**) from inference logic
- Exports a **pure Scikit-learn pipeline** suitable for production
- Results in a **lighter, safer, and faster** deployment artifact

### ✅ Strict Data Contract Enforcement
- Explicit feature schemas via `EXPECTED_DTYPES`
- Early validation prevents:
  - Silent feature drift
  - Inference-time crashes
  - Incorrect business decisions

### ✅ Automated Quality Assurance
- Robust test coverage for real-world edge cases:
  - Missing values (`NaN`)
  - Unknown categorical levels
  - Empty strings in numeric fields
  - Schema mismatches


## 🚀 Execution Guide

### 1️⃣ Environment Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt


### 2️⃣Run Full Pipeline (Train & Package)

Executes:

- Data loading & validation

- Preprocessing

- SMOTE-balanced training

- Model evaluation

- Serialization of production-ready artifacts

`python main.py`

### 3️⃣ Run Robustness Tests

`python -m pytest`

##📊 Model Specifications

- Algorithm: Logistic Regression

- Scaling: RobustScaler

- Class Imbalance Handling: SMOTE (training phase only)

- Primary Metrics:

- Recall (business-driven)

- ROC-AUC (global discrimination performance)



##✍️ Author

Herbert (Eriberto)
Data & Marketing Analytics | Machine Learning | Business-Oriented AI

🔗 Connect with me on professional networks:

- LinkedIn: 

- GitHub: 



























































