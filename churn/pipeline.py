"""Data loading and the model pipeline, shared by training, the notebook and the API.

Keeping preprocessing inside one scikit-learn Pipeline means the API feeds the
model exactly what it saw during training (no train/serve skew).
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "telco_churn.csv"
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "churn_model.joblib"
METADATA_PATH = MODEL_DIR / "metadata.json"

TARGET = "Churn"
CATEGORICAL = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod",
]
BINARY = ["SeniorCitizen"]            # already 0/1
NUMERIC = ["MonthlyCharges"]
LOG_NUMERIC = ["tenure"]              # churn risk falls steeply in the first months
# TotalCharges is left out on purpose: it is ~ tenure x MonthlyCharges
# (r = 0.83 with tenure), and dropping it costs no ROC-AUC.
FEATURES = CATEGORICAL + BINARY + NUMERIC + LOG_NUMERIC


def load_data(path: Path = DATA_PATH) -> tuple[pd.DataFrame, pd.Series]:
    """Return raw features X and binary target y (1 = churned)."""
    df = pd.read_csv(path)
    # 11 blank TotalCharges belong to customers with tenure == 0 (not billed yet)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)
    y = (df[TARGET] == "Yes").astype(int)
    X = df.drop(columns=["customerID", TARGET])
    return X, y


def build_preprocessor() -> ColumnTransformer:
    log_then_scale = make_pipeline(
        FunctionTransformer(np.log1p, feature_names_out="one-to-one"), StandardScaler()
    )
    return ColumnTransformer([
        ("cat", OneHotEncoder(drop="if_binary", handle_unknown="ignore"), CATEGORICAL),
        ("bin", "passthrough", BINARY),
        ("num", StandardScaler(), NUMERIC),
        ("log", log_then_scale, LOG_NUMERIC),
    ])


def build_pipeline(C: float = 1.0) -> Pipeline:
    """Final model: one-hot + scaling + logistic regression."""
    return Pipeline([
        ("prep", build_preprocessor()),
        ("model", LogisticRegression(C=C, max_iter=3000)),
    ])
