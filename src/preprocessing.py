"""Feature engineering shared by training, the Streamlit app and the API.

Keeping this in ONE place means training and deployment can never drift apart.
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.config import FEATURES, REQUIRED_COLUMNS, RISK_BINS, RISK_LABELS, SCALE_COLS


def validate(df: pd.DataFrame) -> list:
    """Return a list of problems with an uploaded transaction table (empty list = OK)."""
    problems = []
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        problems.append("Missing columns: " + ", ".join(missing))
    else:
        n_na = int(df[REQUIRED_COLUMNS].isnull().sum().sum())
        if n_na:
            problems.append(f"{n_na} missing values in the feature columns")
        if (df["Amount"] < 0).any():
            problems.append("Amount contains negative values")
    return problems


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Time -> Hour of day, Amount -> log1p. Other columns (including Class) are kept."""
    out = df.copy()
    out["Hour"] = ((out["Time"] // 3600) % 24).astype(float)
    out["Amount"] = np.log1p(out["Amount"])
    return out


def fit_scaler(X_train: pd.DataFrame) -> StandardScaler:
    """Fit the scaler on TRAINING data only (this is what prevents data leakage)."""
    return StandardScaler().fit(X_train[SCALE_COLS])


def scale_with(scaler: StandardScaler, X: pd.DataFrame) -> pd.DataFrame:
    out = X.copy()
    out[SCALE_COLS] = scaler.transform(out[SCALE_COLS])
    return out


def prepare(df: pd.DataFrame, scaler: StandardScaler) -> pd.DataFrame:
    """Raw transactions (creditcard.csv format) -> model-ready feature matrix."""
    X = add_features(df)[FEATURES]
    return scale_with(scaler, X)


def risk_level(proba) -> pd.Series:
    """Map fraud probabilities to Low / Medium / High / Critical."""
    return pd.cut(pd.Series(np.asarray(proba, dtype=float)), bins=RISK_BINS,
                  labels=RISK_LABELS, include_lowest=True)
