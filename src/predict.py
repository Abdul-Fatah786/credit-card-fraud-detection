"""Scoring new transactions with the saved model. Used by both the Streamlit app and the FastAPI service."""
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.config import BEST_MODEL_PATH
from src.preprocessing import prepare, risk_level, validate

_CACHE = {}


def model_path() -> Path:
    """Where the model lives. Override with the FRAUD_MODEL_PATH environment variable."""
    return Path(os.environ.get("FRAUD_MODEL_PATH", BEST_MODEL_PATH))


def load_bundle(path=None) -> dict:
    path = Path(path) if path else model_path()
    key = str(path)
    if key not in _CACHE:
        if not path.exists():
            raise FileNotFoundError(f"Model not found at {path}. Train it first with:  python -m src.train")
        _CACHE[key] = joblib.load(path)
    return _CACHE[key]


def score_dataframe(df: pd.DataFrame, threshold=None, bundle=None) -> pd.DataFrame:
    """Raw transactions (creditcard.csv columns) -> probability, risk level and decision per row."""
    bundle = bundle or load_bundle()
    problems = validate(df)
    if problems:
        raise ValueError("; ".join(problems))
    X = prepare(df, bundle["scaler"])
    proba = bundle["model"].predict_proba(X)[:, 1]
    thr = float(bundle["threshold"] if threshold is None else threshold)
    out = pd.DataFrame({
        "transaction": df.index,
        "amount": df["Amount"].round(2).to_numpy(),
        "fraud_probability": np.round(proba, 6),
    })
    out["risk_level"] = risk_level(proba).astype(str).to_numpy()
    out["decision"] = np.where(proba >= thr, "Flag for review", "Allow")
    return out


def predict_one(transaction: dict, threshold=None, bundle=None) -> dict:
    bundle = bundle or load_bundle()
    thr = float(bundle["threshold"] if threshold is None else threshold)
    row = score_dataframe(pd.DataFrame([transaction]), thr, bundle).iloc[0]
    return {"fraud_probability": float(row["fraud_probability"]), "risk_level": str(row["risk_level"]),
            "decision": str(row["decision"]), "threshold": thr}
