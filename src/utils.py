"""Small helpers: business simulation, top-K prioritisation, saving JSON."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import DEFAULT_BUSINESS


def business(y_true, proba, thr, params=None) -> dict:
    """Hypothetical cost model. All dollar figures come from `params` (assumptions, not real data)."""
    params = params or DEFAULT_BUSINESS
    pred = np.asarray(proba) >= thr
    yt = np.asarray(y_true).astype(bool)
    caught, missed, flagged = int((pred & yt).sum()), int((~pred & yt).sum()), int(pred.sum())
    prevented = caught * params["avg_fraud_loss"] * params["prevent_rate"]
    cost = flagged * params["alert_cost"]
    return {
        "alerts_raised": flagged,
        "fraud_detected": caught,
        "fraud_missed": missed,
        "losses_prevented": float(prevented),
        "investigation_cost": float(cost),
        "net_benefit": float(prevented - cost),
        "loss_with_no_model": float(int(yt.sum()) * params["avg_fraud_loss"]),
    }


def top_k_table(y_true, proba, ks=(50, 100, 200, 500)) -> pd.DataFrame:
    """If analysts can only check the K highest-risk transactions, how many frauds do we find?"""
    y = np.asarray(y_true)
    order = np.argsort(-np.asarray(proba))
    total = max(int(y.sum()), 1)
    rows = []
    for k in ks:
        k = min(k, len(y))
        hit = int(y[order[:k]].sum())
        rows.append({"K": k, "frauds_found": hit, "precision_at_k": round(hit / k, 4),
                     "share_of_all_fraud": round(hit / total, 4)})
    return pd.DataFrame(rows)


def save_json(obj, path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=float)
