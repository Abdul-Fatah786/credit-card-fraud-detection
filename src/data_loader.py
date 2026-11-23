"""Loading the Kaggle file and creating train / validation / test splits."""
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import DATA_PATH, FEATURES, REQUIRED_COLUMNS, SEED, TARGET
from src.preprocessing import add_features


def load_data(path=DATA_PATH) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}.\n"
            "Download creditcard.csv from https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud "
            "and put it in the data/ folder (see data/README.md)."
        )
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLUMNS + [TARGET] if c not in df.columns]
    if missing:
        raise ValueError(f"The file is missing expected columns: {missing}")
    return df.reset_index(drop=True)


def split_data(df: pd.DataFrame, kind: str = "random", seed: int = SEED) -> dict:
    """60 / 20 / 20 split.

    kind="random": stratified random split (conventional benchmark).
    kind="time"  : sorted by Time, oldest 60% train, next 20% validation, newest 20% test
                   (a more realistic simulation: the model never sees the future).
    """
    X = add_features(df)[FEATURES]
    y = df[TARGET]

    if kind == "random":
        X_dev, X_test, y_dev, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=seed)
        X_train, X_val, y_train, y_val = train_test_split(X_dev, y_dev, test_size=0.25, stratify=y_dev, random_state=seed)
    elif kind == "time":
        order = df["Time"].sort_values(kind="stable").index
        n = len(order)
        tr, va, te = order[: int(n * 0.6)], order[int(n * 0.6): int(n * 0.8)], order[int(n * 0.8):]
        X_train, X_val, X_test = X.loc[tr], X.loc[va], X.loc[te]
        y_train, y_val, y_test = y.loc[tr], y.loc[va], y.loc[te]
        X_dev, y_dev = pd.concat([X_train, X_val]), pd.concat([y_train, y_val])
    else:
        raise ValueError("kind must be 'random' or 'time'")

    return dict(X_train=X_train, X_val=X_val, X_test=X_test, X_dev=X_dev,
                y_train=y_train, y_val=y_val, y_test=y_test, y_dev=y_dev)
