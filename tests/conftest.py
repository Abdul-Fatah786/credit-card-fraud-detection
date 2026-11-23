import numpy as np
import pandas as pd
import pytest

from src.config import V_COLS


def make_fake_transactions(n=3000, fraud_rate=0.03, seed=0) -> pd.DataFrame:
    """Small synthetic data in creditcard.csv format (fraud is easy to find, so tests are fast and stable)."""
    rng = np.random.RandomState(seed)
    y = (rng.rand(n) < fraud_rate).astype(int)
    V = rng.randn(n, 28) * np.linspace(1.6, 0.5, 28)
    for k, s in {3: 2.5, 10: 2.2, 13: -3.0, 11: -2.5, 9: -2.0}.items():
        V[y == 1, k] += s
    df = pd.DataFrame(V, columns=V_COLS)
    df.insert(0, "Time", np.sort(rng.uniform(0, 172792, n)).round(0))
    df["Amount"] = rng.lognormal(3.4, 1.4, n).round(2)
    df["Class"] = y
    return df


@pytest.fixture(scope="session")
def fake_df():
    return make_fake_transactions()


@pytest.fixture(scope="session")
def trained(tmp_path_factory, fake_df):
    """Train once for the whole test session, writing to a temporary folder."""
    from src.train import train_pipeline
    root = tmp_path_factory.mktemp("project")
    out = train_pipeline(fake_df, fast=True, use_xgb=False, models_dir=root / "models",
                         reports_dir=root / "reports", data_dir=root / "data", verbose=False)
    out["root"] = root
    return out
