import numpy as np
import pandas as pd

from src.config import FEATURES, SCALE_COLS
from src.data_loader import split_data
from src.preprocessing import add_features, fit_scaler, prepare, risk_level, validate


def test_validate_ok(fake_df):
    assert validate(fake_df) == []


def test_validate_reports_missing_columns(fake_df):
    problems = validate(fake_df.drop(columns=["V1", "Amount"]))
    assert problems and "V1" in problems[0] and "Amount" in problems[0]


def test_validate_reports_missing_values(fake_df):
    bad = fake_df.copy()
    bad.loc[0, "V5"] = np.nan
    assert any("missing values" in p for p in validate(bad))


def test_add_features_creates_hour_and_logs_amount(fake_df):
    out = add_features(fake_df)
    assert out["Hour"].between(0, 23).all()
    assert np.allclose(out["Amount"], np.log1p(fake_df["Amount"]))
    assert "Class" in out.columns  # target is preserved


def test_prepare_returns_model_columns_in_order(fake_df):
    sp = split_data(fake_df, "random")
    scaler = fit_scaler(sp["X_train"])
    X = prepare(fake_df.drop(columns=["Class"]), scaler)
    assert list(X.columns) == FEATURES


def test_scaler_is_fitted_on_training_data_only(fake_df):
    sp = split_data(fake_df, "random")
    scaler = fit_scaler(sp["X_train"])
    assert np.allclose(scaler.mean_, sp["X_train"][SCALE_COLS].mean().to_numpy())


def test_risk_levels():
    r = risk_level([0.0, 0.1, 0.3, 0.6, 0.9, 1.0]).astype(str).tolist()
    assert r == ["Low", "Low", "Medium", "High", "Critical", "Critical"]


def test_random_split_is_disjoint_and_stratified(fake_df):
    sp = split_data(fake_df, "random")
    tr, va, te = set(sp["X_train"].index), set(sp["X_val"].index), set(sp["X_test"].index)
    assert not (tr & va) and not (tr & te) and not (va & te)
    assert len(tr) + len(va) + len(te) == len(fake_df)
    rates = [sp[k].mean() for k in ("y_train", "y_val", "y_test")]
    assert max(rates) - min(rates) < 0.01


def test_time_split_is_ordered_in_time(fake_df):
    sp = split_data(fake_df, "time")
    t = fake_df["Time"]
    assert t.loc[sp["X_train"].index].max() <= t.loc[sp["X_val"].index].min()
    assert t.loc[sp["X_val"].index].max() <= t.loc[sp["X_test"].index].min()
