import json

import pandas as pd

from src.predict import load_bundle, predict_one, score_dataframe


def test_training_writes_all_artifacts(trained):
    root = trained["root"]
    assert (root / "models" / "best_model.pkl").exists()
    assert (root / "reports" / "experiment_results.csv").exists()
    assert (root / "reports" / "threshold_analysis.csv").exists()
    assert (root / "reports" / "final_report.json").exists()
    assert (root / "data" / "sample_transactions.csv").exists()
    for fig in ("pr_roc_curves.png", "confusion_matrix_test.png", "threshold_curve.png"):
        assert (root / "reports" / "figures" / fig).exists()


def test_results_table_has_all_experiments(trained):
    res = trained["results"]
    assert len(res) == 7 + 3  # 7 models (no xgboost here) + 3 threshold rows
    dummy = res[res["Model"] == "Dummy (majority class)"].iloc[0]
    assert dummy["Fraud Caught"] == 0 and dummy["Accuracy"] > 0.9  # high accuracy, zero fraud caught


def test_model_beats_the_dummy(trained):
    assert trained["test_metrics"]["recall"] > 0.5
    assert trained["test_metrics"]["pr_auc"] > 0.5
    assert 0 < trained["threshold"] < 1


def test_final_report_json(trained):
    rep = json.loads((trained["root"] / "reports" / "final_report.json").read_text())
    assert rep["experiment"] == trained["best_name"]
    assert "business_test" in rep and "top_k_test" in rep


def test_score_dataframe_and_predict_one(trained, fake_df):
    bundle = load_bundle(trained["root"] / "models" / "best_model.pkl")
    scored = score_dataframe(fake_df.drop(columns=["Class"]).head(50), bundle=bundle)
    assert len(scored) == 50
    assert scored["fraud_probability"].between(0, 1).all()
    assert set(scored["risk_level"]) <= {"Low", "Medium", "High", "Critical"}
    one = predict_one(fake_df.drop(columns=["Class"]).iloc[0].to_dict(), bundle=bundle)
    assert set(one) == {"fraud_probability", "risk_level", "decision", "threshold"}


def test_fraud_rows_score_higher_than_legit(trained, fake_df):
    bundle = load_bundle(trained["root"] / "models" / "best_model.pkl")
    scored = score_dataframe(fake_df.drop(columns=["Class"]), bundle=bundle)
    p = scored["fraud_probability"].to_numpy()
    assert p[fake_df["Class"].to_numpy() == 1].mean() > p[fake_df["Class"].to_numpy() == 0].mean() + 0.3
