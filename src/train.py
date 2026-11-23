"""Train, compare and save the fraud models.

Usage (from the project root):
    python -m src.train                 # fast mode (smaller forests), random split
    python -m src.train --full          # full-size models (slower)
    python -m src.train --split time    # time-based split (more realistic)
    python -m src.train --no-xgb        # skip XGBoost
    python -m src.train --save-all      # also save every experiment's model
"""
import argparse
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from imblearn.over_sampling import SMOTE
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from src.config import (BEST_MODEL_PATH, DATA_DIR, DATA_PATH, FEATURES, MODELS_DIR, REPORTS_DIR, SEED, TARGET)
from src.data_loader import load_data, split_data
from src.evaluate import (best_f1_threshold, plot_confusion, plot_pr_roc, plot_threshold_curve, score,
                          threshold_table)
from src.preprocessing import fit_scaler, scale_with
from src.utils import business, save_json, top_k_table

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:  # xgboost is optional
    HAS_XGB = False


def make_model(kind: str, class_weight: bool = False, fast: bool = True, y_train=None):
    cw = "balanced" if class_weight else None
    if kind == "dummy":
        return DummyClassifier(strategy="most_frequent")
    if kind == "logreg":
        return LogisticRegression(max_iter=2000, class_weight=cw)
    if kind == "tree":
        return DecisionTreeClassifier(max_depth=8, min_samples_leaf=5, class_weight=cw, random_state=SEED)
    if kind == "rf":
        return RandomForestClassifier(n_estimators=40 if fast else 100, max_depth=12, min_samples_leaf=2, n_jobs=-1,
                                      class_weight="balanced_subsample" if class_weight else None, random_state=SEED)
    if kind == "xgb":
        spw = 1.0
        if class_weight and y_train is not None:
            spw = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))
        return XGBClassifier(n_estimators=150 if fast else 300, max_depth=5, learning_rate=0.1, subsample=0.8,
                             scale_pos_weight=spw, eval_metric="aucpr", n_jobs=-1, random_state=SEED)
    raise ValueError(f"unknown model kind: {kind}")


def fit_model(kind, class_weight, smote, X, y, fast=True):
    """Fit one model. SMOTE (if on) is applied to the data passed in, which must be TRAINING data only."""
    if smote:
        X, y = SMOTE(sampling_strategy=0.1, random_state=SEED).fit_resample(X, y)
    model = make_model(kind, class_weight, fast, y)
    model.fit(X, y)
    return model


def experiment_plan(use_xgb: bool = True) -> list:
    """(name, kind, label, technique, class_weight, smote)"""
    plan = [
        ("1. Dummy", "dummy", "Dummy (majority class)", "None", False, False),
        ("2. LogReg default", "logreg", "Logistic Regression", "Default", False, False),
        ("3. LogReg class weights", "logreg", "Logistic Regression", "Class weights", True, False),
        ("4. Decision Tree", "tree", "Decision Tree", "Default", False, False),
        ("5. Random Forest", "rf", "Random Forest", "Default", False, False),
        ("6. RF class weights", "rf", "Random Forest", "Class weights", True, False),
        ("7. RF SMOTE", "rf", "Random Forest", "SMOTE", False, True),
    ]
    if use_xgb and HAS_XGB:
        plan.append(("8. XGBoost", "xgb", "XGBoost", "Default", False, False))
    return plan


def _row(name, label, technique, thr, s, seconds=0.0) -> dict:
    return {"Experiment": name, "Model": label, "Technique": technique, "Threshold": thr,
            "Accuracy": round(s["accuracy"], 4), "Precision": round(s["precision"], 4),
            "Recall": round(s["recall"], 4), "F1": round(s["f1"], 4),
            "PR-AUC": round(s["pr_auc"], 4), "ROC-AUC": round(s["roc_auc"], 4),
            "Fraud Caught": s["tp"], "Fraud Missed": s["fn"], "False Alarms": s["fp"],
            "Train Time (s)": round(seconds, 1)}


def train_pipeline(df: pd.DataFrame, fast=True, use_xgb=True, split="random", models_dir=MODELS_DIR,
                   reports_dir=REPORTS_DIR, data_dir=DATA_DIR, save_all=False, verbose=True) -> dict:
    log = print if verbose else (lambda *a, **k: None)
    models_dir, reports_dir, data_dir = Path(models_dir), Path(reports_dir), Path(data_dir)
    fig_dir = reports_dir / "figures"
    for d in (models_dir, fig_dir, data_dir):
        d.mkdir(parents=True, exist_ok=True)

    sp = split_data(df, split)
    log(f"Split ({split}): train={len(sp['y_train']):,} val={len(sp['y_val']):,} test={len(sp['y_test']):,} | "
        f"fraud in train/val/test = {int(sp['y_train'].sum())}/{int(sp['y_val'].sum())}/{int(sp['y_test'].sum())}")

    # 1) scaler is fitted on TRAIN only
    scaler = fit_scaler(sp["X_train"])
    X_tr, X_va = scale_with(scaler, sp["X_train"]), scale_with(scaler, sp["X_val"])
    y_tr, y_va = sp["y_train"], sp["y_val"]

    # 2) experiments, evaluated on the VALIDATION set
    rows, probas, configs, fitted = [], {}, {}, {}
    for name, kind, label, technique, cw, sm in experiment_plan(use_xgb):
        t0 = time.time()
        model = fit_model(kind, cw, sm, X_tr, y_tr, fast)
        p = model.predict_proba(X_va)[:, 1]
        rows.append(_row(name, label, technique, 0.5, score(y_va, p), time.time() - t0))
        probas[name], configs[name], fitted[name] = p, (kind, cw, sm), model
        r = rows[-1]
        log(f"  {name:<26} recall={r['Recall']:.3f} precision={r['Precision']:.3f} PR-AUC={r['PR-AUC']:.3f} ({r['Train Time (s)']}s)")

    # 3) best model = highest validation PR-AUC (ignoring the dummy)
    real = [r for r in rows if r["Model"] != "Dummy (majority class)"]
    best_name = max(real, key=lambda r: (r["PR-AUC"], r["Recall"]))["Experiment"]
    kind, cw, sm = configs[best_name]
    log(f"Best model by validation PR-AUC: {best_name}")

    # experiments 9-11: best model at three thresholds (no retraining)
    for i, thr in zip((9, 10, 11), (0.1, 0.3, 0.5)):
        rows.append(_row(f"{i}. Best @ threshold {thr}", kind, f"Threshold {thr}", thr, score(y_va, probas[best_name], thr)))
    results = pd.DataFrame(rows)

    # 4) threshold chosen on VALIDATION (best F1)
    thr_table = threshold_table(y_va, probas[best_name])
    final_thr = best_f1_threshold(y_va, probas[best_name])
    log(f"Threshold chosen on validation (best F1): {final_thr}")

    # 5) retrain on train + validation, refit scaler on that combined set
    final_scaler = fit_scaler(sp["X_dev"])
    final_model = fit_model(kind, cw, sm, scale_with(final_scaler, sp["X_dev"]), sp["y_dev"], fast)

    # 6) the ONE look at the test set
    test_proba = final_model.predict_proba(scale_with(final_scaler, sp["X_test"]))[:, 1]
    ts = score(sp["y_test"], test_proba, final_thr)
    log(f"\nTEST RESULTS: recall={ts['recall']:.3f} ({ts['tp']}/{ts['tp'] + ts['fn']} frauds) "
        f"precision={ts['precision']:.3f} ({ts['fp']} false alarms) PR-AUC={ts['pr_auc']:.3f} ROC-AUC={ts['roc_auc']:.3f}")

    # 7) save everything
    bundle = {"model": final_model, "scaler": final_scaler, "threshold": final_thr, "features": FEATURES,
              "config": {"experiment": best_name, "model": kind, "class_weight": cw, "smote": sm, "split": split},
              "test_metrics": {k: float(v) for k, v in ts.items()}, "sklearn_version": sklearn.__version__}
    joblib.dump(bundle, models_dir / "best_model.pkl", compress=3)
    if save_all:
        for name, m in fitted.items():
            safe = name.split(". ", 1)[1].lower().replace(" ", "_")
            joblib.dump({"model": m, "scaler": scaler}, models_dir / f"{safe}.pkl", compress=3)

    results.to_csv(reports_dir / "experiment_results.csv", index=False)
    thr_table.to_csv(reports_dir / "threshold_analysis.csv", index=False)
    save_json({"experiment": best_name, "threshold": final_thr, "split": split, "test_metrics": bundle["test_metrics"],
              "business_test": business(sp["y_test"], test_proba, final_thr),
              "top_k_test": top_k_table(sp["y_test"], test_proba).to_dict("records")}, reports_dir / "final_report.json")

    plot_pr_roc(y_va, {n: p for n, p in probas.items() if not n.startswith("1.")}, fig_dir / "pr_roc_curves.png")
    plot_confusion(sp["y_test"], test_proba, final_thr, fig_dir / "confusion_matrix_test.png", "Test confusion matrix")
    plot_threshold_curve(thr_table, fig_dir / "threshold_curve.png")

    # a small demo file for the dashboard and API: every test fraud + up to 1500 random legitimate test rows
    rng = np.random.RandomState(SEED)
    y_test = sp["y_test"]
    fraud_idx = y_test.index[y_test == 1].to_numpy()
    legit_pool = y_test.index[y_test == 0].to_numpy()
    legit_idx = rng.choice(legit_pool, size=min(1500, len(legit_pool)), replace=False)
    df.loc[np.concatenate([fraud_idx, legit_idx])].sample(frac=1, random_state=SEED).to_csv(
        data_dir / "sample_transactions.csv", index=False)

    log(f"\nSaved: {models_dir / 'best_model.pkl'}, {reports_dir / 'experiment_results.csv'}, figures in {fig_dir}")
    return {"bundle": bundle, "results": results, "threshold": final_thr, "test_metrics": ts, "best_name": best_name}


def main(argv=None):
    ap = argparse.ArgumentParser(description="Train the credit card fraud models")
    ap.add_argument("--data", default=str(DATA_PATH), help="path to creditcard.csv")
    ap.add_argument("--split", choices=["random", "time"], default="random")
    ap.add_argument("--full", action="store_true", help="full-size models (slower)")
    ap.add_argument("--no-xgb", action="store_true", help="skip XGBoost")
    ap.add_argument("--save-all", action="store_true", help="save every experiment's model too")
    args = ap.parse_args(argv)

    try:
        df = load_data(args.data)
    except (FileNotFoundError, ValueError) as e:
        sys.exit(str(e))
    print(f"Loaded {len(df):,} transactions, {int(df[TARGET].sum())} fraud ({df[TARGET].mean() * 100:.3f}%)")
    train_pipeline(df, fast=not args.full, use_xgb=not args.no_xgb, split=args.split, save_all=args.save_all)


if __name__ == "__main__":
    main()
