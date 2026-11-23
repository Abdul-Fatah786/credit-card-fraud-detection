"""Metrics, threshold analysis and plots."""
import matplotlib

matplotlib.use("Agg")  # write figures to files; no window needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, average_precision_score, confusion_matrix,
                             f1_score, precision_recall_curve, precision_score, recall_score, roc_auc_score,
                             roc_curve)


def score(y_true, proba, thr: float = 0.5) -> dict:
    """All metrics at one threshold. Fraud (1) is the positive class."""
    y_true = np.asarray(y_true)
    pred = (np.asarray(proba) >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "accuracy": accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "pr_auc": average_precision_score(y_true, proba),
        "roc_auc": roc_auc_score(y_true, proba),
        "tp": int(tp), "fn": int(fn), "fp": int(fp), "tn": int(tn),
    }


def threshold_table(y_true, proba, grid=None) -> pd.DataFrame:
    grid = np.round(np.arange(0.05, 0.96, 0.05), 2) if grid is None else grid
    rows = []
    for t in grid:
        s = score(y_true, proba, t)
        rows.append({"Threshold": float(t), "Precision": round(s["precision"], 4), "Recall": round(s["recall"], 4),
                     "F1": round(s["f1"], 4), "Fraud Caught": s["tp"], "Fraud Missed": s["fn"], "False Alarms": s["fp"]})
    return pd.DataFrame(rows)


def best_f1_threshold(y_true, proba) -> float:
    """Threshold with the best F1 (use VALIDATION data, never the test set)."""
    grid = np.round(np.arange(0.02, 0.99, 0.01), 2)
    f1s = [f1_score(y_true, (np.asarray(proba) >= t).astype(int), zero_division=0) for t in grid]
    return float(grid[int(np.argmax(f1s))])


def plot_pr_roc(y_true, probas: dict, path) -> None:
    fig, ax = plt.subplots(1, 2, figsize=(14, 5.5))
    for name, p in probas.items():
        prec, rec, _ = precision_recall_curve(y_true, p)
        ax[0].plot(rec, prec, lw=2, label=f"{name} (AP={average_precision_score(y_true, p):.2f})")
        fpr, tpr, _ = roc_curve(y_true, p)
        ax[1].plot(fpr, tpr, lw=2, label=name)
    ax[0].axhline(np.mean(y_true), color="grey", ls="--", label="random")
    ax[0].set(xlabel="Recall", ylabel="Precision", title="Precision-Recall curves (validation)")
    ax[0].legend(fontsize=7)
    ax[1].plot([0, 1], [0, 1], "k--")
    ax[1].set(xlabel="False positive rate", ylabel="True positive rate", title="ROC curves (validation)")
    ax[1].legend(fontsize=7)
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close(fig)


def plot_confusion(y_true, proba, thr, path, title="Confusion matrix") -> None:
    cm = confusion_matrix(y_true, (np.asarray(proba) >= thr).astype(int), labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ConfusionMatrixDisplay(cm, display_labels=["Legit", "Fraud"]).plot(ax=ax, cmap="Blues", values_format="d", colorbar=False)
    ax.set_title(f"{title} (threshold {thr})")
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close(fig)


def plot_threshold_curve(table: pd.DataFrame, path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    for col in ("Precision", "Recall", "F1"):
        ax.plot(table["Threshold"], table[col], "o-", label=col)
    ax.set(xlabel="Threshold", title="Metrics vs threshold (validation)")
    ax.legend()
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close(fig)
