"""Exploratory charts saved to reports/figures.   Usage:  python -m src.eda"""
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from src.config import DATA_PATH, FIGURES_DIR, TARGET, V_COLS
from src.data_loader import load_data

COL = {0: "#2a9d8f", 1: "#e63946"}


def run(path=DATA_PATH, out_dir=FIGURES_DIR):
    out_dir.mkdir(parents=True, exist_ok=True)
    df = load_data(path)
    sns.set_theme(style="whitegrid")
    counts = df[TARGET].value_counts().sort_index()
    n, n_fraud = len(df), int(counts.get(1, 0))
    print(f"Transactions: {n:,} | fraud: {n_fraud:,} ({n_fraud / n * 100:.3f}%) | duplicates: {int(df.duplicated().sum())}")
    print(f"Accuracy of 'always legitimate': {counts[0] / n * 100:.2f}% (catches 0 fraud)")

    # class balance
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.bar(["Legitimate", "Fraud"], counts.values, color=[COL[0], COL[1]])
    ax.set_yscale("log"); ax.set_title("Transactions per class (log scale)")
    for i, v in enumerate(counts.values):
        ax.text(i, v, f"{v:,}", ha="center", va="bottom", fontweight="bold")
    plt.tight_layout(); plt.savefig(out_dir / "class_distribution.png", dpi=150); plt.close(fig)

    # amount
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    for c, lab in [(0, "Legitimate"), (1, "Fraud")]:
        sns.histplot(np.log1p(df.loc[df[TARGET] == c, "Amount"]), bins=50, stat="density", color=COL[c], alpha=0.5, label=lab, ax=ax[0])
    ax[0].set_title("log(1 + Amount)"); ax[0].legend()
    sns.boxplot(data=df.assign(logAmount=np.log1p(df["Amount"])), x=TARGET, y="logAmount", hue=TARGET, palette=[COL[0], COL[1]], legend=False, ax=ax[1])
    ax[1].set_title("Amount by class")
    plt.tight_layout(); plt.savefig(out_dir / "amount_distribution.png", dpi=150); plt.close(fig)

    # time
    hour = ((df["Time"] // 3600) % 24).astype(int)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    for c, lab in [(0, "Legitimate"), (1, "Fraud")]:
        ax[0].hist(hour[df[TARGET] == c], bins=24, range=(0, 24), density=True, alpha=0.55, color=COL[c], label=lab)
    ax[0].set_title("Hour of day"); ax[0].legend()
    ax[1].bar(*zip(*(df.groupby(hour)[TARGET].mean() * 100).items()), color=COL[1]); ax[1].set_title("Fraud rate (%) by hour")
    plt.tight_layout(); plt.savefig(out_dir / "time_distribution.png", dpi=150); plt.close(fig)

    # correlation with the target
    corr = df[V_COLS + ["Amount"]].corrwith(df[TARGET]).sort_values()
    fig, ax = plt.subplots(figsize=(7, 8))
    ax.barh(corr.index, corr.values, color=[COL[1] if v > 0 else "#457b9d" for v in corr.values])
    ax.set_title("Correlation with Class (fraud = 1)")
    plt.tight_layout(); plt.savefig(out_dir / "correlation_with_class.png", dpi=150); plt.close(fig)
    print("Figures saved to", out_dir)


if __name__ == "__main__":
    try:
        run()
    except (FileNotFoundError, ValueError) as e:
        sys.exit(str(e))
