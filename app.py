"""Streamlit dashboard.   Run:  streamlit run app.py"""
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, precision_score, recall_score

from src.config import DEFAULT_BUSINESS, RISK_LABELS, ROOT, SAMPLE_PATH
from src.predict import load_bundle, score_dataframe
from src.utils import top_k_table

st.set_page_config(page_title="Fraud Risk Monitor", page_icon="💳", layout="wide")


@st.cache_resource
def get_bundle():
    return load_bundle()


try:
    bundle = get_bundle()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

st.title("💳 Credit Card Fraud Risk Monitor")
st.caption("Educational demo built on the Kaggle creditcard.csv data. Scores are model estimates, not real bank decisions.")

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Data")
    uploaded = st.file_uploader("Upload transactions (CSV, same columns as creditcard.csv)", type="csv")
    st.header("Detection")
    threshold = st.slider("Fraud threshold", 0.01, 0.99, float(bundle["threshold"]), 0.01,
                          help="Lower = more fraud caught but more false alarms")
    st.header("Business assumptions")
    avg_loss = st.number_input("Average fraud loss ($)", 0.0, 100000.0, DEFAULT_BUSINESS["avg_fraud_loss"], 10.0)
    alert_cost = st.number_input("Cost per investigated alert ($)", 0.0, 1000.0, DEFAULT_BUSINESS["alert_cost"], 1.0)
    prevent = st.slider("Share of detected fraud prevented", 0.0, 1.0, DEFAULT_BUSINESS["prevent_rate"], 0.05)

# ---------------------------------------------------------------- data
if uploaded is not None:
    raw, source = pd.read_csv(uploaded), "your upload"
elif SAMPLE_PATH.exists():
    raw, source = pd.read_csv(SAMPLE_PATH), "the built-in demo sample"
else:
    st.info("Upload a CSV in the sidebar, or run `python -m src.train` to create the demo sample.")
    st.stop()

try:
    scored = score_dataframe(raw, threshold, bundle)
except ValueError as e:
    st.error(f"Cannot score this file: {e}")
    st.stop()

scored = scored.rename(columns={"amount": "Amount ($)", "fraud_probability": "Fraud probability",
                                "risk_level": "Risk", "decision": "Decision", "transaction": "Transaction"})
has_labels = "Class" in raw.columns
if has_labels:
    scored["Actually fraud"] = np.where(raw["Class"].to_numpy() == 1, "YES", "no")
flagged = scored[scored["Decision"] == "Flag for review"]

tab_alerts, tab_perf, tab_model = st.tabs(["🚨 Alerts", "📈 Performance & business value", "🧠 Model"])

# ---------------------------------------------------------------- alerts tab
with tab_alerts:
    st.subheader(f"Results for {source}")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Transactions", f"{len(scored):,}")
    c2.metric("Flagged", f"{len(flagged):,}")
    c3.metric("Flagged amount", f"${flagged['Amount ($)'].sum():,.0f}")
    c4.metric("Critical risk", f"{(scored['Risk'] == 'Critical').sum():,}")

    st.bar_chart(scored["Risk"].value_counts().reindex(RISK_LABELS).fillna(0))

    f1c, f2c, f3c = st.columns(3)
    risks = f1c.multiselect("Risk levels", RISK_LABELS, default=RISK_LABELS)
    min_p = f2c.slider("Minimum probability", 0.0, 1.0, 0.0, 0.05)
    amt_hi = float(max(scored["Amount ($)"].max(), 1.0))
    amt_range = f3c.slider("Amount range ($)", 0.0, amt_hi, (0.0, amt_hi))
    view = scored[scored["Risk"].isin(risks) & (scored["Fraud probability"] >= min_p)
                  & scored["Amount ($)"].between(*amt_range)].sort_values("Fraud probability", ascending=False)
    st.caption(f"{len(view):,} transactions match the filters (top 500 shown)")
    st.dataframe(view.head(500), width="stretch", hide_index=True)
    st.download_button("Download flagged transactions (CSV)",
                       flagged.sort_values("Fraud probability", ascending=False).to_csv(index=False).encode(),
                       file_name="flagged_transactions.csv", mime="text/csv")

# ---------------------------------------------------------------- performance tab
with tab_perf:
    if not has_labels:
        st.info("This file has no `Class` column, so performance cannot be measured. Scores and alerts are still shown.")
    else:
        y = raw["Class"].to_numpy()
        proba = scored["Fraud probability"].to_numpy()
        pred = (proba >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        st.subheader("Detection quality at the current threshold")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Recall", f"{recall_score(y, pred, zero_division=0):.1%}", f"{tp} of {tp + fn} frauds caught")
        m2.metric("Precision", f"{precision_score(y, pred, zero_division=0):.1%}", f"{fp} false alarms")
        m3.metric("F1", f"{f1_score(y, pred, zero_division=0):.3f}")
        m4.metric("PR-AUC", f"{average_precision_score(y, proba):.3f}" if y.sum() else "n/a")
        st.dataframe(pd.DataFrame([[tn, fp], [fn, tp]], index=["Actual legit", "Actual fraud"],
                                  columns=["Predicted legit", "Predicted fraud"]), width="content")

        st.subheader("Business simulation (assumptions in the sidebar)")
        prevented, cost = tp * avg_loss * prevent, len(flagged) * alert_cost
        b1, b2, b3, b4 = st.columns(4)
        b1.metric("Losses prevented", f"${prevented:,.0f}")
        b2.metric("Investigation cost", f"${cost:,.0f}")
        b3.metric("Net benefit", f"${prevented - cost:,.0f}")
        b4.metric("Loss with no model", f"${y.sum() * avg_loss:,.0f}")
        st.subheader("Top-K prioritisation")
        st.caption("If analysts can only check the K highest-risk transactions, how many frauds are found?")
        st.dataframe(top_k_table(y, proba, ks=(25, 50, 100, 200)), width="content", hide_index=True)

# ---------------------------------------------------------------- model tab
with tab_model:
    cfg = bundle["config"]
    st.subheader("Deployed model")
    st.write(f"**Chosen experiment:** {cfg['experiment']}  \n"
             f"**Model:** {cfg['model']} · class weights: {cfg['class_weight']} · SMOTE: {cfg['smote']} · split: {cfg.get('split', 'random')}  \n"
             f"**Saved threshold:** {bundle['threshold']}")
    tm = bundle["test_metrics"]
    t1, t2, t3, t4 = st.columns(4)
    t1.metric("Test recall", f"{tm['recall']:.1%}")
    t2.metric("Test precision", f"{tm['precision']:.1%}")
    t3.metric("Test PR-AUC", f"{tm['pr_auc']:.3f}")
    t4.metric("Test ROC-AUC", f"{tm['roc_auc']:.3f}")
    results_csv = ROOT / "reports" / "experiment_results.csv"
    if results_csv.exists():
        st.subheader("All experiments (validation set)")
        st.dataframe(pd.read_csv(results_csv), width="stretch", hide_index=True)
    fig_dir = ROOT / "reports" / "figures"
    for img, cap in (("pr_roc_curves.png", "PR and ROC curves"), ("confusion_matrix_test.png", "Test confusion matrix"),
                     ("threshold_curve.png", "Metrics vs threshold")):
        if (fig_dir / img).exists():
            st.image(str(fig_dir / img), caption=cap)
