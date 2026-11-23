"""Central settings: paths, column names, risk bands, business defaults."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data"
DATA_PATH = DATA_DIR / "creditcard.csv"              # download from Kaggle (see data/README.md)
SAMPLE_PATH = DATA_DIR / "sample_transactions.csv"   # small demo file written by src.train

MODELS_DIR = ROOT / "models"
BEST_MODEL_PATH = MODELS_DIR / "best_model.pkl"

REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

SEED = 42
TARGET = "Class"                                     # 1 = fraud, 0 = legitimate

V_COLS = [f"V{i}" for i in range(1, 29)]             # anonymised PCA features
SCALE_COLS = ["Amount", "Hour"]                      # columns that get standardised
FEATURES = V_COLS + SCALE_COLS                       # exact column order the model expects
REQUIRED_COLUMNS = ["Time", "Amount"] + V_COLS       # what an uploaded file must contain

# Risk bands used by the dashboard and the API (labels, not real bank policy)
RISK_BINS = [0.0, 0.2, 0.5, 0.8, 1.0]
RISK_LABELS = ["Low", "Medium", "High", "Critical"]

# Hypothetical business assumptions (editable in the Streamlit sidebar)
DEFAULT_BUSINESS = {"avg_fraud_loss": 200.0, "alert_cost": 5.0, "prevent_rate": 0.80}
