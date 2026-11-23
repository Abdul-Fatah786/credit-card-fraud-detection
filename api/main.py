"""FastAPI service for fraud scoring.

Run (from the project root):
    uvicorn api.main:app --reload
Then open http://127.0.0.1:8000/docs for the interactive documentation.
"""
import pandas as pd
from fastapi import FastAPI, File, HTTPException, Query, UploadFile

from api.schemas import BatchRequest, BatchResponse, HealthResponse, Prediction, Transaction
from src.config import SAMPLE_PATH
from src.predict import load_bundle, predict_one, score_dataframe

MAX_CSV_ROWS = 100_000

app = FastAPI(
    title="Credit Card Fraud Detection API",
    version="1.0.0",
    description="Scores credit card transactions with a trained model. Educational project, not a real bank system.",
)


def _bundle():
    try:
        return load_bundle()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.get("/", tags=["info"])
def root():
    return {"service": "Credit Card Fraud Detection API", "docs": "/docs", "health": "/health"}


@app.get("/health", response_model=HealthResponse, tags=["info"])
def health():
    try:
        b = load_bundle()
        return HealthResponse(status="ok", model_loaded=True, threshold=float(b["threshold"]))
    except FileNotFoundError as e:
        return HealthResponse(status="model missing", model_loaded=False, detail=str(e))


@app.get("/model-info", tags=["info"])
def model_info():
    b = _bundle()
    return {"config": b["config"], "threshold": b["threshold"], "test_metrics": b["test_metrics"],
            "sklearn_version": b.get("sklearn_version")}


@app.get("/sample", tags=["info"], summary="A real example transaction to paste into /predict")
def sample(fraud: bool = Query(False, description="Return a known fraud example if available")):
    if not SAMPLE_PATH.exists():
        raise HTTPException(404, "data/sample_transactions.csv not found. Run: python -m src.train")
    df = pd.read_csv(SAMPLE_PATH)
    pool = df[df["Class"] == (1 if fraud else 0)]
    if pool.empty:
        raise HTTPException(404, "No example of that class in the sample file")
    features = pool.drop(columns=["Class"]).reset_index(drop=True)
    row = features.iloc[0]
    if fraud:
        # pick the fraud example the model is most confident about, so the demo is easy to follow
        try:
            scored = score_dataframe(features, bundle=load_bundle())
            row = features.iloc[int(scored["fraud_probability"].idxmax())]
        except (FileNotFoundError, ValueError):
            pass
    return {k: float(v) for k, v in row.items()}


@app.post("/predict", response_model=Prediction, tags=["predict"])
def predict(tx: Transaction, threshold: float = Query(None, ge=0, le=1, description="Override the saved threshold")):
    return predict_one(tx.model_dump(), threshold, _bundle())


@app.post("/predict/batch", response_model=BatchResponse, tags=["predict"])
def predict_batch(req: BatchRequest, threshold: float = Query(None, ge=0, le=1)):
    b = _bundle()
    df = pd.DataFrame([t.model_dump() for t in req.transactions])
    out = score_dataframe(df, threshold, b)
    thr = float(b["threshold"] if threshold is None else threshold)
    return BatchResponse(n_transactions=len(out), n_flagged=int((out["decision"] == "Flag for review").sum()),
                         threshold=thr, results=out.to_dict("records"))


@app.post("/predict/csv", response_model=BatchResponse, tags=["predict"], summary="Upload a CSV in creditcard.csv format")
def predict_csv(file: UploadFile = File(...), threshold: float = Query(None, ge=0, le=1),
                only_flagged: bool = Query(False, description="Return only the flagged transactions")):
    b = _bundle()
    try:
        df = pd.read_csv(file.file)
    except Exception:
        raise HTTPException(400, "Could not read the upload as a CSV file")
    if len(df) > MAX_CSV_ROWS:
        raise HTTPException(413, f"Too many rows ({len(df):,}); the limit is {MAX_CSV_ROWS:,}")
    try:
        out = score_dataframe(df, threshold, b)
    except ValueError as e:
        raise HTTPException(422, str(e))
    thr = float(b["threshold"] if threshold is None else threshold)
    n_flagged = int((out["decision"] == "Flag for review").sum())
    if only_flagged:
        out = out[out["decision"] == "Flag for review"]
    return BatchResponse(n_transactions=len(df), n_flagged=n_flagged, threshold=thr, results=out.to_dict("records"))
