import io

import pytest
from fastapi.testclient import TestClient

from src import predict as predict_module
from api.main import app


@pytest.fixture()
def client(trained, monkeypatch):
    monkeypatch.setenv("FRAUD_MODEL_PATH", str(trained["root"] / "models" / "best_model.pkl"))
    predict_module._CACHE.clear()
    return TestClient(app)


def _tx(df, i=0):
    return df.drop(columns=["Class"]).iloc[i].to_dict()


def test_health_ok(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["model_loaded"] is True


def test_health_reports_missing_model(monkeypatch, tmp_path):
    monkeypatch.setenv("FRAUD_MODEL_PATH", str(tmp_path / "nope.pkl"))
    predict_module._CACHE.clear()
    c = TestClient(app)
    assert c.get("/health").json()["model_loaded"] is False
    assert c.post("/predict", json={}).status_code in (422, 503)


def test_model_info(client):
    r = client.get("/model-info")
    assert r.status_code == 200 and "test_metrics" in r.json()


def test_predict_single(client, fake_df):
    r = client.post("/predict", json=_tx(fake_df))
    assert r.status_code == 200
    body = r.json()
    assert 0 <= body["fraud_probability"] <= 1
    assert body["decision"] in ("Flag for review", "Allow")


def test_predict_threshold_override(client, fake_df):
    fraud_row = _tx(fake_df[fake_df["Class"] == 1].reset_index(drop=True))
    low = client.post("/predict?threshold=0.0", json=fraud_row).json()
    high = client.post("/predict?threshold=1.0", json=fraud_row).json()
    assert low["decision"] == "Flag for review"
    assert high["decision"] == "Allow" or high["fraud_probability"] >= 1.0


def test_predict_rejects_bad_input(client, fake_df):
    bad = _tx(fake_df)
    del bad["V3"]
    assert client.post("/predict", json=bad).status_code == 422
    bad2 = _tx(fake_df)
    bad2["Amount"] = -5
    assert client.post("/predict", json=bad2).status_code == 422
    assert client.post("/predict?threshold=2", json=_tx(fake_df)).status_code == 422


def test_predict_batch(client, fake_df):
    txs = [_tx(fake_df, i) for i in range(10)]
    r = client.post("/predict/batch", json={"transactions": txs})
    assert r.status_code == 200
    body = r.json()
    assert body["n_transactions"] == 10 and len(body["results"]) == 10


def test_predict_csv_upload(client, fake_df):
    csv = fake_df.drop(columns=["Class"]).head(40).to_csv(index=False)
    r = client.post("/predict/csv", files={"file": ("tx.csv", io.BytesIO(csv.encode()), "text/csv")})
    assert r.status_code == 200 and r.json()["n_transactions"] == 40
    r2 = client.post("/predict/csv?only_flagged=true", files={"file": ("tx.csv", io.BytesIO(csv.encode()), "text/csv")})
    assert len(r2.json()["results"]) == r2.json()["n_flagged"]


def test_predict_csv_with_missing_columns(client):
    r = client.post("/predict/csv", files={"file": ("tx.csv", io.BytesIO(b"Time,Amount\n1,2\n"), "text/csv")})
    assert r.status_code == 422 and "Missing columns" in r.json()["detail"]
