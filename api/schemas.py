"""Request / response models for the API."""
from typing import List, Optional

from pydantic import BaseModel, Field, create_model

from src.config import V_COLS

# One transaction in creditcard.csv format: Time, V1..V28, Amount.
_fields = {"Time": (float, Field(..., ge=0, description="Seconds since the first transaction in the dataset")),
           "Amount": (float, Field(..., ge=0, description="Transaction amount in dollars"))}
for _v in V_COLS:
    _fields[_v] = (float, Field(..., description="Anonymised PCA feature"))
Transaction = create_model("Transaction", **_fields)


class Prediction(BaseModel):
    fraud_probability: float = Field(..., ge=0, le=1)
    risk_level: str
    decision: str
    threshold: float


class BatchRequest(BaseModel):
    transactions: List[Transaction] = Field(..., min_length=1, max_length=10000)


class BatchItem(BaseModel):
    transaction: int
    amount: float
    fraud_probability: float
    risk_level: str
    decision: str


class BatchResponse(BaseModel):
    n_transactions: int
    n_flagged: int
    threshold: float
    results: List[BatchItem]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    detail: Optional[str] = None
    threshold: Optional[float] = None
