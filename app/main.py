import os
from typing import Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.predict import predict_churn

# Read service settings from environment variables with sensible defaults
APP_ENV = os.getenv("APP_ENV", "dev")
DEFAULT_THRESHOLD = float(os.getenv("CHURN_THRESHOLD", "0.5"))

app = FastAPI(
    title="Telco Customer Churn API",
    version="1.0.0",
    description="Inference API for Telco Customer Churn prediction.",
)


class CustomerFeatures(BaseModel):
    # Accept flexible raw Telco fields while validating core types
    customerID: str | None = None
    gender: str | None = None
    SeniorCitizen: int | None = 0
    Partner: str | None = None
    Dependents: str | None = None
    tenure: float | None = Field(default=0, ge=0)
    PhoneService: str | None = None
    MultipleLines: str | None = None
    InternetService: str | None = None
    OnlineSecurity: str | None = None
    OnlineBackup: str | None = None
    DeviceProtection: str | None = None
    TechSupport: str | None = None
    StreamingTV: str | None = None
    StreamingMovies: str | None = None
    Contract: str | None = None
    PaperlessBilling: str | None = None
    PaymentMethod: str | None = None
    MonthlyCharges: float | None = Field(default=0.0, ge=0)
    TotalCharges: str | float | None = 0.0

    model_config = {"extra": "allow"}


class PredictRequest(BaseModel):
    customers: list[CustomerFeatures] = Field(..., min_length=1)
    threshold: float | None = Field(default=None, gt=0.0, lt=1.0)


@app.get("/health")
def health_check() -> dict[str, Any]:
    # Lightweight health endpoint for containers and CI/CD smoke checks
    return {"status": "ok", "env": APP_ENV}


@app.post("/predict")
def predict_endpoint(payload: PredictRequest) -> dict[str, Any]:
    # Convert validated request models to plain dicts for inference
    records = [item.model_dump() for item in payload.customers]
    active_threshold = payload.threshold if payload.threshold is not None else DEFAULT_THRESHOLD

    try:
        predictions = predict_churn(records=records, threshold=active_threshold)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        "count": len(predictions),
        "threshold": active_threshold,
        "predictions": predictions,
    }