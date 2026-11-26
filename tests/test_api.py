from fastapi.testclient import TestClient

from app.main import app
from src import predict as predict_module

client = TestClient(app)


def _sample_customer_payload() -> dict:
    # Build valid request payload matching PredictRequest schema
    return {
        "customers": [
            {
                "customerID": "2001-TEST",
                "gender": "Female",
                "SeniorCitizen": 0,
                "Partner": "Yes",
                "Dependents": "No",
                "tenure": 6,
                "PhoneService": "Yes",
                "MultipleLines": "No",
                "InternetService": "DSL",
                "OnlineSecurity": "No",
                "OnlineBackup": "Yes",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "No",
                "StreamingMovies": "No",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 55.0,
                "TotalCharges": "330.0",
            }
        ],
        "threshold": 0.5,
    }


def test_health_endpoint():
    # Verify /health returns HTTP 200 and expected status payload
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert "env" in body


def test_predict_endpoint_returns_predictions(monkeypatch):
    # Mock predict_churn so API tests run fast without requiring a saved model artifact
    def fake_predict_churn(records, threshold=0.5, model_path=None, schema_path=None):
        return [
            {
                "churn_probability": 0.72,
                "churn_prediction": 1,
                "churn_label": "Yes",
            }
            for _ in records
        ]

    monkeypatch.setattr("app.main.predict_churn", fake_predict_churn)

    response = client.post("/predict", json=_sample_customer_payload())
    assert response.status_code == 200
    body = response.json()
    assert body["count"] == 1
    assert body["threshold"] == 0.5
    assert body["predictions"][0]["churn_label"] == "Yes"


def test_predict_endpoint_rejects_empty_customers_list():
    # Verify empty customer list fails request validation with HTTP 422
    response = client.post("/predict", json={"customers": []})
    assert response.status_code == 422