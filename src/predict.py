import json
from pathlib import Path
import joblib
import pandas as pd

from process import TARGET_COL, clean_and_engineer_features, get_project_root

DEFAULT_MODEL_PATH = get_project_root() / "artifacts" / "churn_model.joblib"
DEFAULT_SCHEMA_PATH = get_project_root() / "artifacts" / "feature_schema.json"


def load_model_and_schema(
    model_path: str | Path | None = None,
    schema_path: str | Path | None = None,
):
    # Load serialized scikit-learn pipeline and expected feature columns
    m_path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
    s_path = Path(schema_path) if schema_path else DEFAULT_SCHEMA_PATH

    if not m_path.exists():
        raise FileNotFoundError(f"Model artifact not found at: {m_path}. Run src/train.py first.")

    model = joblib.load(m_path)
    feature_columns = None
    if s_path.exists():
        schema = json.loads(s_path.read_text(encoding="utf-8"))
        feature_columns = schema.get("feature_columns")

    return model, feature_columns


def prepare_inference_df(records: list[dict] | pd.DataFrame, feature_columns: list[str] | None = None) -> pd.DataFrame:
    # Convert raw input records/DataFrame into cleaned engineered features
    df = records.copy() if isinstance(records, pd.DataFrame) else pd.DataFrame(records)
    if df.empty:
        raise ValueError("Input records for prediction cannot be empty.")

    # Drop target column if caller accidentally included it
    if TARGET_COL in df.columns:
        df = df.drop(columns=[TARGET_COL])

    cleaned = clean_and_engineer_features(df)

    # Align columns to training schema so missing optional fields remain NaN
    if feature_columns:
        for col in feature_columns:
            if col not in cleaned.columns:
                cleaned[col] = pd.NA
        cleaned = cleaned[feature_columns]

    return cleaned


def predict_churn(
    records: list[dict] | pd.DataFrame,
    threshold: float = 0.5,
    model_path: str | Path | None = None,
    schema_path: str | Path | None = None,
) -> list[dict]:
    # Run churn probability and binary class predictions for one or more customers
    model, feature_columns = load_model_and_schema(model_path=model_path, schema_path=schema_path)
    X_infer = prepare_inference_df(records, feature_columns=feature_columns)

    probabilities = model.predict_proba(X_infer)[:, 1]
    results = []
    for prob in probabilities:
        churn_prob = float(prob)
        churn_pred = int(churn_prob >= threshold)
        results.append(
            {
                "churn_probability": round(churn_prob, 4),
                "churn_prediction": churn_pred,
                "churn_label": "Yes" if churn_pred == 1 else "No",
            }
        )
    return results


# Smoke test when executed directly
if __name__ == "__main__":
    sample_customer = [
        {
            "customerID": "0001-SAMPLE",
            "gender": "Female",
            "SeniorCitizen": 0,
            "Partner": "Yes",
            "Dependents": "No",
            "tenure": 3,
            "PhoneService": "Yes",
            "MultipleLines": "No",
            "InternetService": "Fiber optic",
            "OnlineSecurity": "No",
            "OnlineBackup": "No",
            "DeviceProtection": "No",
            "TechSupport": "No",
            "StreamingTV": "Yes",
            "StreamingMovies": "No",
            "Contract": "Month-to-month",
            "PaperlessBilling": "Yes",
            "PaymentMethod": "Electronic check",
            "MonthlyCharges": 85.0,
            "TotalCharges": "255.0",
        }
    ]
    print(json.dumps(predict_churn(sample_customer), indent=2))