import pandas as pd

from src.monitor import compute_psi, detect_data_drift
from src.process import TARGET_COL, build_preprocessor, clean_and_engineer_features


def _sample_raw_df() -> pd.DataFrame:
    # Build sample raw Telco rows covering edge cases like blank TotalCharges and tenure=0
    return pd.DataFrame(
        [
            {
                "customerID": "1001-AAA",
                "gender": "Female",
                "SeniorCitizen": 0,
                "Partner": "Yes",
                "Dependents": "No",
                "tenure": 1,
                "PhoneService": "No",
                "MultipleLines": "No phone service",
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
                "MonthlyCharges": 29.85,
                "TotalCharges": "29.85",
                "Churn": "No",
            },
            {
                "customerID": "1002-BBB",
                "gender": "Male",
                "SeniorCitizen": 0,
                "Partner": "No",
                "Dependents": "No",
                "tenure": 0,
                "PhoneService": "Yes",
                "MultipleLines": "No",
                "InternetService": "Fiber optic",
                "OnlineSecurity": "No",
                "OnlineBackup": "No",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "Yes",
                "StreamingMovies": "Yes",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 95.50,
                "TotalCharges": " ",
                "Churn": "Yes",
            },
        ]
    )


def test_clean_and_engineer_features_drops_id_and_maps_target():
    # Verify identifier removal, engineered columns, and binary Churn encoding
    raw_df = _sample_raw_df()
    processed = clean_and_engineer_features(raw_df)

    assert "customerID" not in processed.columns
    assert "avg_monthly_spend" in processed.columns
    assert "tenure_group" in processed.columns
    assert processed[TARGET_COL].tolist() == [0, 1]


def test_clean_handles_blank_total_charges_and_zero_tenure():
    # Verify blank TotalCharges becomes NaN and tenure=0 falls back to MonthlyCharges
    raw_df = _sample_raw_df()
    processed = clean_and_engineer_features(raw_df)

    assert pd.isna(processed.loc[1, "TotalCharges"])
    assert processed.loc[1, "avg_monthly_spend"] == 95.50


def test_build_preprocessor_transforms_features():
    # Verify ColumnTransformer imputes, scales, and encodes without error
    processed = clean_and_engineer_features(_sample_raw_df())
    preprocessor = build_preprocessor(processed)

    X = processed.drop(columns=[TARGET_COL])
    transformed = preprocessor.fit_transform(X)

    assert transformed.shape[0] == len(X)
    assert transformed.shape[1] > 0


def test_drift_detection_on_identical_data():
    # Verify identical reference and current frames report zero drift
    df = clean_and_engineer_features(_sample_raw_df())
    report = detect_data_drift(df, df)

    assert report["any_drift_detected"] is False
    assert compute_psi(df["MonthlyCharges"], df["MonthlyCharges"]) == 0.0