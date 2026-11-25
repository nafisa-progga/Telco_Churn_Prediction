from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Target column and identifier columns to remove
TARGET_COL = "Churn"
DROP_COLS = ["customerID"]  # Add "gender" here if you choose to exclude it


def get_project_root() -> Path:
    # Get the Telco_Churn_Prediction root directory
    return Path(__file__).resolve().parent.parent


def load_raw_data(csv_path: str | Path | None = None) -> pd.DataFrame:
    # Load the untouched raw Telco Customer Churn dataset
    default_path = get_project_root() / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
    return pd.read_csv(csv_path or default_path)


def clean_and_engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    # Clean types/missing values and add simple domain features without mutating input
    data = df.copy()

    # Drop identifier column if present
    cols_to_drop = [col for col in DROP_COLS if col in data.columns]
    if cols_to_drop:
        data = data.drop(columns=cols_to_drop)

    # Convert TotalCharges from string (with blanks) to numeric
    if "TotalCharges" in data.columns:
        data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce")

    # Feature engineering: average spend per month & tenure buckets
    if {"TotalCharges", "tenure"}.issubset(data.columns):
        safe_tenure = data["tenure"].replace(0, np.nan)
        data["avg_monthly_spend"] = (data["TotalCharges"] / safe_tenure).fillna(
            data.get("MonthlyCharges", 0.0)
        )

    if "tenure" in data.columns:
        data["tenure_group"] = pd.cut(
            data["tenure"],
            bins=[-1, 12, 24, 48, 120],
            labels=["0-1yr", "1-2yr", "2-4yr", "4yr+"],
        ).astype(str)

    # Encode binary target ("No" -> 0, "Yes" -> 1) if present
    if TARGET_COL in data.columns:
        if data[TARGET_COL].dtype == object or str(data[TARGET_COL].dtype) == "string":
            mapped = data[TARGET_COL].astype(str).str.strip().map({"No": 0, "Yes": 1})
            # Keep original values if already numeric strings like "0"/"1"
            data[TARGET_COL] = mapped.fillna(pd.to_numeric(data[TARGET_COL], errors="coerce"))

    return data


def build_preprocessor(features_df: pd.DataFrame) -> ColumnTransformer:
    # Create preprocessing pipeline for numeric and categorical features
    feature_cols = [col for col in features_df.columns if col != TARGET_COL]
    X = features_df[feature_cols]

    # Separate numeric and categorical feature columns
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

    # Impute missing values and scale numeric columns
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    # Impute missing values and one-hot encode categorical columns
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    # Combine numeric and categorical transformers
    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )


def save_processed_data(
    df: pd.DataFrame,
    filename: str = "telco_churn_processed.csv",
    output_dir: str | Path | None = None,
) -> Path:
    # Save processed dataset under data/processed/
    target_dir = Path(output_dir) if output_dir else get_project_root() / "data" / "processed"
    target_dir.mkdir(parents=True, exist_ok=True)
    output_path = target_dir / filename
    df.to_csv(output_path, index=False)
    return output_path


def preview_processed_data(
    df: pd.DataFrame,
    n_cols: int | None = None,
    n_rows: int = 5,
) -> pd.DataFrame:
    # Select all columns by default, or the first n_cols if specified
    preview = df.iloc[:n_rows, :] if n_cols is None else df.iloc[:n_rows, :n_cols]
    col_label = "all columns" if n_cols is None else f"first {n_cols} columns"

    # Print each row cleanly with one column per line
    print(f"\nFirst {n_rows} rows ({col_label}) of processed dataset (shape: {df.shape}):")
    for idx, row in preview.iterrows():
        print(f"\nRow {idx}:")
        for col, val in row.items():
            print(f"  {col:<20}: {val}")

    return preview


# Run processing, save output, and print preview
if __name__ == "__main__":
    raw_df = load_raw_data()
    processed_df = clean_and_engineer_features(raw_df)
    saved_file = save_processed_data(processed_df)
    print(f"Raw rows: {len(raw_df)} | Saved processed dataset to: {saved_file}")
    preview_processed_data(processed_df, n_rows=5)