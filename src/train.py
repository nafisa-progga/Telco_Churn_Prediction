import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from process import (
    TARGET_COL,
    build_preprocessor,
    clean_and_engineer_features,
    get_project_root,
)

# Default paths and training configuration
DEFAULT_PROCESSED_PATH = get_project_root() / "data" / "processed" / "telco_churn_processed.csv"
DEFAULT_ARTIFACTS_DIR = get_project_root() / "artifacts"
RANDOM_STATE = 42
TEST_SIZE = 0.2


def load_training_data(processed_path: str | Path | None = None) -> pd.DataFrame:
    # Load processed CSV if present; otherwise build it from raw data
    path = Path(processed_path) if processed_path else DEFAULT_PROCESSED_PATH
    if path.exists():
        df = pd.read_csv(path)
    else:
        raw_path = get_project_root() / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
        df = clean_and_engineer_features(pd.read_csv(raw_path))

    # Ensure target column is clean numeric 0/1 without missing labels
    if TARGET_COL not in df.columns:
        raise ValueError(f"Missing target column '{TARGET_COL}' in dataset.")

    if df[TARGET_COL].dtype == object or str(df[TARGET_COL].dtype) == "string":
        mapped = df[TARGET_COL].astype(str).str.strip().map({"No": 0, "Yes": 1})
        df[TARGET_COL] = mapped.fillna(pd.to_numeric(df[TARGET_COL], errors="coerce"))

    df = df.dropna(subset=[TARGET_COL]).copy()
    df[TARGET_COL] = df[TARGET_COL].astype(int)
    return df


def split_features_and_target(df: pd.DataFrame):
    # Separate input features (X) and binary target (y)
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    return train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )


def evaluate_model(pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    # Compute standard classification and ranking metrics on the test split
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    return {
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, y_proba)), 4),
    }


def train_and_select_model(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series):
    # Train candidate pipelines and pick the best model by ROC-AUC
    candidates = {
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_leaf=4,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }

    best_name = ""
    best_pipeline = None
    best_metrics = None
    all_metrics = {}

    for model_name, estimator in candidates.items():
        # Build fresh preprocessor per candidate to avoid shared mutable state
        preprocessor = build_preprocessor(pd.concat([X_train, y_train], axis=1))
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", estimator),
            ]
        )
        pipeline.fit(X_train, y_train)
        metrics = evaluate_model(pipeline, X_test, y_test)
        all_metrics[model_name] = metrics

        # Select model with highest ROC-AUC score
        if best_metrics is None or metrics["roc_auc"] > best_metrics["roc_auc"]:
            best_name = model_name
            best_pipeline = pipeline
            best_metrics = metrics

    return best_name, best_pipeline, best_metrics, all_metrics


def save_artifacts(
    best_name: str,
    pipeline: Pipeline,
    best_metrics: dict,
    all_metrics: dict,
    feature_columns: list[str],
    artifacts_dir: str | Path | None = None,
) -> dict[str, Path]:
    # Persist trained model pipeline, metrics report, and feature metadata
    out_dir = Path(artifacts_dir) if artifacts_dir else DEFAULT_ARTIFACTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    model_path = out_dir / "churn_model.joblib"
    metrics_path = out_dir / "metrics.json"
    schema_path = out_dir / "feature_schema.json"

    joblib.dump(pipeline, model_path)

    report = {
        "selected_model": best_name,
        "test_metrics": best_metrics,
        "all_models": all_metrics,
    }
    metrics_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    schema = {
        "target_col": TARGET_COL,
        "feature_columns": feature_columns,
    }
    schema_path.write_text(json.dumps(schema, indent=2), encoding="utf-8")

    return {
        "model": model_path,
        "metrics": metrics_path,
        "schema": schema_path,
    }


# Run training pipeline when executed directly
if __name__ == "__main__":
    df = load_training_data()
    X_train, X_test, y_train, y_test = split_features_and_target(df)
    best_name, best_pipeline, best_metrics, all_metrics = train_and_select_model(
        X_train, y_train, X_test, y_test
    )
    saved_paths = save_artifacts(
        best_name=best_name,
        pipeline=best_pipeline,
        best_metrics=best_metrics,
        all_metrics=all_metrics,
        feature_columns=list(X_train.columns),
    )

    print(f"Selected model: {best_name}")
    print("Test metrics:", json.dumps(best_metrics, indent=2))
    print("Saved artifacts:", {k: str(v) for k, v in saved_paths.items()})