from datetime import datetime, timezone
import json
from pathlib import Path
import numpy as np
import pandas as pd

from process import TARGET_COL, get_project_root

# Default paths for reference data, drift snapshots, and run logs
DEFAULT_REFERENCE_PATH = get_project_root() / "data" / "processed" / "telco_churn_processed.csv"
DEFAULT_SNAPSHOTS_DIR = get_project_root() / "snapshots"
DEFAULT_LOGS_DIR = DEFAULT_SNAPSHOTS_DIR / "logs"


def compute_psi(expected: pd.Series, actual: pd.Series, bins: int = 10) -> float:
    # Calculate Population Stability Index (PSI) for a numeric feature
    exp_clean = pd.to_numeric(expected, errors="coerce").dropna().to_numpy()
    act_clean = pd.to_numeric(actual, errors="coerce").dropna().to_numpy()

    if len(exp_clean) == 0 or len(act_clean) == 0:
        return 0.0

    # Build quantile bin edges from reference data
    quantiles = np.linspace(0, 100, bins + 1)
    edges = np.unique(np.percentile(exp_clean, quantiles))
    if len(edges) < 2:
        return 0.0

    exp_counts, _ = np.histogram(exp_clean, bins=edges)
    act_counts, _ = np.histogram(act_clean, bins=edges)

    # Smooth zero counts slightly to avoid division/log errors
    eps = 1e-6
    exp_pct = np.clip(exp_counts / max(len(exp_clean), 1), eps, None)
    act_pct = np.clip(act_counts / max(len(act_clean), 1), eps, None)

    psi = np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct))
    return round(float(psi), 4)


def detect_data_drift(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    psi_threshold: float = 0.2,
) -> dict:
    # Compare numeric feature distributions between reference and current datasets
    feature_cols = [
        col for col in reference_df.columns
        if col != TARGET_COL and col in current_df.columns
    ]
    numeric_cols = [
        col for col in feature_cols
        if pd.api.types.is_numeric_dtype(reference_df[col])
    ]

    drift_report = {}
    drifted_features = []

    for col in numeric_cols:
        psi_score = compute_psi(reference_df[col], current_df[col])
        is_drifted = psi_score >= psi_threshold
        drift_report[col] = {
            "psi": psi_score,
            "drifted": is_drifted,
        }
        if is_drifted:
            drifted_features.append(col)

    return {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "reference_rows": int(len(reference_df)),
        "current_rows": int(len(current_df)),
        "psi_threshold": psi_threshold,
        "any_drift_detected": len(drifted_features) > 0,
        "drifted_features": drifted_features,
        "features": drift_report,
    }


def save_drift_snapshot(
    report: dict,
    output_dir: str | Path | None = None,
    filename: str = "drift_report.json",
) -> Path:
    # Save latest or named drift report JSON under snapshots/
    snap_dir = Path(output_dir) if output_dir else DEFAULT_SNAPSHOTS_DIR
    snap_dir.mkdir(parents=True, exist_ok=True)
    out_path = snap_dir / filename
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return out_path


def write_monitor_log(
    report: dict,
    logs_dir: str | Path | None = None,
    status: str = "OK",
    message: str = "",
) -> Path:
    # Write a timestamped monitoring log file under snapshots/logs/
    log_dir = Path(logs_dir) if logs_dir else DEFAULT_LOGS_DIR
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp_slug = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"monitor_{timestamp_slug}.log"

    drifted = ", ".join(report.get("drifted_features", [])) or "none"
    log_lines = [
        f"timestamp_utc : {report.get('checked_at_utc', datetime.now(timezone.utc).isoformat())}",
        f"status        : {status}",
        f"ref_rows      : {report.get('reference_rows', 'n/a')}",
        f"curr_rows     : {report.get('current_rows', 'n/a')}",
        f"psi_threshold : {report.get('psi_threshold', 0.2)}",
        f"drift_detected: {report.get('any_drift_detected', False)}",
        f"drifted_cols  : {drifted}",
    ]
    if message:
        log_lines.append(f"message       : {message}")

    log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    return log_path


# Run baseline drift check, save snapshot, and write run log when executed directly
if __name__ == "__main__":
    if DEFAULT_REFERENCE_PATH.exists():
        ref_df = pd.read_csv(DEFAULT_REFERENCE_PATH)
        report = detect_data_drift(ref_df, ref_df)
        snapshot_file = save_drift_snapshot(report)
        log_file = write_monitor_log(report, status="OK", message="Baseline self-check completed")
        print(f"Saved drift snapshot to : {snapshot_file}")
        print(f"Saved monitor run log to: {log_file}")
        print("Any drift detected      :", report["any_drift_detected"])
    else:
        missing_msg = f"Reference dataset not found at {DEFAULT_REFERENCE_PATH}"
        fallback_report = {
            "checked_at_utc": datetime.now(timezone.utc).isoformat(),
            "any_drift_detected": False,
            "drifted_features": [],
        }
        log_file = write_monitor_log(fallback_report, status="ERROR", message=missing_msg)
        print(missing_msg)
        print(f"Wrote error log to: {log_file}")