#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Backtest RF v2 model on the current comps dataset.

Compares:
  - actual projected_value_ma (from Excel)
  - RF v2 predicted ARV (exp(log_pred))

Outputs a CSV with row-level errors for inspection.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from lmc_estimator_ml.ml.trainer import load_model
from lmc_estimator_ml.ml.data_loader import load_comps
from lmc_estimator_ml.ml.features import preprocess_dataframe
from lmc_estimator_ml.ml.config import ARTIFACT_DIR, DEFAULT_EXCEL_PATH  # adjust if path constant is named differently

# ------------------------------
# Settings
# ------------------------------
TARGET_COL = "projected_value_ma"  # label column in your dataset
ARTIFACT_SUBDIR = "v2_rf"          # RF production model
DIAG_PATH = ARTIFACT_DIR / "diagnostics.json"  # has feature list

# Where to write the backtest CSV
OUTPUT_CSV = ARTIFACT_DIR / "v2_rf_backtest_vs_actual.csv"


def main() -> None:
    # 1) Load diagnostics to get feature list
    if not DIAG_PATH.exists():
        raise FileNotFoundError(f"diagnostics.json not found at {DIAG_PATH}")

    diag = json.loads(DIAG_PATH.read_text())
    feature_names = diag.get("features")
    if not feature_names:
        raise ValueError("No 'features' field found in diagnostics.json")

    print("[INFO] Using RF features:", feature_names)

    # 2) Load the comps dataset (same source as training)
    excel_path = Path(DEFAULT_EXCEL_PATH)
    if not excel_path.exists():
        raise FileNotFoundError(f"DEFAULT_EXCEL_PATH not found: {excel_path}")

    print(f"[INFO] Loading comps from: {excel_path}")
    df_raw = load_comps(excel_path)

    if TARGET_COL not in df_raw.columns:
        raise KeyError(f"Target column '{TARGET_COL}' not found in dataframe columns: {list(df_raw.columns)}")

    # 3) Apply the same pre-DataFrame transforms used in training
    df = preprocess_dataframe(df_raw)

    # Drop rows with missing target
    df = df.dropna(subset=[TARGET_COL]).copy()
    n_rows = len(df)
    print(f"[INFO] Rows with non-null {TARGET_COL}: {n_rows}")

    # 4) Build X and y
    missing_feats = [c for c in feature_names if c not in df.columns]
    if missing_feats:
        raise KeyError(f"The following required features are missing from the dataframe: {missing_feats}")

    X = df[feature_names].copy()
    y_actual = pd.to_numeric(df[TARGET_COL], errors="coerce")

    # 5) Load RF pipeline (same as app.py)
    print(f"[INFO] Loading RF model from ARTIFACT_DIR={ARTIFACT_DIR}")
    model = load_model(ARTIFACT_DIR)

    # 6) Predict log(ARV) then exponentiate to dollars
    log_pred = model.predict(X)
    y_pred = np.exp(log_pred.astype(float))

    # 7) Build comparison dataframe
    # Try to include some useful ID columns if present
    id_cols = [c for c in ["loan_number", "LoanNumber", "address", "street", "city", "state", "zipcode"] if c in df.columns]

    out = pd.DataFrame({
        "actual_arv": y_actual.values,
        "rf_v2_pred_arv": y_pred,
    })

    out["abs_error"] = out["rf_v2_pred_arv"] - out["actual_arv"]
    out["abs_error_dollars"] = out["abs_error"].abs()
    out["pct_error"] = out["abs_error"] / out["actual_arv"]

    # Attach ID columns for easier review
    for col in id_cols:
        out[col] = df[col].values

    # Reorder columns: IDs first, then metrics
    ordered_cols = id_cols + [
        "actual_arv",
        "rf_v2_pred_arv",
        "abs_error_dollars",
        "pct_error",
        "abs_error",
    ]
    out = out[ordered_cols]

    # Sort by absolute dollar error descending
    out = out.sort_values("abs_error_dollars", ascending=False)

    # 8) Save to CSV in artifacts/v2_rf/
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_CSV, index=False)
    print(f"[INFO] Backtest written to: {OUTPUT_CSV}")
    print("[INFO] Top 10 largest misses:")
    print(out.head(10))


if __name__ == "__main__":
    # Ensure project root is on sys.path if needed
    ROOT = Path(__file__).resolve().parent
    if str(ROOT) not in sys.path:
        sys.path.append(str(ROOT))
    main()
