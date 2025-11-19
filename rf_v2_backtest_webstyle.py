#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Backtest RF v2 model under *web-style inputs*:

We simulate exactly what the FastAPI app sees:
  - Inputs: beds, baths, square footage, purchase, rehab, city, state
  - dom, lot_area_value, year_built, school_score are set to NaN
  - total_cost = purchase + rehab
  - rehab_ratio = rehab / purchase (capped)

This lets us compare:
  - model performance with full training-style features (rf_v2_backtest.py)
  - vs. model performance under actual runtime inputs from the public form.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from lmc_estimator_ml.ml.trainer import load_model
from lmc_estimator_ml.ml.data_loader import load_comps
from lmc_estimator_ml.ml.config import ARTIFACT_DIR, DEFAULT_EXCEL_PATH
from lmc_estimator_ml.ml import data_loader as dl

# =====================
# Settings
# =====================

TARGET_COL = "projected_value_ma"
DIAG_PATH = ARTIFACT_DIR / "diagnostics.json"

# Web-style rehab_ratio cap (tune here: 5.0 vs 2.5, etc.)
REHAB_RATIO_CAP = 5.0  # set to 2.5 to test tighter cap

OUTPUT_CSV = ARTIFACT_DIR / f"v2_rf_backtest_webstyle_cap_{REHAB_RATIO_CAP:.1f}.csv"


def build_webstyle_features(df_raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Build X in the *same shape* as the web app's build_features_from_form(),
    but vectorized over the entire comps dataframe.

    Assumes df_raw is the *normalized* output of load_comps(), with at least:
      - 'bed', 'baths', 'square_footage'
      - 'purchase_price_ma', 'rehab_budget_ma'
      - 'city', 'state' (optional but expected)
      - TARGET_COL ('projected_value_ma')
    """
    df = df_raw.copy()

    # ---- Map raw columns to web inputs (normalized names from load_comps) ----
    bed_col = "bed"
    baths_col = "baths"
    sf_col = "square_footage"
    purchase_col = "purchase_price_ma"
    rehab_col = "rehab_budget_ma"
    city_col = "city"
    state_col = "state"

    missing_required = [c for c in [bed_col, baths_col, sf_col, purchase_col, rehab_col] if c not in df.columns]
    if missing_required:
        raise KeyError(f"build_webstyle_features: required columns missing from df_raw: {missing_required}")

    # Numeric coercion with safe defaults
    beds = pd.to_numeric(df[bed_col], errors="coerce").fillna(0.0)
    baths = pd.to_numeric(df[baths_col], errors="coerce").fillna(0.0)
    sf = pd.to_numeric(df[sf_col], errors="coerce").fillna(0.0)
    purchase = pd.to_numeric(df[purchase_col], errors="coerce").fillna(0.0)
    rehab = pd.to_numeric(df[rehab_col], errors="coerce").fillna(0.0)

    # Address-derived fields (already normalized by load_comps)
    city = df.get(city_col, pd.Series(["Other"] * len(df), index=df.index)).fillna("Other")
    state = df.get(state_col, pd.Series(["Unknown"] * len(df), index=df.index)).fillna("Unknown")

    # ---- Web-style engineered features ----
    total_cost = purchase + rehab

    # rehab_ratio = rehab / purchase (capped)
    denom = purchase.replace(0, np.nan)
    rehab_ratio = (rehab / denom).fillna(0.0)
    rehab_ratio = rehab_ratio.clip(lower=0.0, upper=REHAB_RATIO_CAP)

    # dom / lot_area_value / year_built / school_score are unknown for web leads
    idx = df.index
    dom = pd.Series([np.nan] * len(df), index=idx)
    lot_area_value = pd.Series([np.nan] * len(df), index=idx)
    year_built = pd.Series([np.nan] * len(df), index=idx)
    school_score = pd.Series([np.nan] * len(df), index=idx)

    # ---- Build the schema used by the web ----
    # Must match the training feature names expected by the pipeline (plus 'state', which is ignored if not in ColumnTransformer)
    X_web = pd.DataFrame({
        "square_footage": sf,
        "bed": beds,
        "baths": baths,
        "dom": dom,
        "lot_area_value": lot_area_value,
        "year_built": year_built,
        "school_score": school_score,
        "total_cost": total_cost,
        "rehab_ratio": rehab_ratio,
        "city": city,
        "state": state,
    })

    y_actual = pd.to_numeric(df[TARGET_COL], errors="coerce")

    return X_web, y_actual


def main() -> None:
    # 1) Confirm diagnostics exist (primarily to ensure ARTIFACT_DIR is valid)
    if not DIAG_PATH.exists():
        raise FileNotFoundError(f"diagnostics.json not found at {DIAG_PATH}")

    diag = json.loads(DIAG_PATH.read_text())
    print("[INFO] Loaded diagnostics for model_type:", diag.get("model_type"))

    # 2) Load comps from Excel via load_comps (normalized headers)
    excel_path = Path(DEFAULT_EXCEL_PATH)
    if not excel_path.exists():
        raise FileNotFoundError(f"DEFAULT_EXCEL_PATH not found: {excel_path}")

    print(f"[INFO] Loading comps from: {excel_path}")
    df_raw = load_comps(excel_path)
    df_raw = dl.basic_clean(df_raw)
    df_raw = dl.normalize_lot_area(df_raw)
    df_raw = dl.filter_eligible(df_raw)
    df_raw = dl.filter_recent(df_raw)
    df_raw = dl.apply_training_filters(df_raw)

    if TARGET_COL not in df_raw.columns:
        raise KeyError(f"Target column '{TARGET_COL}' not found in dataframe columns: {list(df_raw.columns)}")

    # Drop rows with missing target
    df_raw = df_raw.dropna(subset=[TARGET_COL]).copy()
    print(f"[INFO] Rows with non-null {TARGET_COL}: {len(df_raw)}")

    # 3) Build web-style features
    X_web, y_actual = build_webstyle_features(df_raw)

    # 4) Load RF pipeline (same as app.py)
    print(f"[INFO] Loading RF model from ARTIFACT_DIR={ARTIFACT_DIR}")
    model = load_model(ARTIFACT_DIR)

    # 5) Predict log(ARV) then exponentiate
    log_pred = model.predict(X_web)
    y_pred = np.exp(log_pred.astype(float))

    # 6) Comparison dataframe
    out = pd.DataFrame({
        "actual_arv": y_actual.values,
        "rf_v2_pred_arv_webstyle": y_pred,
    })

    out["abs_error"] = out["rf_v2_pred_arv_webstyle"] - out["actual_arv"]
    out["abs_error_dollars"] = out["abs_error"].abs()
    out["pct_error"] = out["abs_error"] / out["actual_arv"]

    # Try to attach IDs / useful metadata if still present after load_comps
    id_candidates = ["loan_number", "Loan Number", "street_address", "Street Address",
                     "city", "City", "state", "State", "zipcode", "Zipcode"]
    id_cols = [c for c in id_candidates if c in df_raw.columns]

    for col in id_cols:
        out[col] = df_raw[col].values

    ordered_cols = id_cols + [
        "actual_arv",
        "rf_v2_pred_arv_webstyle",
        "abs_error_dollars",
        "pct_error",
        "abs_error",
    ]
    out = out[ordered_cols]

    out = out.sort_values("abs_error_dollars", ascending=False)

    # 7) Save CSV
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_CSV, index=False)
    print(f"[INFO] Web-style backtest written to: {OUTPUT_CSV}")
    print("[INFO] Top 10 largest misses (web-style):")
    print(out.head(10))


if __name__ == "__main__":
    ROOT = Path(__file__).resolve().parent
    if str(ROOT) not in sys.path:
        sys.path.append(str(ROOT))
    main()
