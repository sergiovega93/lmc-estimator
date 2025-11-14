# sanity_check_avm.py
"""
Quick sanity checker for AVM v1.

- Loads the same cleaned dataset used for training.
- Reconstructs total_cost and rehab_ratio from purchase_price_ma and rehab_budget_ma.
- Uses the same feature list as training (NUMERIC_FEATURES + CATEGORICAL_FEATURES).
- Predicts log(ARV) -> exp to dollars.
- Prints a 10-row sample and descriptive stats of predicted_arv / total_cost.
"""

import numpy as np
import pandas as pd

from lmc_estimator_ml.ml import data_loader as dl
from lmc_estimator_ml.ml.trainer import load_model
from lmc_estimator_ml.ml.config import (
    ARTIFACT_DIR,
    DEFAULT_EXCEL_PATH,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    TARGET_COL,
)


def main():
    print("Loading data from:", DEFAULT_EXCEL_PATH)

    df = dl.load_comps(DEFAULT_EXCEL_PATH)
    df = dl.basic_clean(df)
    df = dl.normalize_lot_area(df)
    df = dl.filter_eligible(df)
    df = dl.filter_recent(df)

    print("Rows after filtering:", len(df))
    print("Columns after clean/filter:", sorted(df.columns))

    # ---- Rebuild total_cost and rehab_ratio exactly as in training ----
    if "purchase_price_ma" not in df.columns or "rehab_budget_ma" not in df.columns:
        raise ValueError(
            "Expected 'purchase_price_ma' and 'rehab_budget_ma' in df.\n"
            f"Available columns: {sorted(df.columns)}"
        )

    df["total_cost"] = df["purchase_price_ma"].fillna(0) + df["rehab_budget_ma"].fillna(0)
    # Avoid division by zero for rehab_ratio
    denom = df["purchase_price_ma"].replace(0, np.nan)
    df["rehab_ratio"] = df["rehab_budget_ma"] / denom

    # ---- Ensure all feature columns exist (create as NaN if missing) ----
    feature_cols = list(NUMERIC_FEATURES) + list(CATEGORICAL_FEATURES)
    for col in feature_cols:
        if col not in df.columns:
            df[col] = np.nan

    # Now safely build X in the same order as training
    X = df[feature_cols]

    # ---- Load the trained model and predict ----
    model = load_model(ARTIFACT_DIR)
    log_preds = model.predict(X)
    df["predicted_arv"] = np.exp(log_preds.astype(float))

    # ---- Ratio predicted_arv / total_cost ----
    df["ratio_pred_total"] = df["predicted_arv"] / df["total_cost"].replace(0, np.nan)

    # ---- Show 10 random rows ----
    sample = df.sample(n=10, random_state=42)
    cols_to_show = [
        "total_cost",
        TARGET_COL,
        "predicted_arv",
        "ratio_pred_total",
    ]
    print("\nSample of 10 rows (total_cost, actual, predicted, ratio):")
    print(sample[cols_to_show].round(0))

    # ---- Stats of the ratio ----
    print("\nRatio predicted_arv / total_cost stats:")
    print(df["ratio_pred_total"].describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9]))


if __name__ == "__main__":
    main()
