from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from lmc_estimator_ml.ml.config import BASE_DIR, ARTIFACT_DIR, DEFAULT_EXCEL_PATH
from lmc_estimator_ml.ml.data_loader import (
    load_comps,
    basic_clean,
    normalize_lot_area,
    filter_eligible,
    apply_training_filters,
    filter_recent,
)


# Path to the big Zillow ZIP ZHVI CSV (you already have this locally)
ZHVI_CSV_PATH = BASE_DIR / "data" / "Zip_zhvi_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv"


def _detect_zip_column(df: pd.DataFrame) -> str:
    """
    Try to find the zipcode column in your Comps data.
    Adjust the candidate list if your column name differs.
    """
    candidates = ["zipcode", "zip_code", "zip"]
    for col in candidates:
        if col in df.columns:
            return col
    raise ValueError(f"Could not find any zipcode column in df.columns = {df.columns.tolist()}")


def main() -> None:
    # 1) Load comps using the same pipeline as RF training
    df = load_comps(DEFAULT_EXCEL_PATH)
    df = basic_clean(df)
    df = normalize_lot_area(df)
    df = filter_eligible(df)
    df = apply_training_filters(df)
    df = filter_recent(df)

    zip_col = _detect_zip_column(df)

    # Normalize ZIP to 5-digit strings
    zip_series = (
        df[zip_col]
        .dropna()
        .astype(str)
        .str.extract(r"(\d{5})")[0]
        .dropna()
    )
    # Define training ZIPs as all ZIPs that appear at least once
    train_zips = sorted(zip_series.unique().tolist())

    if not train_zips:
        raise ValueError("No valid 5-digit ZIP codes found in the training data.")

    # 2) Load ZHVI CSV and compute latest month + baseline
    zhvi = pd.read_csv(ZHVI_CSV_PATH)
    date_cols = [c for c in zhvi.columns if c[:4].isdigit()]
    if not date_cols:
        raise ValueError("Could not find any YYYY-* date columns in ZHVI CSV.")

    latest_month_col = sorted(date_cols)[-1]

    zhvi["RegionName"] = zhvi["RegionName"].astype(str).str.zfill(5)
    zhvi_latest = zhvi[["RegionName", latest_month_col]].copy()
    zhvi_latest.rename(columns={latest_month_col: "zhvi_latest"}, inplace=True)

    train_mask = zhvi_latest["RegionName"].isin(train_zips)
    train_zhvi = zhvi_latest.loc[train_mask, "zhvi_latest"]

    if train_zhvi.empty:
        raise ValueError("No overlap between training ZIP codes and ZHVI ZIPs.")

    baseline_zhvi = float(train_zhvi.median())

    # 3) Build artifacts
    geo_meta = {
        "train_zips": sorted(train_zips),
        "baseline_zhvi": baseline_zhvi,
        "latest_zhvi_month": latest_month_col,
        "min_zip_count": 1,
    }

    zhvi_lookup = {
        str(row.RegionName): float(row.zhvi_latest)
        for _, row in zhvi_latest.iterrows()
    }

    # 4) Write JSONs into the current model artifact directory (e.g., artifacts/v2_rf/)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    geo_meta_path = ARTIFACT_DIR / "geo_reference.json"
    zhvi_lookup_path = ARTIFACT_DIR / "zhvi_zip_latest.json"

    geo_meta_path.write_text(json.dumps(geo_meta, indent=2))
    zhvi_lookup_path.write_text(json.dumps(zhvi_lookup, indent=2))

    print(f"Wrote geo reference to {geo_meta_path}")
    print(f"Wrote ZHVI ZIP lookup to {zhvi_lookup_path}")


if __name__ == "__main__":
    main()
