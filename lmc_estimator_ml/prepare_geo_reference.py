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


def _detect_city_column(df: pd.DataFrame) -> str:
    """
    Try to find the city column in your Comps data.
    We keep this flexible and conservative: if you ever change the column
    name in the comps sheet, just extend this candidate list.
    """
    candidates = ["city", "City", "CITY", "city_name", "CityName"]
    for col in candidates:
        if col in df.columns:
            return col
    raise ValueError(f"Could not find any city column in df.columns = {df.columns.tolist()}")


def main() -> None:
    # 1) Load comps using the same pipeline as RF training
    df = load_comps(DEFAULT_EXCEL_PATH)
    df = basic_clean(df)
    df = normalize_lot_area(df)
    df = filter_eligible(df)
    df = apply_training_filters(df)
    df = filter_recent(df)

    zip_col = _detect_zip_column(df)
    city_col = _detect_city_column(df)

    # Normalize ZIP to 5-digit strings for training ZIPs
    zip_series = (
        df[zip_col]
        .dropna()
        .astype(str)
        .str.extract(r"(\d{5})")[0]
        .dropna()
    )
    # Define training ZIPs as all ZIPs that appear at least once in the filtered comps
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

    # Global baseline: only training ZIPs
    train_mask = zhvi_latest["RegionName"].isin(train_zips)
    train_zhvi = zhvi_latest.loc[train_mask, "zhvi_latest"]

    if train_zhvi.empty:
        raise ValueError("No overlap between training ZIP codes and ZHVI ZIPs.")

    baseline_zhvi = float(train_zhvi.median())

    # 3) Build city_medians_train using only ZIPs that are in training AND have ZHVI
    # First, build a (zip, city) table from the comps
    df_zip_city = df[[zip_col, city_col]].dropna().copy()
    df_zip_city[zip_col] = (
        df_zip_city[zip_col]
        .astype(str)
        .str.extract(r"(\d{5})")[0]
        .dropna()
    )

    # Drop rows where we failed to extract a 5-digit ZIP
    df_zip_city = df_zip_city.dropna(subset=[zip_col])

    # Normalize city to uppercase string keys
    df_zip_city["city_norm"] = df_zip_city[city_col].astype(str).str.strip().str.upper()

    # Deduplicate by (zip, city_norm) to avoid double-counting
    df_zip_city = df_zip_city.drop_duplicates(subset=[zip_col, "city_norm"])

    # Restrict to ZIPs that we know are in training_zips (for consistency)
    df_zip_city = df_zip_city[df_zip_city[zip_col].isin(train_zips)]

    # Merge ZHVI latest onto the (zip, city_norm) table
    df_zip_city = df_zip_city.merge(
        zhvi_latest,
        left_on=zip_col,
        right_on="RegionName",
        how="inner",
    )

    # Group by city_norm and compute median zhvi_latest
    if not df_zip_city.empty:
        city_medians_train_series = (
            df_zip_city.groupby("city_norm")["zhvi_latest"].median()
        )
        city_medians_train = {
            str(city): float(val)
            for city, val in city_medians_train_series.items()
            if pd.notnull(val)
        }
    else:
        city_medians_train = {}

    # 4) Build artifacts
    geo_meta = {
        "train_zips": sorted(train_zips),
        "baseline_zhvi": baseline_zhvi,
        "latest_zhvi_month": latest_month_col,
        "min_zip_count": 1,
        "city_medians_train": city_medians_train,
    }

    zhvi_lookup = {
        str(row.RegionName): float(row.zhvi_latest)
        for _, row in zhvi_latest.iterrows()
        if pd.notnull(row.zhvi_latest)
    }

    # 5) Write JSONs into the current model artifact directory (e.g., artifacts/v2_rf/)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    geo_meta_path = ARTIFACT_DIR / "geo_reference.json"
    zhvi_lookup_path = ARTIFACT_DIR / "zhvi_zip_latest.json"

    geo_meta_path.write_text(json.dumps(geo_meta, indent=2))
    zhvi_lookup_path.write_text(json.dumps(zhvi_lookup, indent=2))

    print(f"Wrote geo reference to {geo_meta_path}")
    print(f"Wrote ZHVI ZIP lookup to {zhvi_lookup_path}")


if __name__ == "__main__":
    main()
