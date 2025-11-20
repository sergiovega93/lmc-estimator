#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
comps_sanity_check.py

Standalone script to inspect the Comps Database and flag suspicious rows using
rule-based checks. It DOES NOT modify the dataset or training pipeline.

It:
  - Loads the comps from DEFAULT_EXCEL_PATH via load_comps (same normalization).
  - Applies a series of named "rules" (e.g. high ARV/price ratio, zero SF, etc).
  - Prints how many rows matched each rule.
  - Writes an Excel file with all flagged rows and a "flag_reasons" summary.

Usage (from repo root):
    python comps_sanity_check.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

from lmc_estimator_ml.ml.data_loader import load_comps
from lmc_estimator_ml.ml.config import DEFAULT_EXCEL_PATH, TARGET_COL

# =========================
# Configuration
# =========================

# Where to write the output Excel file (relative to repo root)
OUTPUT_DIR = Path("lmc_estimator_ml") / "artifacts"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Rule thresholds (you can tweak these)
HIGH_ARV_TO_PRICE_RATIO = 3.0       # ARV / Purchase > 3 → suspicious
TINY_SF_THRESHOLD = 400             # sqft < 400 → tiny / possible data issue
ACRE_TINY_SF_THRESHOLD = 800        # acres + sf < 800 → odd lot/structure mix
HIGH_ARV_ABS = 1_500_000            # ARV above this → flag as "high_arv_abs"
MIN_PRICE_FOR_ZERO_REHAB = 50_000   # zero rehab but high purchase → likely refi


# =========================
# Rule definitions
# =========================

def rule_high_arv_to_price(df: pd.DataFrame) -> pd.Series:
    """
    projected_value_ma / purchase_price_ma > HIGH_ARV_TO_PRICE_RATIO
    """
    if {"projected_value_ma", "purchase_price_ma"} - set(df.columns):
        return pd.Series(False, index=df.index)

    arv = pd.to_numeric(df["projected_value_ma"], errors="coerce")
    price = pd.to_numeric(df["purchase_price_ma"], errors="coerce").replace({0: np.nan})
    ratio = arv / price
    return (ratio > HIGH_ARV_TO_PRICE_RATIO) & price.notna()


def rule_zero_sf(df: pd.DataFrame) -> pd.Series:
    """
    square_footage <= 0 (missing or obviously wrong). This is the same pattern
    we will later filter out for training, but here we only flag it.
    """
    if "square_footage" not in df.columns:
        return pd.Series(False, index=df.index)

    sf = pd.to_numeric(df["square_footage"], errors="coerce").fillna(0)
    return sf <= 0


def rule_tiny_sf_with_structure(df: pd.DataFrame) -> pd.Series:
    """
    square_footage < TINY_SF_THRESHOLD but has some beds/baths.
    Very small SF might be data error or unusual property type.
    """
    required = {"square_footage", "bed", "baths"}
    if required - set(df.columns):
        return pd.Series(False, index=df.index)

    sf = pd.to_numeric(df["square_footage"], errors="coerce").fillna(0)
    bed = pd.to_numeric(df["bed"], errors="coerce").fillna(0)
    baths = pd.to_numeric(df["baths"], errors="coerce").fillna(0)

    return (sf > 0) & (sf < TINY_SF_THRESHOLD) & ((bed > 0) | (baths > 0))


def rule_acres_with_tiny_sf(df: pd.DataFrame) -> pd.Series:
    """
    lot_area_units contains 'acre' AND square_footage < ACRE_TINY_SF_THRESHOLD.

    This often indicates a LOT or land-like deal with very small or missing SF.
    """
    required = {"lot_area_units", "square_footage"}
    if required - set(df.columns):
        return pd.Series(False, index=df.index)

    units = df["lot_area_units"].astype(str).str.lower().str.strip()
    sf = pd.to_numeric(df["square_footage"], errors="coerce").fillna(0)

    mask_acres = units.str.contains("acre", na=False)
    return mask_acres & (sf < ACRE_TINY_SF_THRESHOLD)


def rule_zero_rehab_high_price(df: pd.DataFrame) -> pd.Series:
    """
    rehab_budget_ma == 0 AND purchase_price_ma > MIN_PRICE_FOR_ZERO_REHAB.

    These are usually refis / cash-outs / hidden rehab deals, not true flips.
    """
    required = {"rehab_budget_ma", "purchase_price_ma"}
    if required - set(df.columns):
        return pd.Series(False, index=df.index)

    rehab = pd.to_numeric(df["rehab_budget_ma"], errors="coerce").fillna(0)
    price = pd.to_numeric(df["purchase_price_ma"], errors="coerce").fillna(0)

    return (rehab == 0) & (price > MIN_PRICE_FOR_ZERO_REHAB)


def rule_high_arv_abs(df: pd.DataFrame) -> pd.Series:
    """
    ARV (projected_value_ma) > HIGH_ARV_ABS.

    These are very high-end deals that deserve extra eyeballing, because even
    small percentage errors yield large dollar errors.
    """
    if "projected_value_ma" not in df.columns:
        return pd.Series(False, index=df.index)

    arv = pd.to_numeric(df["projected_value_ma"], errors="coerce").fillna(0)
    return arv > HIGH_ARV_ABS


def rule_rare_city(df: pd.DataFrame, min_count: int = 2) -> pd.Series:
    """
    Cities that appear less than min_count times in the dataset.

    These are sparse-location comps that may be noisier and less reliable.
    """
    if "city" not in df.columns:
        return pd.Series(False, index=df.index)

    counts = df["city"].astype(str).str.strip().value_counts()
    rare_cities = counts[counts < min_count].index
    return df["city"].astype(str).str.strip().isin(rare_cities)


# List of rules: (id, description, function)
RULES = [
    (
        "high_arv_to_price",
        f"ARV/Price > {HIGH_ARV_TO_PRICE_RATIO:.1f}",
        rule_high_arv_to_price,
    ),
    (
        "zero_sf",
        "Square footage <= 0 (missing / non-structural / land-like)",
        rule_zero_sf,
    ),
    (
        "tiny_sf_with_structure",
        f"SF > 0 & SF < {TINY_SF_THRESHOLD} & (bed>0 or baths>0)",
        rule_tiny_sf_with_structure,
    ),
    (
        "acres_with_tiny_sf",
        f"lot_area_units contains 'acre' & SF < {ACRE_TINY_SF_THRESHOLD}",
        rule_acres_with_tiny_sf,
    ),
    (
        "zero_rehab_high_price",
        f"rehab_budget_ma == 0 & purchase_price_ma > {MIN_PRICE_FOR_ZERO_REHAB}",
        rule_zero_rehab_high_price,
    ),
    (
        "high_arv_abs",
        f"projected_value_ma > {HIGH_ARV_ABS:,}",
        rule_high_arv_abs,
    ),
    (
        "rare_city",
        "city occurs < 2 times in dataset",
        rule_rare_city,
    ),
]


# =========================
# Main logic
# =========================

def main() -> None:
    print(f"[INFO] Loading comps from: {DEFAULT_EXCEL_PATH}")
    df = load_comps(DEFAULT_EXCEL_PATH)

    if TARGET_COL not in df.columns:
        print(f"[WARN] TARGET_COL '{TARGET_COL}' not found in columns: {list(df.columns)}")

    print(f"[INFO] Total rows loaded: {len(df)}")

    # Apply rules
    rule_masks: dict[str, pd.Series] = {}
    for rule_id, desc, func in RULES:
        try:
            mask = func(df)
            if not isinstance(mask, pd.Series) or mask.index is not df.index:
                raise ValueError(f"Rule {rule_id} did not return a proper Series aligned with df.index.")
            rule_masks[rule_id] = mask
            count = mask.sum()
            print(f"[RULE] {rule_id:22s} | matches: {count:4d} | {desc}")
        except Exception as e:
            print(f"[ERROR] Rule {rule_id} failed: {e!r}")
            rule_masks[rule_id] = pd.Series(False, index=df.index)

    # Combine all rules to find any flagged row
    if not rule_masks:
        print("[INFO] No rules defined; nothing to do.")
        return

    any_flagged = None
    for mask in rule_masks.values():
        any_flagged = mask if any_flagged is None else (any_flagged | mask)

    total_flagged = int(any_flagged.sum())
    print(f"\n[SUMMARY] Total unique rows flagged by at least one rule: {total_flagged}")

    if total_flagged == 0:
        print("[SUMMARY] No suspicious rows detected under current rules.")
        return

    # Build output dataframe of flagged rows
    flagged = df[any_flagged].copy()
    # Add boolean columns for each rule + a combined summary
    for rule_id, _, _ in RULES:
        flagged[f"flag_{rule_id}"] = rule_masks[rule_id].loc[flagged.index].astype(bool)

    rule_cols = [c for c in flagged.columns if c.startswith("flag_")]

    def _collect_reasons(row) -> str:
        reasons = []
        for col in rule_cols:
            if row.get(col):
                reasons.append(col.replace("flag_", ""))
        return ", ".join(reasons) if reasons else ""

    flagged["flag_reasons"] = flagged.apply(_collect_reasons, axis=1)
    flagged["flag_rule_count"] = flagged[rule_cols].sum(axis=1)

    # Move flag columns near the front for convenience
    ordered_cols = (
        ["flag_reasons", "flag_rule_count"] +
        rule_cols +
        [c for c in flagged.columns if not c.startswith("flag_") and c not in ("flag_reasons", "flag_rule_count")]
    )
    flagged = flagged[ordered_cols]

    # Write to Excel
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = OUTPUT_DIR / f"suspect_comps_{timestamp}.xlsx"
    flagged.to_excel(out_path, index=False)

    print(f"[OUTPUT] Wrote flagged rows to: {out_path}")
    print(f"[OUTPUT] Columns include per-rule flags and 'flag_reasons' summary.")
    print("\n[NOTE] This script does NOT modify the original Excel. Use it as a review tool to:")
    print("       - Manually correct obvious data entry errors in the Comps Database")
    print("       - Decide which patterns you eventually want to hard-filter in training.")


if __name__ == "__main__":
    ROOT = Path(__file__).resolve().parent
    if str(ROOT) not in sys.path:
        sys.path.append(str(ROOT))
    main()
