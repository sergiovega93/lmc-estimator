from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple, Optional

from .config import ARTIFACT_DIR


# Shrinkage exponent: 0 => no adjustment, 1 => full ZHVI ratio.
# 0.2 is a conservative setting we already agreed on.
ALPHA = 0.2

# In-memory caches
_GEO_META: Optional[dict] = None
_ZHVI_LOOKUP: Optional[dict] = None


def _load_geo_artifacts() -> Tuple[dict, dict]:
    """
    Load geo_reference.json and zhvi_zip_latest.json from the current model artifact dir.
    Returns (geo_meta, zhvi_lookup). If either file is missing, returns empty dicts.
    """
    global _GEO_META, _ZHVI_LOOKUP

    if _GEO_META is not None and _ZHVI_LOOKUP is not None:
        return _GEO_META, _ZHVI_LOOKUP

    meta_path = ARTIFACT_DIR / "geo_reference.json"
    zhvi_path = ARTIFACT_DIR / "zhvi_zip_latest.json"

    if meta_path.exists():
        _GEO_META = json.loads(meta_path.read_text())
    else:
        _GEO_META = {}

    if zhvi_path.exists():
        _ZHVI_LOOKUP = json.loads(zhvi_path.read_text())
    else:
        _ZHVI_LOOKUP = {}

    return _GEO_META, _ZHVI_LOOKUP


def _normalize_zip(zipcode: str) -> str:
    """
    Normalize a user-provided zipcode string into a 5-digit ZIP code.
    Handles cases like '27104-1234' by extracting the last 5 digits.
    """
    zip_norm = str(zipcode).strip()
    if len(zip_norm) > 5:
        digits = "".join(ch for ch in zip_norm if ch.isdigit())
        if len(digits) >= 5:
            zip_norm = digits[-5:]
    zip_norm = zip_norm.zfill(5)
    return zip_norm


def adjust_arv_for_geo(
    arv: float,
    total_cost: float,
    zipcode: Optional[str],
    city: Optional[str] = None,
) -> Tuple[float, float, str, Optional[float]]:
    """
    Apply a ZHVI-based geographic adjustment to the RF-predicted ARV.

    Regimes:
      1) ZIP seen in training (zip ∈ train_zips):
         - location_status = "in_distribution"
         - factor = 1.0
         - adjusted_arv = arv

      2) ZIP unseen, city seen in training (zip ∉ train_zips, city ∈ city_medians_train):
         - baseline = median ZHVI over training ZIPs for that city
         - ratio = zhvi_target / baseline
         - factor = ratio ** ALPHA
         - location_status = "ood_adjusted"

      3) ZIP unseen, city unseen in training (zip ∉ train_zips, city ∉ city_medians_train):
         - baseline = global baseline_zhvi (median over all training ZIPs)
         - ratio = zhvi_target / baseline
         - factor = ratio ** ALPHA
         - location_status = "ood_adjusted"

    Fallback statuses:
      - "geo_disabled"   : geo artifacts missing / invalid baseline
      - "no_zip"         : no zipcode provided
      - "no_zhvi_for_zip": zipcode not found in ZHVI lookup

    Returns:
        adjusted_arv: float          # ARV after applying the factor (or original on fallback)
        factor: float                # multiplier actually applied (1.0 if no adjustment)
        location_status: str         # one of the statuses above
        zhvi_target: Optional[float] # ZHVI for the target ZIP if available, else None
    """
    geo_meta, zhvi_lookup = _load_geo_artifacts()

    if not geo_meta or not zhvi_lookup:
        return arv, 1.0, "geo_disabled", None

    if not zipcode:
        return arv, 1.0, "no_zip", None

    zip_norm = _normalize_zip(zipcode)

    train_zips = set(geo_meta.get("train_zips", []))
    baseline_zhvi = float(geo_meta.get("baseline_zhvi", 0.0)) or 0.0
    city_medians_train = geo_meta.get("city_medians_train", {}) or {}

    # If baseline is not available, bail out gracefully
    if baseline_zhvi <= 0.0:
        return arv, 1.0, "geo_disabled", None

    zhvi_target_raw = zhvi_lookup.get(zip_norm)

    # Regime 1: ZIP seen in training → trust RF, no scaling
    if zip_norm in train_zips:
        return arv, 1.0, "in_distribution", zhvi_target_raw

    # For OOD ZIPs, we need ZHVI for the target ZIP to adjust
    if zhvi_target_raw is None:
        return arv, 1.0, "no_zhvi_for_zip", None

    zhvi_target = float(zhvi_target_raw)

    # Determine which baseline to use
    baseline = baseline_zhvi  # default: global median across training ZIPs

    # Regime 2 vs 3: check if the city is known in training
    if city:
        city_norm = city.strip().upper()
        city_baseline = city_medians_train.get(city_norm)
        if city_baseline is not None and city_baseline > 0.0:
            baseline = float(city_baseline)

    # If for some reason baseline falls back to non-positive, disable geo
    if baseline <= 0.0:
        return arv, 1.0, "geo_disabled", None

    ratio = zhvi_target / baseline
    factor = ratio ** ALPHA

    adjusted_arv = arv * factor

    return adjusted_arv, factor, "ood_adjusted", zhvi_target
