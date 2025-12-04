from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple, Optional

from .config import ARTIFACT_DIR


# Shrinkage exponent: 0 => no adjustment, 1 => full ZHVI ratio.
# 0.5–0.7 is a reasonable “conservative” band.
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


def adjust_arv_for_geo(
    arv: float,
    total_cost: float,
    zipcode: Optional[str],
) -> Tuple[float, float, str, Optional[float]]:
    """
    Apply a ZHVI-based geographic adjustment to the (already clamped) ARV.

    Returns:
        (adjusted_arv, factor, location_status, zhvi_target)

    location_status ∈ {
        "geo_disabled",           # no artifacts → no adjustment
        "no_zip",                 # no zipcode provided
        "in_distribution",        # ZIP seen in training data → no adjustment
        "no_zhvi_for_zip",        # ZIP not in training + no ZHVI entry → no adjustment
        "ood_adjusted",           # out-of-distribution ZIP, adjusted via ZHVI
    }
    """
    geo_meta, zhvi_lookup = _load_geo_artifacts()

    if not geo_meta or not zhvi_lookup:
        return arv, 1.0, "geo_disabled", None

    if not zipcode:
        return arv, 1.0, "no_zip", None

    # Normalize ZIP to 5 digits
    zip_norm = str(zipcode).strip()
    # If user passed "27104-1234", try to extract the 5-digit core
    if len(zip_norm) > 5:
        # simple heuristic: last 5 digits in the string
        digits = "".join(ch for ch in zip_norm if ch.isdigit())
        if len(digits) >= 5:
            zip_norm = digits[-5:]
    zip_norm = zip_norm.zfill(5)

    train_zips = set(geo_meta.get("train_zips", []))
    baseline_zhvi = float(geo_meta.get("baseline_zhvi", 0.0)) or 0.0

    # If baseline is not available, bail out gracefully
    if baseline_zhvi <= 0.0:
        return arv, 1.0, "geo_disabled", None

    zhvi_target = zhvi_lookup.get(zip_norm)
    if zip_norm in train_zips:
        # In-distribution: trust RF v2 as-is, no extra scaling
        return arv, 1.0, "in_distribution", zhvi_target

    if zhvi_target is None:
        # Out-of-distribution ZIP, but no ZHVI entry → cannot adjust
        return arv, 1.0, "no_zhvi_for_zip", None

    # Compute partial adjustment factor
    ratio = float(zhvi_target) / baseline_zhvi
    factor = ratio ** ALPHA

    # Scale ARV and re-clamp to [1×, 2×] total_cost
    adjusted_arv = arv * factor


    return adjusted_arv, factor, "ood_adjusted", float(zhvi_target)
