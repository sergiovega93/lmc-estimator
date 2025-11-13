from __future__ import annotations
from pathlib import Path
from typing import Optional, Dict, Any

import pandas as pd

from .ml import data_loader as dl
from .ml.trainer import train, load_model
from .ml.config import DEFAULT_EXCEL_PATH


# =========================
# 1) TRAINING ENTRY POINT
# =========================

def train_from_excel(excel_path: Path | str) -> dict:
    """
    Full training pipeline used locally (not by the web app).
    - Loads comps from Excel
    - Normalizes / filters
    - Trains the regression model and writes artifacts (model.joblib, meta.json, ols_summary, diagnostics)
    """
    df = dl.load_comps(excel_path)
    df = dl.basic_clean(df)
    df = dl.normalize_lot_area(df)      # acres -> sqft
    df = dl.filter_eligible(df)         # require projected_value_ma, pipeline-ready
    df = dl.filter_recent(df)           # last 6 months if DATE_COL exists
    return train(df)


# =========================
# 2) PREDICTION HELPERS
# =========================

_model_cache = None  # lazy-loaded sklearn Pipeline


def get_model():
    """
    Lazy-load the trained model (model.joblib) from artifacts.
    This is what the FastAPI app will call.
    """
    global _model_cache
    if _model_cache is None:
        _model_cache = load_model()
    return _model_cache


def _infer_city_from_address(address: str) -> str:
    """
    Very simple heuristic:
    - If 'charlotte' in address -> 'Charlotte'
    - If 'indianapolis' in address -> 'Indianapolis'
    - Else -> 'Other'
    This aligns with the city dummies we used in the regression.
    """
    if not address:
        return "Other"
    a = address.lower()
    if "charlotte" in a:
        return "Charlotte"
    if "indianapolis" in a:
        return "Indianapolis"
    return "Other"


def predict_arv_from_inputs(
    address: str,
    beds: Optional[float],
    baths: Optional[float],
    sf: Optional[float],
    purchase: Optional[float],
    rehab: Optional[float],
) -> Dict[str, Any]:
    """
    Build the feature row in the exact shape the sklearn Pipeline expects and return:
      - arv (float)
      - total_cost
      - rehab_ratio
      - city_bucket (Charlotte / Indianapolis / Other)

    We keep missing-values handling extremely simple:
      - If beds/baths/sf are None, we treat them as 0
      - DOM, lot_area_value, year_built, school_score default to 0 for now
    """
    beds = beds or 0.0
    baths = baths or 0.0
    sf = sf or 0.0
    purchase = purchase or 0.0
    rehab = rehab or 0.0

    total_cost = purchase + rehab
    rehab_ratio = (rehab / total_cost) if total_cost > 0 else 0.0
    city_bucket = _infer_city_from_address(address)

    # This must match the feature names used during training (in features.py)
    # We can include extra columns; the ColumnTransformer will select what it needs.
    row = {
        "square_footage": sf,
        "bed": beds,
        "baths": baths,
        "dom": 0.0,
        "lot_area_value": 0.0,
        "year_built": 0.0,
        "school_score": 0.0,
        "total_cost": total_cost,
        "rehab_ratio": rehab_ratio,
        "city": city_bucket,
    }

    df = pd.DataFrame([row])
    model = get_model()
    arv_pred = float(model.predict(df)[0])

    return {
        "arv": arv_pred,
        "total_cost": total_cost,
        "rehab_ratio": rehab_ratio,
        "city_bucket": city_bucket,
    }


def estimate_loan_terms(
    address: str,
    beds: Optional[float],
    baths: Optional[float],
    sf: Optional[float],
    purchase: Optional[float],
    rehab: Optional[float],
) -> Dict[str, Any]:
    """
    High-level engine used by the FastAPI app:
      1) Predict ARV from the regression model
      2) Size the loan using 70% LTV and 90% LTC caps
      3) Estimate initial advance, fees, and cash to close

    Returns a dict with:
      - arv
      - total_cost
      - total_loan
      - max_loan_ltv
      - max_loan_ltc
      - initial_advance
      - placement_fee
      - account_setup_fee
      - fixed_fees
      - cash_to_close
      - city_bucket
      - rehab_ratio
    """
    purchase = purchase or 0.0
    rehab = rehab or 0.0

    core = predict_arv_from_inputs(address, beds, baths, sf, purchase, rehab)
    arv = core["arv"]
    total_cost = core["total_cost"]
    rehab_ratio = core["rehab_ratio"]
    city_bucket = core["city_bucket"]

    # 1) Loan sizing caps
    max_loan_ltv = 0.70 * arv
    max_loan_ltc = 0.90 * total_cost
    total_loan = min(max_loan_ltv, max_loan_ltc)

    # 2) Simple fee logic for public estimator (approximate internal economics)
    initial_advance = total_loan - rehab  # funds at closing
    placement_fee = 0.0225 * total_loan   # 2.25 points on the loan
    account_setup_fee = 1195.0
    fixed_fees = 594.0                    # draw + underwriting
    # totalcustomfees assumed ~0 for estimator

    cash_to_close = purchase - initial_advance + placement_fee + account_setup_fee + fixed_fees
    cash_to_close = max(0.0, cash_to_close)

    return {
        "arv": arv,
        "total_cost": total_cost,
        "total_loan": total_loan,
        "max_loan_ltv": max_loan_ltv,
        "max_loan_ltc": max_loan_ltc,
        "initial_advance": initial_advance,
        "placement_fee": placement_fee,
        "account_setup_fee": account_setup_fee,
        "fixed_fees": fixed_fees,
        "cash_to_close": cash_to_close,
        "city_bucket": city_bucket,
        "rehab_ratio": rehab_ratio,
    }


# =========================
# 3) CLI TRAINING ENTRY
# =========================

if __name__ == "__main__":
    meta = train_from_excel(DEFAULT_EXCEL_PATH)
    print("Training complete. Meta:", meta)
