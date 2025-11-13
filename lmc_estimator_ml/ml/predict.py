# lmc_estimator_ml/ml/predict.py (if you want to keep it)
from __future__ import annotations
from typing import Dict, Any
import numpy as np
import pandas as pd
from .trainer import load_model
from .config import ARTIFACT_DIR

MODEL = load_model(ARTIFACT_DIR)

def predict(payload: Dict[str, Any]) -> dict:
    beds = float(payload.get("beds") or 0)
    baths = float(payload.get("baths") or 0)
    sf = float(payload.get("sf") or 0)
    purchase = float(payload.get("purchase_price") or 0)
    rehab = float(payload.get("rehab_budget") or 0)
    total_cost = purchase + rehab

    row = {
        "square_footage": sf,
        "bed": beds,
        "baths": baths,
        "dom": 0,
        "lot_area_value": 0,
        "year_built": 0,
        "school_score": 0,
        "total_cost": total_cost,
        "rehab_ratio": (rehab / total_cost) if total_cost > 0 else 0.0,
        "city": "Other",
    }
    X = pd.DataFrame([row])

    log_arv = float(MODEL.predict(X)[0])
    arv = float(np.exp(log_arv))

    # reuse the same loan/cash-to-close logic as in app.py
    ltv_limit = 0.70
    ltc_limit = 0.90
    max_loan_ltv = ltv_limit * arv
    max_loan_ltc = ltc_limit * total_cost
    max_loan = min(max_loan_ltv, max_loan_ltc)

    initial_advance = max(max_loan - rehab, 0.0)
    placement_fee = 0.0225 * max_loan
    account_setup_fee = 1195.0
    draw_uw_fees = 594.0

    cash_to_close = (
        purchase
        - initial_advance
        + placement_fee
        + account_setup_fee
        + draw_uw_fees
    )
    cash_to_close = max(0.0, round(cash_to_close, 0))

    return {
        "predicted_arv": arv,
        "max_loan": max_loan,
        "cash_to_close": cash_to_close,
    }
