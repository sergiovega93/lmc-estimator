from __future__ import annotations

from pathlib import Path
from typing import Optional

import json
import numpy as np
import pandas as pd
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from lmc_estimator_ml.ml.trainer import load_model
from lmc_estimator_ml.ml.config import ARTIFACT_DIR

# ------------------------------
# FastAPI + templates
# ------------------------------
app = FastAPI()
templates = Jinja2Templates(directory="templates")

# ------------------------------
# Load trained AVM model + meta
# ------------------------------
MODEL = load_model(ARTIFACT_DIR)

META_PATH = ARTIFACT_DIR / "meta.json"
if META_PATH.exists():
    META = json.loads(META_PATH.read_text())
else:
    META = {}

print(
    "Loaded AVM model.",
    "Target:", META.get("target"),
    "R2 train:", META.get("r2_train"),
    "R2 test:", META.get("r2_test"),
)

# ------------------------------
# ARV clamp settings
# ------------------------------
ARV_TOTALCOST_MIN = 1.0   # ARV >= 1.0x total_cost
ARV_TOTALCOST_MAX = 2.0   # ARV <= 2.0x total_cost

# ------------------------------
# Helper: build feature row from form
# ------------------------------
def build_features_from_form(
    beds: Optional[float],
    baths: Optional[float],
    sf: Optional[float],
    purchase: Optional[float],
    rehab: Optional[float],
    city: str = "Other",
    state: str = "Unknown",
) -> tuple[pd.DataFrame, float]:
    """
    Build a single-row DataFrame with the same columns used in training.
    Unknown numeric fields are set to NaN so the pipeline imputers can do their job.
    Returns (X, total_cost).
    """
    beds = beds or 0.0
    baths = baths or 0.0
    sf = sf or 0.0
    purchase = purchase or 0.0
    rehab = rehab or 0.0

    total_cost = purchase + rehab
    rehab_ratio = (rehab / total_cost) if total_cost > 0 else 0.0

    row = {
        "square_footage": sf,
        "bed": beds,
        "baths": baths,
        "dom": np.nan,
        "lot_area_value": np.nan,
        "year_built": np.nan,
        "school_score": np.nan,
        "total_cost": total_cost,
        "rehab_ratio": rehab_ratio,
        "city": city,
        "state": state,
    }

    X = pd.DataFrame([row])
    return X, total_cost

# ------------------------------
# Cash-to-close logic — LTV-only version
# ------------------------------
def compute_loan_and_cash_to_close_ltv_only(
    arv: float,
    purchase: float,
    rehab: float,
    ltv_limit: float = 0.70,
    placement_points: float = 0.0225,
    account_setup_fee: float = 1195.0,
    fixed_other_fees: float = 594.0,
) -> dict:
    """
    Compute:
      - total_cost
      - total_loan = ltv_limit * arv  (LTV-only cap)
      - initial_advance
      - placement_fee
      - estimated cash_to_close using the MA-style formula:

          purchase_price
        - initial_advance
        + placement_fee
        + account_setup_fee
        + fixed_other_fees
    """
    purchase = purchase or 0.0
    rehab = rehab or 0.0
    total_cost = purchase + rehab

    total_loan = max(0.0, ltv_limit * arv)

    # Initial advance = total loan - rehab budget (never below 0)
    initial_advance = max(total_loan - rehab, 0.0)

    placement_fee = total_loan * placement_points

    cash_to_close = (
        purchase
        - initial_advance
        + placement_fee
        + account_setup_fee
        + fixed_other_fees
    )
    cash_to_close = max(round(cash_to_close, 0), 0.0)

    return {
        "total_cost": total_cost,
        "total_loan": total_loan,
        "initial_advance": initial_advance,
        "placement_fee": placement_fee,
        "cash_to_close": cash_to_close,
        "ltv_limit": ltv_limit,
    }

# ------------------------------
# Routes
# ------------------------------
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/estimate", response_class=HTMLResponse)
def estimate(
    request: Request,
    address: str = Form(...),
    beds: float | None = Form(None),
    baths: float | None = Form(None),
    sf: float | None = Form(None),
    purchase: float | None = Form(None),
    rehab: float | None = Form(None),
):
    # 1) Normalize inputs
    beds = beds or 0
    baths = baths or 0
    sf = sf or 0
    purchase = purchase or 0
    rehab = rehab or 0

    # 2) Build model input row and get total_cost
    X, total_cost = build_features_from_form(
        beds=beds,
        baths=baths,
        sf=sf,
        purchase=purchase,
        rehab=rehab,
        city="Other",     # later you can expose city/state in the form
        state="Unknown",
    )

    # 3) Predict log(ARV) and convert back to dollars
    log_arv = float(MODEL.predict(X)[0])
    arv_raw = float(np.exp(log_arv))

    # 4) Business guard-rail: clamp ARV relative to total_cost
    arv = arv_raw
    clamped = False
    clamped_ratio = None

    if total_cost > 0:
        ratio = arv_raw / total_cost
        clamped_ratio = min(max(ratio, ARV_TOTALCOST_MIN), ARV_TOTALCOST_MAX)
        if abs(clamped_ratio - ratio) > 1e-6:
            clamped = True
        arv = clamped_ratio * total_cost

    # 5) Loan structure & cash to close (LTV-only)
    finance = compute_loan_and_cash_to_close_ltv_only(
        arv=arv,
        purchase=purchase,
        rehab=rehab,
        ltv_limit=0.70,
    )

    # 6) Prepare data for pretty HTML
    context = {
        "request": request,
        "address": address,
        "beds": beds,
        "baths": baths,
        "sf": sf,
        "purchase": purchase,
        "rehab": rehab,
        "total_cost": finance["total_cost"],
        "arv_raw": arv_raw,
        "arv": arv,
        "arv_clamped": clamped,
        "clamped_ratio": clamped_ratio,
        "total_loan": finance["total_loan"],
        "initial_advance": finance["initial_advance"],
        "placement_fee": finance["placement_fee"],
        "cash_to_close": finance["cash_to_close"],
        "ltv_limit": finance["ltv_limit"],
        "model_r2": META.get("r2_test"),
        "model_type": META.get("model_type", "RandomForestRegressor"),
    }

    # Render a dedicated result template
    return templates.TemplateResponse("result.html", context)


@app.get("/health")
def health():
    return {
        "ok": True,
        "model_target": META.get("target"),
        "r2_train": META.get("r2_train"),
        "r2_test": META.get("r2_test"),
        "model_type": META.get("model_type", "Unknown"),
    }
