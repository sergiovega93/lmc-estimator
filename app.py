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
# Helper: build feature row from form
# ------------------------------
# Clamp of ARV / total_cost ratio for business sanity checks
ARV_TOTALCOST_MIN = 1.0   # tweak later using data
ARV_TOTALCOST_MAX = 2.0   # tweak later using data

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
# Cash-to-close logic (MA-style)
# ------------------------------
def compute_loan_and_cash_to_close(
    arv: float,
    purchase: float,
    rehab: float,
    ltv_limit: float = 0.70,
    ltc_limit: float = 0.90,
    placement_points: float = 0.0225,
    account_setup_fee: float = 1195.0,
    fixed_other_fees: float = 594.0,
) -> dict:
    """
    Compute:
      - total_cost
      - total_loan (min of LTV- and LTC-based caps)
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

    # Caps based on ARV and total cost
    max_loan_ltv = ltv_limit * arv
    max_loan_ltc = ltc_limit * total_cost
    total_loan = min(max_loan_ltv, max_loan_ltc)

    # Initial advance = total loan - rehab budget
    initial_advance = max(total_loan - rehab, 0.0)

    # Fees
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
        "max_loan_ltv": max_loan_ltv,
        "max_loan_ltc": max_loan_ltc,
    }


# ------------------------------
# Routes
# ------------------------------
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    # Render the HTML form from templates/index.html
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
    if total_cost > 0:
        ratio = arv_raw / total_cost
        clamped_ratio = min(max(ratio, ARV_TOTALCOST_MIN), ARV_TOTALCOST_MAX)
        arv = clamped_ratio * total_cost

    # 5) Loan structure & cash to close (MA-style approximation)
    max_loan_ltv = 0.70 * arv
    max_loan_ltc = 0.90 * total_cost
    max_loan = min(max_loan_ltv, max_loan_ltc)

    rehab_budget = rehab
    initial_advance = max_loan - rehab_budget

    placement_fee = 0.0225 * max_loan    # 2.25 pts
    account_setup_fee = 1195.0
    draw_uw_fees = 594.0                 # draw + UW fee
    total_custom_fees = 0.0              # for now

    cash_to_close = (
        purchase
        - initial_advance
        + placement_fee
        + account_setup_fee
        + total_custom_fees
        + draw_uw_fees
    )
    cash_to_close = max(0.0, round(cash_to_close, 0))

    # 6) Render result nicely
    html = f"""
    <h1>LMC Estimator — Preliminary Results</h1>
    <p><b>Address:</b> {address}</p>

    <h2>Inputs</h2>
    <ul>
      <li><b>Beds:</b> {beds}</li>
      <li><b>Baths:</b> {baths}</li>
      <li><b>Square Feet:</b> {sf:,.0f}</li>
      <li><b>Purchase Price:</b> ${purchase:,.0f}</li>
      <li><b>Rehab Budget:</b> ${rehab:,.0f}</li>
    </ul>

    <h2>Estimated After-Repair Value (ARV)</h2>
    <p><b>Estimated ARV:</b> ${arv:,.0f}</p>
    <p style="font-size:0.85em; color:#666;">
      (Internally capped between {ARV_TOTALCOST_MIN:.2f}× and {ARV_TOTALCOST_MAX:.2f}× total project cost for business sanity.)
    </p>

    <h2>Estimated Loan & Cash to Close</h2>
    <ul>
      <li><b>Total Project Cost:</b> ${total_cost:,.0f}</li>
      <li><b>Max Loan by 70% ARV:</b> ${max_loan_ltv:,.0f}</li>
      <li><b>Max Loan by 90% LTC:</b> ${max_loan_ltc:,.0f}</li>
      <li><b>Estimated Total Loan (cap):</b> ${max_loan:,.0f}</li>
      <li><b>Estimated Initial Advance:</b> ${initial_advance:,.0f}</li>
      <li><b>Estimated Placement Fee (2.25 pts):</b> ${placement_fee:,.0f}</li>
      <li><b>Estimated Cash to Close:</b> ${cash_to_close:,.0f}</li>
    </ul>

    <p style="margin-top:16px; font-size:0.9em; color:#555;">
      This is a preliminary, non-binding estimate based on LMC’s internal AVM and 
      standard assumptions (70% ARV cap, 90% LTC cap, typical fees). Actual terms 
      depend on full underwriting.
    </p>

    <p style="margin-top:16px;"><a href="/">Back to form</a></p>
    """
    return HTMLResponse(html)

@app.get("/health")
def health():
    return {
        "ok": True,
        "model_target": META.get("target"),
        "r2_train": META.get("r2_train"),
        "r2_test": META.get("r2_test"),
    }
