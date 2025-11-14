from __future__ import annotations

from pathlib import Path
from typing import Optional

import json
import os
from datetime import datetime

import numpy as np
import pandas as pd
from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from lmc_estimator_ml.ml.trainer import load_model
from lmc_estimator_ml.ml.config import ARTIFACT_DIR
from collections import Counter
from datetime import datetime as dt
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

# Derive friendly meta fields
MODEL_TYPE = META.get("model_type", "RandomForestRegressor")
MODEL_VERSION = META.get("artifact_subdir", ARTIFACT_DIR.name)
MODEL_R2 = META.get("r2_test") or META.get("r2_test_log")

print(
    "Loaded AVM model.",
    "Target:", META.get("target"),
    "Type:", MODEL_TYPE,
    "Version:", MODEL_VERSION,
    "R2 test:", MODEL_R2,
)

# ------------------------------
# ARV clamp settings
# ------------------------------
ARV_TOTALCOST_MIN = 1.0   # ARV >= 1.0x total_cost
ARV_TOTALCOST_MAX = 2.0   # ARV <= 2.0x total_cost

# ------------------------------
# Logging setup (events + leads)
# ------------------------------
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

EVENTS_LOG = LOG_DIR / "events.jsonl"
LEADS_LOG = LOG_DIR / "leads.jsonl"

# Admin token for /admin-stats
ADMIN_TOKEN = os.getenv("st4ts0nmyREND3R")

def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows

def log_event(event_type: str, request: Request, data: dict) -> None:
    record = {
        "timestamp": datetime.utcnow().isoformat(),
        "event": event_type,
        "path": request.url.path,
        "client_ip": request.client.host if request.client else None,
        "user_agent": request.headers.get("user-agent"),
        **data,
    }
    with EVENTS_LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def log_lead(payload: dict) -> None:
    record = {
        "timestamp": datetime.utcnow().isoformat(),
        **payload,
    }
    with LEADS_LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def send_lead_email(lead: dict) -> None:
    """
    Optional: send lead details to sales@loanmountaincapital.com via SMTP.

    Configure these env vars on Render (or locally) for this to work:
      LMC_SMTP_HOST
      LMC_SMTP_PORT   (e.g. 587)
      LMC_SMTP_USER   (from-address / login)
      LMC_SMTP_PASS

    If they are missing, this function quietly does nothing.
    """
    host = os.getenv("LMC_SMTP_HOST")
    user = os.getenv("LMC_SMTP_USER")
    password = os.getenv("LMC_SMTP_PASS")
    port = int(os.getenv("LMC_SMTP_PORT", "587"))

    if not host or not user or not password:
        # No SMTP config → skip email sending
        return

    import smtplib
    from email.message import EmailMessage

    msg = EmailMessage()
    msg["Subject"] = f"[LMC Estimator Lead] {lead.get('name')} - {lead.get('address')}"
    msg["From"] = user
    msg["To"] = "sergio@loanmountaincapital.com"

    body_lines = [
        "New LMC Estimator lead",
        "",
        f"Name:  {lead.get('name')}",
        f"Email: {lead.get('email')}",
        f"Phone: {lead.get('phone')}",
        "",
        f"Address:                {lead.get('address')}",
        f"ARV (clamped):          ${lead.get('arv')}",
        f"Estimated Loan (70%):   ${lead.get('total_loan')}",
        f"Estimated Cash to Close:${lead.get('cash_to_close')}",
        "",
        "Comments:",
        lead.get("comments") or "(none)",
    ]
    msg.set_content("\n".join(body_lines))

    with smtplib.SMTP(host, port) as server:
        server.starttls()
        server.login(user, password)
        server.send_message(msg)

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
    Unknown numeric fields are set to defaults so the pipeline imputers can do their job.
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

    # 6) Prepare data for HTML
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
        "model_r2": MODEL_R2,
        "model_type": MODEL_TYPE,
        "model_version": MODEL_VERSION,
    }

    # 7) Log event for basic analytics
    log_event(
        "estimate_submitted",
        request,
        {
            "address": address,
            "beds": beds,
            "baths": baths,
            "sf": sf,
            "purchase": purchase,
            "rehab": rehab,
            "total_cost": finance["total_cost"],
            "arv_raw": arv_raw,
            "arv_clamped": arv,
            "total_loan": finance["total_loan"],
            "cash_to_close": finance["cash_to_close"],
            "model_type": MODEL_TYPE,
            "model_version": MODEL_VERSION,
        },
    )

    # 8) Render result template
    return templates.TemplateResponse("result.html", context)


@app.post("/send-lead")
def send_lead(
    request: Request,
    name: str = Form(...),
    email: str = Form(...),
    phone: str = Form(...),
    comments: str | None = Form(None),
    address: str | None = Form(None),
    arv: str | None = Form(None),
    total_loan: str | None = Form(None),
    cash_to_close: str | None = Form(None),
):
    lead = {
        "name": name,
        "email": email,
        "phone": phone,
        "comments": comments,
        "address": address,
        "arv": arv,
        "total_loan": total_loan,
        "cash_to_close": cash_to_close,
        "source": "lmc_estimator",
        "client_ip": request.client.host if request.client else None,
        "user_agent": request.headers.get("user-agent"),
        "model_type": MODEL_TYPE,
        "model_version": MODEL_VERSION,
    }

    # 1) Log locally
    log_lead(lead)

    # 2) Optionally send email (no-op if SMTP not configured)
    send_lead_email(lead)

    # 3) Redirect back to home with success flag
    return RedirectResponse(url="/?lead=ok", status_code=303)


@app.get("/health")
def health():
    return {
        "ok": True,
        "model_target": META.get("target"),
        "r2_train": META.get("r2_train") or META.get("r2_train_log"),
        "r2_test": MODEL_R2,
        "model_type": MODEL_TYPE,
        "model_version": MODEL_VERSION,
    }

@app.get("/admin-stats", response_class=HTMLResponse)
def admin_stats(request: Request, token: str | None = None):
    # 1) Simple token check
    if not ADMIN_TOKEN or token != ADMIN_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized")

    # 2) Read logs
    events = read_jsonl(EVENTS_LOG)
    leads = read_jsonl(LEADS_LOG)

    total_estimates = sum(1 for e in events if e.get("event") == "estimate_submitted")
    total_leads = len(leads)

    # 3) Counts by day (ISO date string yyyy-mm-dd)
    def day_key(ts: str | None) -> str | None:
        if not ts:
            return None
        try:
            return ts[:10]
        except Exception:
            return None

    estimate_days = Counter()
    for e in events:
        if e.get("event") != "estimate_submitted":
            continue
        d = day_key(e.get("timestamp"))
        if d:
            estimate_days[d] += 1

    lead_days = Counter()
    for l in leads:
        d = day_key(l.get("timestamp"))
        if d:
            lead_days[d] += 1

    # Sort by date desc, limit
    estimate_days_list = sorted(estimate_days.items(), key=lambda x: x[0], reverse=True)[:14]
    lead_days_list = sorted(lead_days.items(), key=lambda x: x[0], reverse=True)[:14]

    # Last 20 leads
    leads_sorted = sorted(
        leads,
        key=lambda r: r.get("timestamp", ""),
        reverse=True
    )[:20]

    context = {
        "request": request,
        "total_estimates": total_estimates,
        "total_leads": total_leads,
        "estimate_days": estimate_days_list,
        "lead_days": lead_days_list,
        "leads": leads_sorted,
        "model_type": MODEL_TYPE,
        "model_version": MODEL_VERSION,
        "r2_test": MODEL_R2,
    }
    return templates.TemplateResponse("admin.html", context)