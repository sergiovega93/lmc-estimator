from __future__ import annotations

from pathlib import Path
from typing import Optional
from fastapi.staticfiles import StaticFiles
from urllib.parse import quote_plus
import json
import os
from datetime import datetime
import csv
import io
import numpy as np
import pandas as pd
from fastapi import FastAPI, Request, Form, HTTPException, Response
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from lmc_estimator_ml.ml.geo_adjustment import adjust_arv_for_geo

from lmc_estimator_ml.ml.trainer import load_model
from lmc_estimator_ml.ml.config import ARTIFACT_DIR
from collections import Counter
from datetime import datetime as dt
# ------------------------------
# FastAPI + templates
# ------------------------------
app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")
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
GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY", "")


# ------------------------------
# Logging setup (events + leads)
# ------------------------------
# Use Render persistent disk (mounted at /var/data) for durable logs
LOG_DIR = Path("/var/data/logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

EVENTS_LOG = LOG_DIR / "events.jsonl"
LEADS_LOG = LOG_DIR / "leads.jsonl"

# Admin token for /admin-stats
ADMIN_TOKEN = os.getenv("LMC_ADMIN_TOKEN")
print("Loaded ADMIN_TOKEN:", repr(ADMIN_TOKEN))

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

def _to_float(value: str | None) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except ValueError:
        return None

def send_lead_email(lead: dict) -> None:
    """
    Send lead details to sales via SMTP.

    Requires these env vars (set in Render dashboard):
      LMC_SMTP_HOST
      LMC_SMTP_PORT   (e.g. 587)
      LMC_SMTP_USER   (from-address / login)
      LMC_SMTP_PASS

    If any are missing, this function quietly does nothing.
    """
    host = os.getenv("LMC_SMTP_HOST")
    user = os.getenv("LMC_SMTP_USER")
    password = os.getenv("LMC_SMTP_PASS")
    port = int(os.getenv("LMC_SMTP_PORT", "587"))

    if not host or not user or not password:
        # No SMTP config → skip email sending
        print("send_lead_email: SMTP env vars missing, skipping email.")
        return

    try:
        import smtplib
        from email.message import EmailMessage

        msg = EmailMessage()
        msg["Subject"] = f"[LMC Estimator Lead] {lead.get('name')} - {lead.get('address')}"
        msg["From"] = user
        msg["To"] = "sergio@loanmountaincapital.com"

        # ---- Improved formatting ----

        def fmt_money(x):
            try:
                return f"${float(x):,.0f}"
            except:
                return "-" if x in (None, "", "None") else str(x)

        def fmt_triple(b, bt, sf):
            b = b or "-"
            bt = bt or "-"
            sf = sf or "-"
            return f"{b} / {bt} / {sf}"

        # Plain-text fallback (for logs or non-HTML clients)
        text_body = f"""New LMC Estimator lead

        Lead details
        Name:   {lead.get('name')}
        Email:  {lead.get('email')}
        Phone:  {lead.get('phone')}

        Property
        Address:                   {lead.get('address')}
        Beds / Baths / SF:         {fmt_triple(lead.get('beds'), lead.get('baths'), lead.get('sf'))}
        Purchase Price:            {fmt_money(lead.get('purchase'))}
        Rehab Budget:              {fmt_money(lead.get('rehab'))}
        ARV (clamped):             {fmt_money(lead.get('arv'))}
        Estimated Loan (70%):      {fmt_money(lead.get('total_loan'))}
        Estimated Cash to Close:   {fmt_money(lead.get('cash_to_close'))}

        Comments
        {lead.get('comments') or '-'}
        """

        # HTML Outlook-optimized body
        html_body = f"""\
        <body style="font-family:Segoe UI, Arial, sans-serif; font-size:14px;">

            <h3 style="margin-bottom:6px;">LMC Estimator — New Lead</h3>
        
            <h4 style="margin-bottom:2px;">Lead details</h4>
            <table cellpadding="3" cellspacing="0" style="margin-bottom:12px;">
              <tr><td><b>Name:</b></td><td>{lead.get('name')}</td></tr>
              <tr><td><b>Email:</b></td><td><a href="mailto:{lead.get('email')}">{lead.get('email')}</a></td></tr>
              <tr><td><b>Phone:</b></td><td>{lead.get('phone')}</td></tr>
            </table>
        
            <h4 style="margin-bottom:2px;">Property</h4>
            <table cellpadding="3" cellspacing="0" style="margin-bottom:12px;">
              <tr><td><b>Address:</b></td><td>{lead.get('address')}</td></tr>
              <tr><td><b>Beds / Baths / SF:</b></td>
                  <td>{fmt_triple(lead.get('beds'), lead.get('baths'), lead.get('sf'))}</td></tr>
              <tr><td><b>Purchase Price:</b></td><td>{fmt_money(lead.get('purchase'))}</td></tr>
              <tr><td><b>Rehab Budget:</b></td><td>{fmt_money(lead.get('rehab'))}</td></tr>
              <tr><td><b>ARV (clamped):</b></td><td>{fmt_money(lead.get('arv'))}</td></tr>
              <tr><td><b>Estimated Loan (70%):</b></td><td>{fmt_money(lead.get('total_loan'))}</td></tr>
              <tr><td><b>Estimated Cash to Close:</b></td><td>{fmt_money(lead.get('cash_to_close'))}</td></tr>
            </table>

            <h4 style="margin-bottom:2px;">Comments</h4>
            <p style="white-space:pre-wrap; margin:0 0 12px 0;">{lead.get('comments') or '-'}</p>

          </body>
        </html>
        """

        msg.set_content(text_body)
        msg.add_alternative(html_body, subtype="html")

        with smtplib.SMTP(host, port) as server:
            server.starttls()
            server.login(user, password)
            server.send_message(msg)

        print("send_lead_email: email sent successfully.")

    except Exception as e:
        # Don’t crash the request if email fails; just log.
        print(f"send_lead_email: failed to send email: {e!r}")

# ------------------------------
# Helper: build feature row from form
# ------------------------------
def build_features_from_form(
    beds: Optional[float],
    baths: Optional[float],
    sf: Optional[float],
    purchase: Optional[float],
    rehab: Optional[float],
    city: Optional[str],
    zipcode: Optional[str],
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
    city = (city or "Other").strip()
    zipcode = (zipcode or "Other").strip()

    total_cost = purchase + rehab
    # rehab_ratio = rehab / purchase_price, capped [0, 5]
    pp = purchase
    rb = rehab
    if pp and pp > 0:
        ratio = rb / pp
    else:
        ratio = 0.0

    # cap crazy outliers, same as preprocess_dataframe
    ratio = max(0.0, min(ratio, 5.0))

    rehab_ratio = ratio

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
        "zipcode": zipcode,
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
    ltc_limit: float = 0.90,
    placement_points: float = 0.0225,
    account_setup_fee: float = 1195.0,
    fixed_other_fees: float = 594.0,
) -> dict:
    """
    Compute:
      - total_cost
      - total_loan = min(ltv_limit * arv, ltc_limit * total_cost)
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

    # LTV cap
    ltv_cap = max(0.0, ltv_limit * arv)

    # LTC cap
    if total_cost > 0 and ltc_limit is not None:
        ltc_cap = max(0.0, ltc_limit * total_cost)
        total_loan = min(ltv_cap, ltc_cap)
    else:
        total_loan = ltv_cap

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
        "ltc_limit": ltc_limit,
    }

# ------------------------------
# Routes
# ------------------------------
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    error_message = request.query_params.get("error")
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "google_maps_api_key": GOOGLE_MAPS_API_KEY,
            "error_message": error_message,
        },
    )

@app.post("/estimate", response_class=HTMLResponse)
def estimate(
    request: Request,
    address: Optional[str] = Form(None),
    beds: float | None = Form(None),
    baths: float | None = Form(None),
    sf: float | None = Form(None),
    purchase: float | None = Form(None),
    rehab: float | None = Form(None),
    city: str | None = Form(None),
    state: str | None = Form(None),
    postal_code: str | None = Form(None),
    latitude: float | None = Form(None),
    longitude: float | None = Form(None),
):
    if not address:
        msg = "Please enter a property address to run an estimate."
        url = request.url_for("home") + f"?error={quote_plus(msg)}"
        return RedirectResponse(url, status_code=303)

    # 1) Normalize inputs
    beds = beds or 0
    baths = baths or 0
    sf = sf or 0
    purchase = purchase or 0
    rehab = rehab or 0
    city_value = city.strip() if city else "Other"
    state_value = state.strip() if state else "Unknown"
    zipcode_value = (postal_code or "Other").strip()

    # Quick sanity checks to avoid nonsense estimates
    if purchase < 0 or rehab < 0 or sf < 0:
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "error_message": "Values for square footage, purchase, and rehab must be zero or positive.",
            },
            status_code=400,
        )

    if beds < 0 or baths < 0:
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "error_message": "Beds and baths can’t be negative.",
            },
            status_code=400,
        )


    # 2) Build model input row and get total_cost
    X, total_cost = build_features_from_form(
        beds=beds,
        baths=baths,
        sf=sf,
        purchase=purchase,
        rehab=rehab,
        city=city_value,     # later you can expose city/state in the form
        zipcode=zipcode_value,
    )

    # 3) Predict log(ARV) and convert back to dollars
    log_arv = float(MODEL.predict(X)[0])
    arv_raw = float(np.exp(log_arv))

    # We no longer clamp ARV to [1×, 2×] total_cost here.
    arv = arv_raw
    clamped = False
    clamped_ratio = None


    # >>> ADD THIS BLOCK: GEO ADJUSTMENT <<<
    geo_factor = 1.0
    location_status = "geo_disabled"
    zhvi_used = None
    try:
        arv_geo, geo_factor, location_status, zhvi_used = adjust_arv_for_geo(
            arv=arv,
            total_cost=total_cost,
            zipcode=zipcode_value,
        )
        arv = arv_geo
    except Exception as e:
        # Fail-safe: never break the estimator because of geo heuristics
        print(f"[WARN] Geo adjustment failed: {e}")

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
        # NEW: geo metadata for logs / UI
        "location_status": location_status,
        "geo_factor": geo_factor,
        "zipcode": zipcode_value,
        "zhvi_used": zhvi_used,
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
            "arv_final": arv,
            "total_loan": finance["total_loan"],
            "cash_to_close": finance["cash_to_close"],
            "model_type": MODEL_TYPE,
            "model_version": MODEL_VERSION,
            "city": city_value,
            "state": state_value,
            "postal_code": postal_code,
            "latitude": latitude,
            "longitude": longitude,
            "location_status": location_status,
            "geo_factor": geo_factor,
            "zipcode": zipcode_value,
            "zhvi_used": zhvi_used,
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
    purchase: str | None = Form(None),
    rehab: str | None = Form(None),
    beds: str | None = Form(None),
    baths: str | None = Form(None),
    sf: str | None = Form(None),
    location_status: str | None = Form(None),
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
        "purchase": purchase,
        "rehab": rehab,
        "beds": beds,
        "baths": baths,
        "sf": sf,
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

    # 3) Re-render the result page so user keeps seeing the numbers
    context = {
        "request": request,
        "address": address,
        "beds": _to_float(beds),
        "baths": _to_float(baths),
        "sf": _to_float(sf),
        "purchase": _to_float(purchase),
        "rehab": _to_float(rehab),
        "total_cost": None,  # not shown in UI currently
        "arv": _to_float(arv),
        "arv_clamped": False,
        "clamped_ratio": None,
        "total_loan": _to_float(total_loan),
        "initial_advance": None,  # not shown in UI; safe to omit
        "placement_fee": None,    # same
        "cash_to_close": _to_float(cash_to_close),
        "ltv_limit": None,
        "model_r2": MODEL_R2,
        "model_type": MODEL_TYPE,
        "model_version": MODEL_VERSION,
        "location_status": location_status or "geo_disabled",
        "zipcode": None,
        "zhvi_used": None,
        "lead_sent": True,
    }

    return templates.TemplateResponse("result.html", context)


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

import csv
import io

@app.get("/admin-export")
def admin_export(
    token: str | None = None,
    kind: str = "events",
):
    """
    Export full history as CSV.
    kind = 'events' or 'leads'.
    """
    if not ADMIN_TOKEN or token != ADMIN_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized")

    if kind == "events":
        path = EVENTS_LOG
        filename = "estimator_events.csv"
    elif kind == "leads":
        path = LEADS_LOG
        filename = "estimator_leads.csv"
    else:
        raise HTTPException(status_code=400, detail="Invalid kind; use 'events' or 'leads'")

    rows = read_jsonl(path)
    if not rows:
        csv_data = ""
    else:
        # union of all keys across rows
        fieldnames = sorted({k for row in rows for k in row.keys()})
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        csv_data = buf.getvalue()

    return Response(
        content=csv_data,
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )

@app.get("/terms", response_class=HTMLResponse)
async def terms(request: Request):
    return templates.TemplateResponse(
        "terms.html",
        {
            "request": request,
            "last_updated": "2025-12-03",  # optional
        },
    )
