from typing import Optional

from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from lmc_estimator_ml.service import estimate_loan_terms

app = FastAPI()
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    # Your templates/index.html should render the input form with:
    # - address, beds, baths, sf, purchase, rehab
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/estimate", response_class=HTMLResponse)
def estimate(
    request: Request,
    address: str = Form(...),
    beds: Optional[float] = Form(None),
    baths: Optional[float] = Form(None),
    sf: Optional[float] = Form(None),
    purchase: Optional[float] = Form(None),
    rehab: Optional[float] = Form(None),
):
    # Use the regression engine + loan logic from lmc_estimator_ml.service
    results = estimate_loan_terms(
        address=address,
        beds=beds,
        baths=baths,
        sf=sf,
        purchase=purchase,
        rehab=rehab,
    )

    arv = results["arv"]
    total_cost = results["total_cost"]
    total_loan = results["total_loan"]
    max_loan_ltv = results["max_loan_ltv"]
    max_loan_ltc = results["max_loan_ltc"]
    cash_to_close = results["cash_to_close"]

    html = f"""
    <h1>Preliminary Estimate</h1>
    <p><b>Address:</b> {address}</p>
    <p><b>Beds/Baths/SF:</b> {beds or '-'} / {baths or '-'} / {sf or '-'}</p>

    <h2>Valuation</h2>
    <p><b>Estimated ARV:</b> ${arv:,.0f}</p>
    <p><b>Total Project Cost (Purchase + Rehab):</b> ${total_cost:,.0f}</p>

    <h2>Loan Sizing (Internal Logic Approximation)</h2>
    <p><b>Max Loan @ 70% ARV:</b> ${max_loan_ltv:,.0f}</p>
    <p><b>Max Loan @ 90% LTC:</b> ${max_loan_ltc:,.0f}</p>
    <p><b>Estimated Total Loan (min of the two):</b> ${total_loan:,.0f}</p>

    <h2>Estimated Cash to Close</h2>
    <p><b>Estimated Cash to Close:</b> ${cash_to_close:,.0f}</p>

    <p style="margin-top:16px;"><a href="/">Back</a></p>
    """
    return HTMLResponse(html)


@app.get("/health")
def health():
    return {"ok": True}
