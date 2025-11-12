from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

app = FastAPI()
templates = Jinja2Templates(directory="templates")

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
    # Placeholder logic for now — we'll wire comps next
    sf = sf or 0
    purchase = purchase or 0
    rehab = rehab or 0

    # Simple demo assumptions:
    psf = 200  # placeholder $/SF; next step will compute from comps
    arv_base = sf * psf
    max_loan_ltv = 0.70 * arv_base
    max_loan_ltc = 0.85 * (purchase + rehab)
    max_loan = min(max_loan_ltv, max_loan_ltc)
    cash_to_close = max(0, purchase + rehab - max_loan)

    html = f"""
    <h1>Preliminary Estimate</h1>
    <p><b>Address:</b> {address}</p>
    <p><b>Beds/Baths/SF:</b> {beds or '-'} / {baths or '-'} / {sf:,.0f}</p>
    <p><b>ARV (placeholder):</b> ${arv_base:,.0f}</p>
    <p><b>Max Loan (min of 70% ARV, 85% LTC):</b> ${max_loan:,.0f}</p>
    <p><b>Estimated Cash to Close:</b> ${cash_to_close:,.0f}</p>
    <p style="margin-top:16px;"><a href="/">Back</a></p>
    """
    return HTMLResponse(html)

@app.get("/health")
def health():
    return {"ok": True}
