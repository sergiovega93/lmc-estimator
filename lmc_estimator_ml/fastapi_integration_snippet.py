# === FastAPI wiring example (add to your existing app.py) ===
from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Optional
from pathlib import Path
from lmc_estimator_ml.ml.predict import predict as ml_predict
from lmc_estimator_ml.service import train_from_excel

app = FastAPI()

class EstimateRequest(BaseModel):
    address: Optional[str] = None
    beds: Optional[float] = None
    baths: Optional[float] = None
    sqft: Optional[float] = None
    lot_size: Optional[float] = None
    year_built: Optional[float] = None
    property_type: Optional[str] = None
    zip: Optional[str] = None
    city: Optional[str] = None
    county: Optional[str] = None
    purchase_price: Optional[float] = Field(0)
    rehab_budget: Optional[float] = Field(0)

@app.post("/train")
def train_endpoint(excel_path: str = "Comps Database.xlsx"):
    meta = train_from_excel(Path(excel_path))
    return {"status": "ok", "meta": meta}

@app.post("/estimate")
def estimate(req: EstimateRequest):
    return ml_predict(req.model_dump())
