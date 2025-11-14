import numpy as np
import pandas as pd
from lmc_estimator_ml.ml.trainer import load_model
from lmc_estimator_ml.ml.config import ARTIFACT_DIR

MODEL = load_model(ARTIFACT_DIR)

beds = 3
baths = 2
sf = 1600
purchase = 150_000
rehab = 100_000
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
print("log_arv:", log_arv)
print("ARV:", arv)
