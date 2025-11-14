# LMC Estimator — ML Layer (Starter)

What this gives you:
- Excel -> clean -> 6-month window -> train LinearRegression
- Stats snapshot via statsmodels (p-values + backward elimination)
- Artifacts persisted to ./artifacts (model.joblib, meta.json)
- FastAPI snippets for /train and /estimate

How to use locally:
1) `pip install -r requirements.txt`
2) Put your Excel as `Comps Database.xlsx` in this folder (or pass explicit path).
3) Train: `python -m lmc_estimator_ml.service` (or call POST /train from FastAPI).
4) Predict: POST /estimate with JSON fields (beds, baths, sqft, etc.).

Customize:
- Edit `ml/config.py` to match your 32 variables and column names.
- Ensure `TARGET_COL` = ARV column (e.g., 'sale_price'); `DATE_COL` = close/sale date.
- Expand NUMERIC_FEATURES & CATEGORICAL_FEATURES to your real schema.

Notes on significance:
- `trainer.backward_elimination` builds a design matrix and iteratively drops features with p>0.05; results saved in meta.json -> 'significance'.
- The deployed model used for predictions is scikit-learn LinearRegression on preprocessed features. You can swap to Lasso/Ridge later if needed.

Weekly refresh:
- Automate calling POST /train (Render cron, GitHub Action, or a small Windows Task) to re-train with a rolling 6-month window.
