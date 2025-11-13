from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan, normal_ad

def run_diagnostics(X_df: pd.DataFrame, y: pd.Series, ols_results, artifact_dir: Path) -> dict:
    """
    X_df: design matrix (no constant column), numeric-only with one-hots (same as statsmodels fit)
    y:    target aligned with X_df
    ols_results: statsmodels RegressionResults (already fit on [const|X_df])
    """
    artifact_dir.mkdir(parents=True, exist_ok=True)
    out = {}

    # --- VIF on numeric design matrix (add const for VIF calc) ---
    X_for_vif = sm.add_constant(X_df, has_constant="add")
    vif_list = []
    for i, col in enumerate(X_for_vif.columns):
        try:
            vif_val = variance_inflation_factor(X_for_vif.values, i)
        except Exception:
            vif_val = np.nan
        vif_list.append({"feature": col, "VIF": float(vif_val) if np.isfinite(vif_val) else None})
    out["vif"] = vif_list

    # --- Breusch–Pagan for heteroskedasticity ---
    lm, lm_pvalue, fvalue, f_pvalue = het_breuschpagan(ols_results.resid, ols_results.model.exog)
    out["breusch_pagan"] = {"lm_pvalue": float(lm_pvalue), "f_pvalue": float(f_pvalue)}

    # --- Normality (Anderson–Darling) ---
    stat, p_norm = normal_ad(ols_results.resid)
    out["normality_ad_pvalue"] = float(p_norm)

    # Persist
    (artifact_dir / "diagnostics.json").write_text(json.dumps(out, indent=2))
    return out
