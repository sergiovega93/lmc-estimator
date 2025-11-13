from __future__ import annotations
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import RidgeCV  # swap to RidgeCV if you want
from sklearn.pipeline import Pipeline

from .config import (
    ARTIFACT_DIR,
    TARGET_COL,
    TEST_SIZE,
    RANDOM_STATE,
    P_VALUE_THRESHOLD,
)
from .features import build_preprocessor, split_X_y
from .diagnostics import run_diagnostics
USE_LOG_TARGET = True  # set False to go back to linear target

ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)


def backward_elimination(X: pd.DataFrame, y: pd.Series, p_threshold: float = P_VALUE_THRESHOLD) -> dict:
    """
    Safety net significance on a simplified auto-built design matrix (not used if the
    main OLS on the preprocessed matrix succeeds). Kept as a fallback.
    """
    try:
        X_ = X.copy()

        # try numeric coercion
        for c in X_.columns:
            if X_[c].dtype == object:
                coerced = pd.to_numeric(X_[c], errors="coerce")
                if coerced.notna().mean() >= 0.10:
                    X_[c] = coerced

        # one-hot objects
        obj_cols = X_.select_dtypes(include=["object"]).columns.tolist()
        if obj_cols:
            X_ = pd.get_dummies(X_, columns=obj_cols, drop_first=True)

        # numeric only
        X_ = X_.apply(pd.to_numeric, errors="coerce").dropna(axis=1, how="all")
        if X_.empty:
            return {"n_obs": len(y), "r2": np.nan, "adj_r2": np.nan, "significant_features": [], "pvalues": {}}
        X_ = X_.loc[:, X_.nunique(dropna=True) > 1]
        if X_.empty:
            return {"n_obs": len(y), "r2": np.nan, "adj_r2": np.nan, "significant_features": [], "pvalues": {}}

        # fill NaNs
        X_ = X_.fillna(X_.median(numeric_only=True)).fillna(0.0)

        # align y
        y_ = pd.to_numeric(y, errors="coerce")
        mask = y_.notna()
        X_, y_ = X_.loc[mask], y_.loc[mask]

        # OLS + BE
        X_sm = sm.add_constant(X_, has_constant="add").astype(float)
        y_ = y_.astype(float)
        cols = list(X_sm.columns)
        model = RidgeCV(alphas=[0.1, 1.0, 3.0, 10.0])
        while True:
            pvals = model.pvalues
            worst = pvals.idxmax()
            if worst != "const" and pvals[worst] > p_threshold:
                cols.remove(worst)
                model = sm.OLS(y_, X_sm[cols]).fit()
            else:
                break

        # write summary
        (ARTIFACT_DIR / "ols_summary.txt").write_text(model.summary().as_text())

        return {
            "n_obs": int(model.nobs),
            "r2": float(model.rsquared),
            "adj_r2": float(model.rsquared_adj),
            "significant_features": [c for c in cols if c != "const"],
            "pvalues": {k: float(v) for k, v in model.pvalues.items()},
        }
    except Exception:
        return {"n_obs": len(y), "r2": np.nan, "adj_r2": np.nan, "significant_features": [], "pvalues": {}}


def train(df: pd.DataFrame, artifact_dir: Path = ARTIFACT_DIR) -> dict:
    artifact_dir.mkdir(parents=True, exist_ok=True)

    # Split and build pipeline
    X, y = split_X_y(df, TARGET_COL)
    # ADD THIS BLOCK
    if USE_LOG_TARGET:
        # train on log1p(target)
        y = np.log1p(pd.to_numeric(y, errors="coerce"))
    preprocessor = build_preprocessor()

    # ---- choose model ----
    # from sklearn.linear_model import RidgeCV
    # model = RidgeCV(alphas=[0.1, 1.0, 10.0])
    model = RidgeCV()

    pipe = Pipeline(steps=[("pre", preprocessor), ("est", model)])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    # Fit pipeline
    pipe.fit(X_train, y_train)
    r2_train = pipe.score(X_train, y_train)
    r2_test = pipe.score(X_test, y_test)

    # ==== Significance & diagnostics on the SAME transformed matrix ====
    try:
        # Use preprocessor (already fitted in pipe.fit) to transform train
        X_train_mat = pipe.named_steps["pre"].transform(X_train)
        feature_names = pipe.named_steps["pre"].get_feature_names_out()

        # Dense DataFrame for statsmodels
        X_train_dense = X_train_mat.toarray() if hasattr(X_train_mat, "toarray") else X_train_mat
        X_train_df = pd.DataFrame(X_train_dense, columns=feature_names, index=X_train.index)

        # OLS on transformed design
        X_sm = sm.add_constant(X_train_df, has_constant="add")
        y_sm = pd.to_numeric(y_train, errors="coerce").astype(float)
        ols_model = sm.OLS(y_sm, X_sm).fit(cov_type="HC3")

        # Write summary + diagnostics
        (artifact_dir / "ols_summary.txt").write_text(ols_model.summary().as_text())
        run_diagnostics(X_train_df, y_sm, ols_model, artifact_dir)

        # Significant features by p-value
        pvals = ols_model.pvalues
        sig_feats = [k for k, v in pvals.items() if k != "const" and float(v) <= P_VALUE_THRESHOLD]
        sig = {
            "n_obs": int(ols_model.nobs),
            "r2": float(ols_model.rsquared),
            "adj_r2": float(ols_model.rsquared_adj),
            "significant_features": sig_feats,
            "pvalues": {k: float(v) for k, v in pvals.items()},
        }
    except Exception:
        # Fallback snapshot if anything above fails
        sig = backward_elimination(X_train, y_train)

    # Persist model + meta
    joblib.dump(pipe, artifact_dir / "model.joblib")
    meta = {
        "target": TARGET_COL,
        "r2_train": float(r2_train),
        "r2_test": float(r2_test),
        "significance": sig,
        "feature_sets": {
            "numeric": [n for n in feature_names if n.startswith("num__")] if "feature_names" in locals() else [],
            "categorical": [n for n in feature_names if n.startswith("cat__")] if "feature_names" in locals() else [],
        },
    }
    (artifact_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def load_model(artifact_dir: Path = ARTIFACT_DIR):
    return joblib.load(artifact_dir / "model.joblib")
