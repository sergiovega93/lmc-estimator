#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RF v2 Analysis Script (A–D)
Run from project root:  python rf_v2_analysis.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import load
from sklearn.tree import export_text
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.preprocessing import OneHotEncoder

from lmc_estimator_ml.ml.config import ARTIFACT_DIR, MODEL_VERSION


BASE_DIR = Path(__file__).resolve().parent
RF_ARTIFACT_DIR = ARTIFACT_DIR  # active model, should be v2_rf
OLS_ARTIFACT_DIR = RF_ARTIFACT_DIR.parent / "v1_ols"  # assumes artifacts/v1_ols exists


def load_rf_pipeline_and_meta():
    model_path = RF_ARTIFACT_DIR / "model.joblib"
    diag_path = RF_ARTIFACT_DIR / "diagnostics.json"
    meta_path = RF_ARTIFACT_DIR / "meta.json"

    print(f"\n[INFO] Loading RF pipeline from: {model_path}")
    pipeline = load(model_path)

    diagnostics = json.loads(diag_path.read_text())
    meta = json.loads(meta_path.read_text())

    print(
        f"[INFO] Loaded RF model_type={diagnostics.get('model_type')} "
        f"artifact_subdir={diagnostics.get('artifact_subdir')}"
    )
    print(f"[INFO] R² (train log): {meta.get('r2_train_log'):.4f}")
    print(f"[INFO] R² (test  log): {meta.get('r2_test_log'):.4f}")

    base_features = diagnostics["features"]
    print(f"[INFO] Base feature list ({len(base_features)}): {base_features}")
    return pipeline, diagnostics, meta, base_features


def load_ols_pipeline():
    model_path = OLS_ARTIFACT_DIR / "model.joblib"
    if not model_path.exists():
        print(
            f"[WARN] OLS model not found at {model_path}. "
            f"Skipping OLS comparison."
        )
        return None

    print(f"\n[INFO] Loading OLS pipeline from: {model_path}")
    pipeline = load(model_path)
    return pipeline


def get_preprocessor_and_rf(pipeline):
    """
    From a pipeline, return (preprocessor, rf_estimator).
    preprocessor: ColumnTransformer
    rf_estimator: RandomForestRegressor (the object with feature_importances_)
    """
    pre = None
    rf = pipeline

    if hasattr(pipeline, "steps"):
        # Sklearn Pipeline
        for name, step in pipeline.steps:
            if isinstance(step, ColumnTransformer):
                pre = step
            # Capture RF if present as a step
            if hasattr(step, "feature_importances_"):
                rf = step

    # Fall back: if no steps/ColumnTransformer found, assume bare RF
    return pre, rf


def get_model_feature_names(pipeline, base_features):
    """
    Use the fitted preprocessor to get the REAL expanded feature names
    that the RF sees, including one-hot encoded categorical variables.

    If we can't find a preprocessor, we fall back to base_features.
    """
    pre, rf = get_preprocessor_and_rf(pipeline)

    if pre is None:
        print(
            "\n[WARN] No ColumnTransformer found in pipeline. "
            "Using base_features as-is for importances."
        )
        return base_features

    feature_names = []

    # pre.transformers_ is a list of (name, transformer, columns)
    for name, trans, cols in pre.transformers_:
        if name == "num":
            # Numeric columns: after scaler, still one feature per original col
            feature_names.extend(list(cols))
        elif name == "cat":
            # Categorical pipeline: needs to dig out the OneHotEncoder
            cat_pipe = trans
            ohe = None
            if isinstance(cat_pipe, SkPipeline):
                for step_name, step_obj in cat_pipe.steps:
                    if isinstance(step_obj, OneHotEncoder):
                        ohe = step_obj
                        break
            elif isinstance(cat_pipe, OneHotEncoder):
                ohe = cat_pipe

            if ohe is None:
                print(
                    "\n[WARN] Could not find OneHotEncoder in cat pipeline; "
                    "categorical features will not be named precisely."
                )
                continue

            # cols is the list of original categorical columns
            ohe_feature_names = list(ohe.get_feature_names_out(cols))
            feature_names.extend(ohe_feature_names)

    if not feature_names:
        print(
            "\n[WARN] Preprocessor present but yielded no feature names; "
            "falling back to base_features."
        )
        return base_features

    print(
        f"[INFO] Expanded model feature list ({len(feature_names)} "
        f"vs base {len(base_features)}):"
    )
    print(feature_names)
    return feature_names


# === A) Feature Importances ===
def print_feature_importances(pipeline, expanded_feature_names):
    """
    Print feature importances using the FULL expanded feature name list
    (including one-hot encoded categorical variables).
    """
    _, rf = get_preprocessor_and_rf(pipeline)

    if not hasattr(rf, "feature_importances_"):
        print(
            "\n[WARN] Model does not expose feature_importances_. "
            "It might be wrapped in a TransformedTargetRegressor or similar."
        )
        return

    importances = rf.feature_importances_
    n_imp = len(importances)
    n_feat = len(expanded_feature_names)

    if n_imp != n_feat:
        print(
            f"\n[WARN] len(importances)={n_imp} != len(expanded_feature_names)={n_feat}. "
            "Falling back to generic indexing."
        )
        fi = (
            pd.DataFrame(
                {
                    "feature": [f"feat_{i}" for i in range(n_imp)],
                    "importance": importances,
                }
            )
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )
    else:
        fi = (
            pd.DataFrame(
                {"feature": expanded_feature_names, "importance": importances}
            )
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )

    print("\n===== A) Random Forest Feature Importances =====")
    print(fi.to_string(index=False))
    return fi


# === Helper: prediction with exp + clamp same as app.py ===
def predict_arv_from_df(pipeline, df, clamp=True):
    """Replicate app.py logic: model returns log(ARV); we exp() and clamp."""
    log_pred = pipeline.predict(df)
    arv = np.exp(log_pred)

    if clamp:
        total_cost = df["total_cost"].values
        lower = total_cost  # 1x
        upper = 2.0 * total_cost  # 2x
        arv = np.clip(arv, lower, upper)
    return arv


# === B) Simulate scenarios: flip vs heavy rehab vs new construction ===
def simulate_scenarios(pipeline, base_features):
    """
    Build a small DataFrame with 3–4 scenarios:
      - Normal cosmetic flip
      - Heavy rehab
      - New construction-like pattern
    We keep city, dom, lot_area_value, year_built, school_score fixed for now.
    Adjust 'base_city' to one of your real training cities.
    """
    print("\n===== B) Scenario Simulation (Flip vs New Construction) =====")

    base_city = "Atlanta, GA"  # <-- change to a known city from your data
    base = {
        "square_footage": 1800,
        "bed": 3,
        "baths": 2,
        "dom": 15,
        "lot_area_value": 7000,
        "year_built": 1995,
        "school_score": 7,
        "city": base_city,
    }

    scenarios = []

    # 1) Normal flip: 300k total, 60k rehab (20%)
    sc1 = base.copy()
    sc1["total_cost"] = 300_000
    sc1["rehab_ratio"] = 0.20
    sc1["label"] = "Normal Flip (20% rehab)"

    # 2) Heavy rehab: 300k total, 150k rehab (50%)
    sc2 = base.copy()
    sc2["total_cost"] = 300_000
    sc2["rehab_ratio"] = 0.50
    sc2["label"] = "Heavy Rehab (50% rehab)"

    # 3) New construction-like: 400k total, 320k rehab (80%)
    sc3 = base.copy()
    sc3["total_cost"] = 400_000
    sc3["rehab_ratio"] = 0.80
    sc3["label"] = "New Construction-like (80% rehab)"

    # 4) High-end new construction: 800k total, 640k rehab (80%)
    sc4 = base.copy()
    sc4["square_footage"] = 3200
    sc4["total_cost"] = 800_000
    sc4["rehab_ratio"] = 0.80
    sc4["label"] = "High-end New Construction (80% rehab)"

    scenarios.extend([sc1, sc2, sc3, sc4])

    df = pd.DataFrame(scenarios)

    # Ensure all required base feature columns exist, in correct order
    df_for_model = df[base_features]

    arv = predict_arv_from_df(pipeline, df_for_model, clamp=True)
    df["predicted_arv"] = arv
    df["arv_to_total_cost"] = df["predicted_arv"] / df["total_cost"]

    print(
        df[
            [
                "label",
                "total_cost",
                "rehab_ratio",
                "square_footage",
                "predicted_arv",
                "arv_to_total_cost",
            ]
        ].to_string(index=False)
    )
    return df


# === C) Tree path / explain one scenario ===
def inspect_single_tree_and_path(pipeline, base_features, example_row):
    """
    - Grab first tree in RF
    - Print its structure (on preprocessed features)
    - Show decision path for one scenario
    """
    pre, rf = get_preprocessor_and_rf(pipeline)

    if not hasattr(rf, "estimators_"):
        print(
            "\n[WARN] Cannot inspect trees; underlying estimator "
            "does not expose 'estimators_'."
        )
        return

    # Expanded feature names from the preprocessor
    expanded_feature_names = get_model_feature_names(pipeline, base_features)

    # Preprocess the example row using the same preprocessor
    if pre is not None:
        X_pre = pre.transform(example_row[base_features].to_frame().T)
    else:
        # No preprocessor -> assume example_row already matches RF input space
        X_pre = example_row[base_features].values.reshape(1, -1)

    tree = rf.estimators_[0]  # first tree
    print("\n===== C1) Text dump of first tree (truncated) =====")
    # export_text expects a dense array; handle sparse if needed
    if hasattr(X_pre, "toarray"):
        _ = X_pre.toarray()  # just to ensure it works; we don't really use it here
    print(export_text(tree, feature_names=expanded_feature_names, max_depth=4))

    if hasattr(X_pre, "toarray"):
        X_for_path = X_pre.toarray()
    else:
        X_for_path = X_pre

    path = tree.decision_path(X_for_path)
    leaf_id = tree.apply(X_for_path)[0]

    node_indicator = path.indices  # nodes visited
    print("\n===== C2) Decision path for first scenario =====")
    print(f"Leaf id: {leaf_id}")
    print(f"Visited node indices: {node_indicator.tolist()}")


# === D) Compare v2_rf vs v1_ols on the same scenarios ===
def compare_rf_vs_ols(rf_pipeline, ols_pipeline, scenarios_df, base_features):
    if ols_pipeline is None:
        print("\n[INFO] Skipping D) OLS comparison; no OLS model loaded.")
        return

    print("\n===== D) RF v2 vs OLS v1 on same scenarios =====")

    X = scenarios_df[base_features]

    # RF: log(ARV) -> exp -> clamp (like app.py)
    rf_arv = predict_arv_from_df(rf_pipeline, X, clamp=True)

    # OLS: depending on how you trained v1, you may need exp() or not.
    # If v1 was also trained on log(target), do the same. Otherwise, remove np.exp.
    ols_pred_raw = ols_pipeline.predict(X)
    # If OLS was trained on log(projected_value_ma), uncomment:
    # ols_arv = np.exp(ols_pred_raw)
    # If OLS was trained directly on dollar ARV, use raw:
    ols_arv = ols_pred_raw

    out = scenarios_df[["label", "total_cost", "rehab_ratio"]].copy()
    out["rf_v2_arv"] = rf_arv
    out["rf_v2_ratio"] = out["rf_v2_arv"] / out["total_cost"]
    out["ols_v1_arv"] = ols_arv
    out["ols_v1_ratio"] = out["ols_v1_arv"] / out["total_cost"]

    print(out.to_string(index=False))


def main():
    print(f"[INFO] Using MODEL_VERSION from config: {MODEL_VERSION}")
    print(f"[INFO] RF artifacts dir: {RF_ARTIFACT_DIR}")

    rf_pipeline, diagnostics, meta, base_features = load_rf_pipeline_and_meta()
    ols_pipeline = load_ols_pipeline()

    # Build expanded feature names from the actual fitted preprocessor
    expanded_feature_names = get_model_feature_names(rf_pipeline, base_features)

    # A) Feature importances
    fi = print_feature_importances(rf_pipeline, expanded_feature_names)

    # B) Scenario simulation
    scenarios_df = simulate_scenarios(rf_pipeline, base_features)

    # C) Tree path for first scenario
    inspect_single_tree_and_path(rf_pipeline, base_features, scenarios_df.iloc[0])

    # D) RF vs OLS on same scenarios
    compare_rf_vs_ols(rf_pipeline, ols_pipeline, scenarios_df, base_features)


if __name__ == "__main__":
    main()
