from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from .config import NUMERIC_FEATURES, CATEGORICAL_FEATURES


# --------- helpers ---------
def collapse_rare(series: pd.Series, min_count: int = 25) -> pd.Series:
    """
    Replace infrequent categories (count < min_count) with 'Other'.
    Works on string/object columns; leaves NaNs as-is (imputer will handle).
    """
    vc = series.value_counts(dropna=True)
    keep = set(vc[vc >= min_count].index)
    return series.where(series.isin(keep), "Other")


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Light deterministic transforms that happen BEFORE sklearn:
      - Collapse rare 'city' / 'zipcode' categories.
      - Create engineered numeric features: total_cost, rehab_ratio.
    """
    df = df.copy()

    # --- rare-category collapse ---
    if "city" in df.columns:
        df["city"] = collapse_rare(df["city"], min_count=25)
    if "zipcode" in df.columns and ("zipcode" in CATEGORICAL_FEATURES):
        df["zipcode"] = collapse_rare(df["zipcode"], min_count=25)

    # --- engineered features ---
    # total_cost = purchase + rehab
    # rehab_ratio = rehab / purchase (capped, avoids div-by-zero)
    if "purchase_price_ma" in df.columns and "rehab_budget_ma" in df.columns:
        pp = pd.to_numeric(df["purchase_price_ma"], errors="coerce")
        rb = pd.to_numeric(df["rehab_budget_ma"], errors="coerce")
        df["total_cost"] = pp + rb

        ratio = rb / pp.replace(0, np.nan)
        ratio = ratio.clip(lower=0, upper=5)  # cap crazy outliers
        df["rehab_ratio"] = ratio

    return df


# --------- sklearn preprocessor ---------
def build_preprocessor() -> ColumnTransformer:
    """
    ColumnTransformer:
      - numeric: median impute + standardize (StandardScaler)
      - categorical: most-frequent impute + OneHotEncoder(drop='first')
    This reduces multicollinearity and improves condition number for OLS.
    """
    num_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    cat_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(
            handle_unknown="ignore",
            drop="first"  # <--- drop baseline dummy per category group
        )),
    ])

    pre = ColumnTransformer(
        transformers=[
            ("num", num_pipe, NUMERIC_FEATURES),
            ("cat", cat_pipe, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )
    return pre


# --------- X / y split ---------
def split_X_y(df: pd.DataFrame, target_col: str):
    """
    Apply pre-DataFrame transforms, then split target from features.
    """
    df = preprocess_dataframe(df)
    y = df[target_col]
    X = df.drop(columns=[target_col])
    return X, y
