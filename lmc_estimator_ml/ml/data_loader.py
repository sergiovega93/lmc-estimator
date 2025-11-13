from __future__ import annotations
import re
import pandas as pd
from pathlib import Path
from .config import DEFAULT_EXCEL_PATH, TARGET_COL, DATE_COL, ID_COL, MONTH_WINDOW

def load_comps(excel_path: Path | str = DEFAULT_EXCEL_PATH) -> pd.DataFrame:
    path = Path(excel_path)
    if not path.exists():
        raise FileNotFoundError(f"Comps Excel not found: {path}")

    # Try intelligent sheet detection
    xls = pd.ExcelFile(path)
    sheet_name = xls.sheet_names[0]
    if any(s.lower().startswith(('comp', 'sales', 'export', 'property')) for s in xls.sheet_names):
        sheet_name = [s for s in xls.sheet_names if s.lower().startswith(('comp', 'sales', 'export', 'property'))][0]

    df = pd.read_excel(path, sheet_name=sheet_name)

    # --- normalize columns (robust) ---
    def _norm_col(name: str) -> str:
        s = re.sub(r'[^0-9a-zA-Z]+', '_', str(name).strip().lower())
        return re.sub(r'_+', '_', s).strip('_')

    df.columns = [_norm_col(c) for c in df.columns]

    # parse known date column if present
    if "fecha_de_modificacion" in df.columns:
        df["fecha_de_modificacion"] = pd.to_datetime(df["fecha_de_modificacion"], errors="coerce")

    # safe sort / de-dup
    if DATE_COL in df.columns:
        df = df.sort_values(DATE_COL, ascending=False)
    if ID_COL in df.columns:
        df = df.drop_duplicates(subset=[ID_COL])

    return df

def filter_recent(df: pd.DataFrame, months: int = MONTH_WINDOW) -> pd.DataFrame:
    if DATE_COL not in df.columns or df[DATE_COL].isna().all():
        # No date column — just return df unchanged
        return df
    cutoff = pd.Timestamp.now().normalize() - pd.DateOffset(months=months)
    return df.loc[df[DATE_COL] >= cutoff].copy()


def filter_eligible(df: pd.DataFrame) -> pd.DataFrame:
    # Keep rows with a usable proxy target
    if "projected_value_ma" in df.columns:
        df = df[df["projected_value_ma"].apply(pd.to_numeric, errors="coerce").notna()].copy()
        df["projected_value_ma"] = pd.to_numeric(df["projected_value_ma"], errors="coerce")
        df = df[df["projected_value_ma"] > 0]
    # (Optional) If you have an internal flag like 'passed_pipeline' == 'Yes', enforce it here:
    # if "passed_pipeline" in df.columns:
    #     df = df[df["passed_pipeline"].astype(str).str.lower().isin(["yes","true","1"])]
    return df

def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    # Remove impossible or zero targets
    df = df.loc[df[TARGET_COL].astype(float) > 0].copy()
    # Strip whitespace for object columns
    for c in df.select_dtypes(include='object').columns:
        df[c] = df[c].astype(str).str.strip()
    return df.reset_index(drop=True)

def normalize_lot_area(df: pd.DataFrame) -> pd.DataFrame:
    """Convert lot_area_value to square feet whenever units are acres."""
    if "lot_area_value" in df.columns and "lot_area_units" in df.columns:
        # Work on a copy to avoid SettingWithCopyWarning
        df = df.copy()
        df["lot_area_units"] = df["lot_area_units"].astype(str).str.lower().str.strip()
        mask_acres = df["lot_area_units"].str.contains("acre", na=False)
        mask_sqft = df["lot_area_units"].str.contains("foot|ft|sq", na=False)
        # Convert acres → square feet (1 acre = 43,560 sq ft)
        df.loc[mask_acres, "lot_area_value"] = (
            pd.to_numeric(df.loc[mask_acres, "lot_area_value"], errors="coerce") * 43560
        )
        # Ensure numeric dtype
        df["lot_area_value"] = pd.to_numeric(df["lot_area_value"], errors="coerce")
        # Standardize units string for consistency
        df.loc[mask_acres, "lot_area_units"] = "sqft"
    return df
