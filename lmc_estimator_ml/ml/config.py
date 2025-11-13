from pathlib import Path

ARTIFACT_DIR = Path(__file__).resolve().parents[1] / "artifacts"
DEFAULT_EXCEL_PATH = Path(__file__).resolve().parents[1] / "Comps Database.xlsx"

# ===== Target =====
# For now we’ll train to MA’s projected value as a proxy for ARV.
# When true ARV labels (e.g., actual post-rehab sales) become available,
# switch TARGET_COL to that column.
TARGET_COL = "projected_value_ma"

# Used only for rolling window filter (not as a feature)
DATE_COL = "fecha_de_modificacion"  # change if you later add a better timestamp (e.g., 'fecha_de_modificacion')
ID_COL = "loan_number"  # or 'api_loan_id' if more stable

# ===== Features (allow-list only; prevents leakage) =====
# Numeric (robust to blanks via median impute)
NUMERIC_FEATURES = [
    "square_footage", "bed", "baths", "dom",
    "lot_area_value", "year_built", "school_score",
    "total_cost", "rehab_ratio", #"purchase_price_ma", #"rehab_budget_ma"
]

# Categorical (one-hot). Keep short lists—no addresses/URLs/IDs.
CATEGORICAL_FEATURES = [
    "city",
    #"state",
    #"zipcode",
    #"type",
]

# IMPORTANT: we DO NOT include:
#   - 'price' (that’s borrower purchase price; we’ll use it only for finance calc)
#   - 'price_per_square_foot' (leaks target-like info; remove to avoid circularity)
#   - identifiers/addresses/links (loan_number, api_loan_id, street_address, zillow_link, source_file_path, comp_property)
#   - school name text columns (e.g., 'high_school') to avoid high-cardinality sparse noise

MONTH_WINDOW = 6
TEST_SIZE = 0.2
RANDOM_STATE = 42
P_VALUE_THRESHOLD = 0.05
