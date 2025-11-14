from __future__ import annotations

from lmc_estimator_ml.ml import data_loader as dl, trainer
from lmc_estimator_ml.ml.config import DEFAULT_EXCEL_PATH


def main():
    print("Loading data from:", DEFAULT_EXCEL_PATH)
    df = dl.load_comps(DEFAULT_EXCEL_PATH)
    df = dl.basic_clean(df)
    df = dl.normalize_lot_area(df)
    df = dl.filter_eligible(df)
    df = dl.filter_recent(df)

    print("Rows after filtering:", len(df))

    meta = trainer.train_rf(df, artifact_subdir="v2_rf")
    print("RF training complete. Meta:", meta)


if __name__ == "__main__":
    main()
