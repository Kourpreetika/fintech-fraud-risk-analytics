"""
Data loading, validation, and cleaning for the fraud risk analytics
pipeline. See notebooks/01_data_understanding.ipynb and
notebooks/02_data_quality.ipynb for the investigation that motivated each
decision here.
"""

from pathlib import Path

import pandas as pd

RAW_PATH = Path(__file__).resolve().parent.parent / "data" / "raw" / "transactions.csv"
VALID_CLASS_VALUES = {0, 1}


def load_raw(path: Path = RAW_PATH) -> pd.DataFrame:
    """Load the raw dataset exactly as sourced (see data/README.md)."""
    return pd.read_csv(path)


def validate(df: pd.DataFrame) -> dict:
    """
    Data-quality checks (Phase 3). Returns a dict of findings — this
    function only reports, it never mutates or drops rows.
    """
    findings = {}

    missing = df.isnull().sum()
    findings["missing_values"] = {
        "total": int(missing.sum()),
        "by_column": missing[missing > 0].to_dict(),
        "pct_by_column": (missing[missing > 0] / len(df) * 100).round(3).to_dict(),
    }

    findings["duplicate_rows"] = {
        "count": int(df.duplicated().sum()),
        "fraud_among_duplicates": int(df[df.duplicated()]["Class"].sum()) if "Class" in df else None,
    }

    findings["dtypes"] = df.dtypes.astype(str).to_dict()

    findings["invalid_values"] = {
        "negative_amount": int((df["Amount"] < 0).sum()),
        "negative_time": int((df["Time"] < 0).sum()),
        "class_outside_0_1": int((~df["Class"].isin(VALID_CLASS_VALUES)).sum()) if "Class" in df else None,
    }

    q1, q3 = df["Amount"].quantile([0.25, 0.75])
    iqr = q3 - q1
    upper_fence = q3 + 1.5 * iqr
    findings["amount_outliers_iqr"] = {
        "upper_fence": round(float(upper_fence), 2),
        "count_above_fence": int((df["Amount"] > upper_fence).sum()),
        "pct_above_fence": round(100 * (df["Amount"] > upper_fence).mean(), 2),
    }

    return findings


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the cleaning decisions justified in notebooks/02_data_quality.ipynb:

    1. Rename Class -> is_fraud (this project's target variable name).
    2. Drop exact duplicate rows. With 28 continuous PCA components plus
       Time and Amount, two genuinely different transactions matching on
       every single field by chance is effectively impossible — these are
       treated as duplicate records (e.g. re-submitted/re-logged events),
       not as informative repeat transactions. This is a validated
       decision, not a blanket "drop duplicates" reflex: outliers and
       legitimately repeated values are NOT touched by this function.
    3. Enforce dtypes: Amount/Time numeric, is_fraud binary int.

    No outliers are removed here — Phase 3/4 treat large transaction
    amounts as a signal worth investigating, not noise to discard.
    """
    out = df.rename(columns={"Class": "is_fraud"}).copy()
    out = out.drop_duplicates().reset_index(drop=True)
    out["Amount"] = pd.to_numeric(out["Amount"])
    out["Time"] = pd.to_numeric(out["Time"])
    out["is_fraud"] = out["is_fraud"].astype(int)
    out.insert(0, "transaction_id", [f"TXN{i:07d}" for i in range(1, len(out) + 1)])
    return out
