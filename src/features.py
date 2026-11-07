"""
Feature engineering (Phase 6) — with the leakage reasoning (Phase 7) kept
next to each feature rather than in a separate document, so it's obvious
*why* a feature is safe to use at the point it's created.

Ground rule applied throughout: a feature is only included if it could
realistically be computed from information available *before* a fraud
decision is made on that transaction. Nothing here uses `is_fraud` (or
anything derived from a fraud investigation outcome) as an input.

Features NOT created, and why:
  - transactions_per_customer, amount_vs_customer_average — this dataset
    has no customer/card identifier (see data/README.md), so any
    "per-customer" feature would have to be fabricated. Faking an ID
    would make these features meaningless at best and misleading at
    worst, so they are correctly out of scope for this dataset rather
    than approximated.
  - day_of_week — `Time` is seconds since the first transaction in the
    dataset, not a real timestamp, so there's no genuine calendar date to
    derive a weekday from. `elapsed_day` (below) is the honest version of
    this: which of the two observed days a transaction fell on.
"""

import numpy as np
import pandas as pd


def add_amount_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    log_amount: log1p(Amount). Amount is available the instant a
    transaction is initiated, so this is leakage-free — it's just a
    variance-stabilizing transform of an already-available field (Amount
    is heavily right-skewed; see notebooks/03_eda.ipynb).
    """
    out = df.copy()
    out["log_amount"] = np.log1p(out["Amount"])
    return out


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    hour_of_day, elapsed_day: both derived purely from `Time`, which (like
    Amount) is known at transaction time. hour_of_day assumes the dataset
    starts at hour 0 — an approximation flagged in data/README.md, not a
    verified clock time, since `Time` is relative-seconds, not a
    timestamp.
    """
    out = df.copy()
    out["hour_of_day"] = (out["Time"] // 3600 % 24).astype(int)
    out["elapsed_day"] = (out["Time"] // 86400).astype(int)
    return out


def add_frequency_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    transactions_per_hour_bucket: how many transactions in the whole
    dataset fall in the same hour_of_day bucket as this one. This is a
    property of the surrounding transaction stream (a volume/context
    signal), computed without reference to any transaction's fraud
    outcome — so it carries no label information and is leakage-free.

    Caveat (documented, not hidden): this is a symmetric, whole-dataset
    count. A real-time system could only count *past* transactions in the
    current hour, not future ones — so this is a reasonable offline proxy
    for "is this an unusually busy/quiet hour", not a feature you could
    reproduce identically in a live system. Flagged here as a known
    simplification of an offline research pipeline vs. production.
    """
    out = df.copy()
    if "hour_of_day" not in out.columns:
        out = add_time_features(out)
    counts = out.groupby("hour_of_day")["hour_of_day"].transform("count")
    out["transactions_per_hour_bucket"] = counts
    return out


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all feature engineering steps in one call."""
    out = add_amount_features(df)
    out = add_time_features(out)
    out = add_frequency_features(out)
    return out


# Columns that are safe model inputs (post feature engineering).
# V1-V28 are PCA components of the ORIGINAL pre-transaction features
# (merchant/location/device-type signals, per the dataset's own
# documentation) — computed independently of the fraud outcome, so they
# are legitimate predictive inputs, not leakage.
MODEL_FEATURE_COLUMNS = (
    [f"V{i}" for i in range(1, 29)]
    + ["Amount", "log_amount", "hour_of_day", "elapsed_day", "transactions_per_hour_bucket"]
)
