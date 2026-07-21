"""
pipeline/clean_output.py
==========================
The same cleaning rules as your Colab "CSV Cleaning Script", minus the
Colab-specific upload/download bits, so it can run headlessly as part of
run_pipeline.py step 5.
"""

from __future__ import annotations

import logging

import pandas as pd

log = logging.getLogger(__name__)

NUMERIC_COLS = ["score", "ratings_count", "installs", "days_since_released", "months_since_launch"]
BOOLEAN_COLS = ["free", "offers_iap"]
STRING_COLS = [
    "bundle_id", "app_name", "description", "summary", "genreid",
    "content_rating", "developerid", "developer",
]


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Lowercase/strip columns, coerce numeric & boolean types, collapse
    whitespace in text columns, and drop rows with no usable bundle_id.
    Any columns from the enrichment step (flags, bins, etc.) that aren't in
    the lists above are passed through untouched.
    """
    df = df.copy()
    df.columns = df.columns.str.lower().str.strip()

    for col in NUMERIC_COLS:
        if col in df.columns:
            before = df[col].notna().sum()
            df[col] = pd.to_numeric(df[col], errors="coerce")
            after = df[col].notna().sum()
            if before - after:
                log.info("  %-22s: %d values were non-numeric -> NULL", col, before - after)

    for col in BOOLEAN_COLS:
        if col in df.columns:
            df[col] = df[col].astype(str).str.lower().isin(["true", "1", "yes", "t"])

    for col in STRING_COLS:
        if col in df.columns:
            df[col] = (
                df[col].astype(str).str.strip()
                .str.replace(r"\s+", " ", regex=True)
            )

    if "bundle_id" in df.columns:
        before_rows = len(df)
        df = df[df["bundle_id"].notna() & ~df["bundle_id"].isin(["", "nan", "None"])]
        dropped = before_rows - len(df)
        if dropped:
            log.info("  Dropped %d rows with invalid bundle_id", dropped)

    return df.reset_index(drop=True)


def add_default_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Stamp every row with the constant columns the rest of the pipeline
    doesn't otherwise produce. Overwrites the columns if they already
    exist, so re-running this is always safe.
    """
    df = df.copy()
    df["country_code"] = "IND"
    df["os_type"] = "ANDROID"
    return df