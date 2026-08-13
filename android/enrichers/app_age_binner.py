"""
enrichers/app_age_binner.py
============================
App-age bins from catgory.sql.

months_since_launch is derived as:
    CAST(days_since_released / 30.44 AS INTEGER)

Bins:
    months_since_launch <= 3            → apps_0_3months
    months_since_launch > 3  and <= 12  → apps_3_12months
    months_since_launch > 12 and <= 24  → apps_1_2years
    months_since_launch > 24            → apps_2plus_years
    months_since_launch IS NULL         → app_age_missing
"""

import logging
import pandas as pd

log = logging.getLogger(__name__)

AGE_COLS: list[str] = [
    "apps_0_3months", "apps_3_12months", "apps_1_2years",
    "apps_2plus_years", "app_age_missing",
]


def _derive_months(df: pd.DataFrame) -> pd.Series:
    """
    Use months_since_launch if already present, otherwise compute from
    days_since_released using the SQL formula CAST(days / 30.44 AS INT).
    """
    if "months_since_launch" in df.columns:
        months = pd.to_numeric(df["months_since_launch"], errors="coerce")
        if months.notna().any():
            return months

    if "days_since_released" in df.columns:
        days = pd.to_numeric(df["days_since_released"], errors="coerce")
        return (days / 30.44).apply(lambda x: int(x) if pd.notna(x) else float("nan"))

    return pd.Series([float("nan")] * len(df), index=df.index)


def enrich(df: pd.DataFrame) -> pd.DataFrame:
    """
    Input  : DataFrame with [bundle_id] and either
             [months_since_launch] or [days_since_released].
    Output : DataFrame with [bundle_id] + 5 binary age-bin columns.
    """
    log.info("app_age_binner: processing %d rows …", len(df))

    m = _derive_months(df)
    missing = m.isna()

    result = pd.DataFrame(index=df.index)
    result["bundle_id"]        = df["bundle_id"].values
    result["apps_0_3months"]   = ((~missing) & (m <= 3)).astype(int)
    result["apps_3_12months"]  = ((~missing) & (m > 3)  & (m <= 12)).astype(int)
    result["apps_1_2years"]    = ((~missing) & (m > 12) & (m <= 24)).astype(int)
    result["apps_2plus_years"] = ((~missing) & (m > 24)).astype(int)
    result["app_age_missing"]  = missing.astype(int)

    return result.reset_index(drop=True)
