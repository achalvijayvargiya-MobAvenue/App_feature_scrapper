"""
enrichers/pricing_flagger.py
=============================
Pricing flags from catgory.sql:

    free = TRUE     → is_free_app
    free = FALSE    → is_paid_app
    offers_iap = TRUE   → offers_iap_flag
    offers_iap = FALSE  → no_iap_flag

Raw CSV values for boolean columns may be: True/False, true/false, 1/0, or
the string literals "True"/"False".  We normalise all of them.
"""

import logging
import pandas as pd

log = logging.getLogger(__name__)

PRICING_COLS: list[str] = [
    "is_free_app", "is_paid_app", "offers_iap_flag", "no_iap_flag",
]

_TRUE_VALS  = {"true", "1", "yes"}
_FALSE_VALS = {"false", "0", "no"}


def _to_bool_series(series: pd.Series) -> pd.Series:
    """
    Returns a nullable boolean Series:
        True  → truthy strings/values
        False → falsy strings/values
        NaN   → unrecognised / missing
    """
    normalised = series.astype(str).str.strip().str.lower()
    result = pd.Series(pd.NA, index=series.index, dtype="boolean")
    result[normalised.isin(_TRUE_VALS)]  = True
    result[normalised.isin(_FALSE_VALS)] = False
    return result


def enrich(df: pd.DataFrame) -> pd.DataFrame:
    """
    Input  : DataFrame with [bundle_id, free, offers_iap / offers_IAP]
    Output : DataFrame with [bundle_id] + 4 binary pricing columns
    """
    log.info("pricing_flagger: processing %d rows …", len(df))

    # offers_iap may appear with different casing in raw data
    iap_col = next(
        (c for c in df.columns if c.lower() == "offers_iap"),
        None,
    )

    free = _to_bool_series(df["free"])
    iap  = _to_bool_series(df[iap_col]) if iap_col else pd.Series(pd.NA, index=df.index, dtype="boolean")

    result = pd.DataFrame(index=df.index)
    result["bundle_id"]       = df["bundle_id"].values
    result["is_free_app"]     = (free == True).fillna(False).astype(int)   # noqa: E712
    result["is_paid_app"]     = (free == False).fillna(False).astype(int)  # noqa: E712
    result["offers_iap_flag"] = (iap  == True).fillna(False).astype(int)   # noqa: E712
    result["no_iap_flag"]     = (iap  == False).fillna(False).astype(int)  # noqa: E712

    return result.reset_index(drop=True)
