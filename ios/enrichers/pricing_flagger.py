"""
ios/enrichers/pricing_flagger.py
==================================
Pricing flags (same schema as Android):

    free = TRUE       -> is_free_app
    free = FALSE      -> is_paid_app
    offers_iap = TRUE  -> offers_iap_flag
    offers_iap = FALSE -> no_iap_flag

Apple's iTunes Lookup API does not reliably expose in-app-purchase data, so
offers_iap is always None for iOS — no_iap_flag is set for missing values too.
"""

import logging
import pandas as pd

log = logging.getLogger(__name__)

PRICING_COLS: list[str] = [
    "is_free_app", "is_paid_app", "offers_iap_flag", "no_iap_flag",
]


def enrich(df: pd.DataFrame) -> pd.DataFrame:
    """
    Input  : DataFrame with [bundle_id, free, offers_iap]
    Output : DataFrame with [bundle_id] + 4 binary pricing columns
    """
    log.info("pricing_flagger: processing %d rows ...", len(df))

    is_free = df["free"].apply(lambda x: x is True or str(x).lower() == "true")
    is_paid = df["free"].apply(lambda x: x is False or str(x).lower() == "false")
    has_iap = df["offers_iap"].apply(lambda x: x is True or str(x).lower() == "true")
    no_iap = df["offers_iap"].apply(
        lambda x: x is False or str(x).lower() == "false" or pd.isna(x)
    )

    result = pd.DataFrame(index=df.index)
    result["bundle_id"]       = df["bundle_id"].values
    result["is_free_app"]     = is_free.astype(int)
    result["is_paid_app"]     = is_paid.astype(int)
    result["offers_iap_flag"] = has_iap.astype(int)
    result["no_iap_flag"]     = no_iap.astype(int)

    return result.reset_index(drop=True)
