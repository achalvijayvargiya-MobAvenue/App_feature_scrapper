"""
ios/enrichers/content_rating_flagger.py
=========================================
Content-rating flags mapped from Apple's age-rating strings
(e.g. "4+", "9+", "12+", "17+") onto the same flag schema Android uses.

    missing                 -> rating_missing
    "4+"                     -> rating_rated3plus   (mapped to closest Android bucket)
    "9+"                     -> rating_rated7plus   (approximate match to Android)
    "12+"                    -> rating_rated12plus
    "17+"                    -> rating_rated16plus + rating_mature
    "18+"                    -> rating_rated18plus + rating_adults18plus
    anything else            -> rating_other
"""

import logging
import pandas as pd

log = logging.getLogger(__name__)

RATING_COLS: list[str] = [
    "rating_everyone", "rating_teen", "rating_mature", "rating_everyone10plus",
    "rating_rated3plus", "rating_rated7plus", "rating_rated12plus",
    "rating_rated16plus", "rating_rated18plus", "rating_adults18plus",
    "rating_missing", "rating_other",
]


def enrich(df: pd.DataFrame) -> pd.DataFrame:
    """
    Input  : DataFrame with [bundle_id, content_rating]
    Output : DataFrame with [bundle_id] + 12 binary rating-flag columns
    """
    log.info("content_rating_flagger: processing %d rows ...", len(df))

    result = pd.DataFrame(index=df.index)
    result["bundle_id"] = df["bundle_id"].values
    for col in RATING_COLS:
        result[col] = 0

    normed = df["content_rating"].astype(str).str.strip().str.lower()

    for idx in df.index:
        val = normed.at[idx]
        if pd.isna(df["content_rating"].at[idx]) or val in ("nan", "none", ""):
            result.at[idx, "rating_missing"] = 1
        elif "4+" in val:
            result.at[idx, "rating_rated3plus"] = 1
        elif "9+" in val:
            result.at[idx, "rating_rated7plus"] = 1
        elif "12+" in val:
            result.at[idx, "rating_rated12plus"] = 1
        elif "17+" in val:
            result.at[idx, "rating_rated16plus"] = 1
            result.at[idx, "rating_mature"] = 1
        elif "18+" in val:
            result.at[idx, "rating_rated18plus"] = 1
            result.at[idx, "rating_adults18plus"] = 1
        else:
            result.at[idx, "rating_other"] = 1

    return result.reset_index(drop=True)
