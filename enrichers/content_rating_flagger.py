"""
enrichers/content_rating_flagger.py
=====================================
Content-rating flags from catgory.sql:

    LOWER(content_rating) = 'everyone'              → rating_everyone
    LOWER(content_rating) = 'teen'                  → rating_teen
    LOWER(content_rating) = 'mature 17+'            → rating_mature
    LOWER(content_rating) = 'everyone 10+'          → rating_everyone10plus
    IN ('rated for 3+',  'pegi 3')                  → rating_rated3plus
    IN ('rated for 7+',  'pegi 7')                  → rating_rated7plus
    IN ('rated for 12+', 'pegi 12')                 → rating_rated12plus
    IN ('rated for 16+', 'pegi 16')                 → rating_rated16plus
    IN ('rated for 18+', 'pegi 18')                 → rating_rated18plus
    LOWER(content_rating) = 'adults only 18+'       → rating_adults18plus
    content_rating IS NULL                          → rating_missing
    not null AND not in any known bucket            → rating_other
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

# (flag_column, set of matching lowercased values)
_RATING_RULES: list[tuple[str, set[str]]] = [
    ("rating_everyone",       {"everyone"}),
    ("rating_teen",           {"teen"}),
    ("rating_mature",         {"mature 17+"}),
    ("rating_everyone10plus", {"everyone 10+"}),
    ("rating_rated3plus",     {"rated for 3+", "pegi 3"}),
    ("rating_rated7plus",     {"rated for 7+", "pegi 7"}),
    ("rating_rated12plus",    {"rated for 12+", "pegi 12"}),
    ("rating_rated16plus",    {"rated for 16+", "pegi 16"}),
    ("rating_rated18plus",    {"rated for 18+", "pegi 18"}),
    ("rating_adults18plus",   {"adults only 18+"}),
]

_ALL_KNOWN: set[str] = {v for _, vals in _RATING_RULES for v in vals}


def enrich(df: pd.DataFrame) -> pd.DataFrame:
    """
    Input  : DataFrame with [bundle_id, content_rating]
    Output : DataFrame with [bundle_id] + 12 binary rating-flag columns
    """
    log.info("content_rating_flagger: processing %d rows …", len(df))

    raw     = df["content_rating"]
    missing = raw.isna() | (raw.astype(str).str.strip() == "")
    normed  = raw.astype(str).str.strip().str.lower()

    result = pd.DataFrame(index=df.index)
    result["bundle_id"] = df["bundle_id"].values

    for col, valid_vals in _RATING_RULES:
        result[col] = ((~missing) & normed.isin(valid_vals)).astype(int)

    result["rating_missing"] = missing.astype(int)
    result["rating_other"]   = (
        (~missing) & (~normed.isin(_ALL_KNOWN))
    ).astype(int)

    return result.reset_index(drop=True)
