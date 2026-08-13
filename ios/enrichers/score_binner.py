"""
ios/enrichers/score_binner.py
==============================
Score bins (same thresholds as Android):

    score >= 4.5                  -> score_45plus
    score >= 4.0 and score < 4.5  -> score_40_45
    score >= 3.0 and score < 4.0  -> score_30_40
    score < 3.0                   -> score_below30
    score IS NULL                 -> score_missing
"""

import logging
import pandas as pd

log = logging.getLogger(__name__)

SCORE_COLS: list[str] = [
    "score_45plus", "score_40_45", "score_30_40", "score_below30", "score_missing",
]


def enrich(df: pd.DataFrame) -> pd.DataFrame:
    """
    Input  : DataFrame with columns [bundle_id, score]
    Output : DataFrame with [bundle_id] + 5 binary score-bin columns
    """
    log.info("score_binner: processing %d rows ...", len(df))

    s = pd.to_numeric(df["score"], errors="coerce")
    missing = s.isna()

    result = pd.DataFrame(index=df.index)
    result["bundle_id"]     = df["bundle_id"].values
    result["score_45plus"]  = ((~missing) & (s >= 4.5)).astype(int)
    result["score_40_45"]   = ((~missing) & (s >= 4.0) & (s < 4.5)).astype(int)
    result["score_30_40"]   = ((~missing) & (s >= 3.0) & (s < 4.0)).astype(int)
    result["score_below30"] = ((~missing) & (s < 3.0)).astype(int)
    result["score_missing"] = missing.astype(int)

    return result.reset_index(drop=True)
