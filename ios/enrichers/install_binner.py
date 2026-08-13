"""
ios/enrichers/install_binner.py
=================================
Install popularity bins. Apple's iTunes Lookup API does not expose install
counts, so every iOS row falls into installs_missing.
"""

import logging
import pandas as pd

log = logging.getLogger(__name__)

INSTALL_COLS: list[str] = [
    "installs_10m_plus", "installs_1m_10m", "installs_100k_1m",
    "installs_below100k", "installs_missing",
]


def enrich(df: pd.DataFrame) -> pd.DataFrame:
    """
    Input  : DataFrame with [bundle_id]
    Output : DataFrame with [bundle_id] + 5 binary install-bin columns
             (always installs_missing=1 — Apple does not provide install counts)
    """
    log.info("install_binner: processing %d rows ...", len(df))

    result = pd.DataFrame(index=df.index)
    result["bundle_id"] = df["bundle_id"].values
    for col in ("installs_10m_plus", "installs_1m_10m", "installs_100k_1m", "installs_below100k"):
        result[col] = 0
    result["installs_missing"] = 1

    return result.reset_index(drop=True)
