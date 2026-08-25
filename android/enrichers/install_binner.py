"""
enrichers/install_binner.py
============================
Install popularity bins from catgory.sql:

    CASE WHEN installs >= 10_000_000             → installs_10m_plus
    CASE WHEN installs >= 1_000_000  < 10M       → installs_1m_10m
    CASE WHEN installs >= 100_000    < 1M        → installs_100k_1m
    CASE WHEN installs < 100_000                 → installs_below100k
    CASE WHEN installs IS NULL                   → installs_missing

The raw CSV `installs` column is a string like "1,000,000+".
We parse it by stripping commas and "+" then converting to float.
If parsing fails we fall back to the `min_installs` column when present.
"""

import logging
import re

import pandas as pd

log = logging.getLogger(__name__)

INSTALL_COLS: list[str] = [
    "installs_10m_plus", "installs_1m_10m", "installs_100k_1m",
    "installs_below100k", "installs_missing",
]

_STRIP = re.compile(r"[,+\s]")
_SUFFIX_MULTIPLIER = {"K": 1_000, "M": 1_000_000, "B": 1_000_000_000}
_SUFFIX_RE = re.compile(r"^([\d.]+)([KMB])$", re.IGNORECASE)


def _parse_installs(series: pd.Series, fallback: pd.Series | None) -> pd.Series:
    """
    Return a numeric Series of install counts. Handles:
        "1,000,000+"    (Western grouping)
        "1,00,00,000+"  (Indian grouping)
        "10M+", "500K+", "1.2B+"  (letter-suffix shorthand)
    """
    cleaned = series.astype(str).str.replace(_STRIP, "", regex=True)
    numeric = pd.to_numeric(cleaned, errors="coerce")

    suffix_match = cleaned.str.extract(_SUFFIX_RE)
    has_suffix = suffix_match[0].notna()
    if has_suffix.any():
        suffix_num = pd.to_numeric(suffix_match[0], errors="coerce")
        multiplier = suffix_match[1].str.upper().map(_SUFFIX_MULTIPLIER)
        numeric = numeric.where(~has_suffix, suffix_num * multiplier)

    if fallback is not None:
        # fill gaps from min_installs when string parse fails
        numeric = numeric.fillna(pd.to_numeric(fallback, errors="coerce"))
    return numeric


def enrich(df: pd.DataFrame) -> pd.DataFrame:
    """
    Input  : DataFrame with columns [bundle_id, installs]
             Optionally also [min_installs] for fallback.
    Output : DataFrame with [bundle_id] + 5 binary install-bin columns
    """
    log.info("install_binner: processing %d rows …", len(df))

    fallback = df["min_installs"] if "min_installs" in df.columns else None
    inst = _parse_installs(df["installs"], fallback)
    missing = inst.isna()

    result = pd.DataFrame(index=df.index)
    result["bundle_id"]          = df["bundle_id"].values
    result["installs_10m_plus"]  = ((~missing) & (inst >= 10_000_000)).astype(int)
    result["installs_1m_10m"]    = ((~missing) & (inst >= 1_000_000)  & (inst < 10_000_000)).astype(int)
    result["installs_100k_1m"]   = ((~missing) & (inst >= 100_000)    & (inst < 1_000_000)).astype(int)
    result["installs_below100k"] = ((~missing) & (inst < 100_000)).astype(int)
    result["installs_missing"]   = missing.astype(int)

    return result.reset_index(drop=True)
