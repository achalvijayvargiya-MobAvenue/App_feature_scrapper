"""
fill_invalid_generic.py
======================
Fills missing score, ratings_count, and days_since_released with derived
values. Output contains all input rows; only missing values are updated.

Usage:
    python fill_invalid_generic.py
    python fill_invalid_generic.py --input invalid_records.csv --output generic_filled_records.csv
"""

import argparse
import logging
import math
import random
import re
import sys
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

FILLABLE_COLS = ["score", "ratings_count", "days_since_released"]
_STRIP = re.compile(r"[,+\s]")


def _parse_installs(record: dict) -> float | None:
    """Parse installs from record (installs, real_installs, or min_installs)."""
    for col in ("installs", "real_installs", "min_installs"):
        val = record.get(col)
        if val is None or (isinstance(val, float) and pd.isna(val)):
            continue
        s = str(val).strip()
        if not s:
            continue
        num = pd.to_numeric(re.sub(_STRIP, "", s), errors="coerce")
        if pd.notna(num) and num > 0:
            return float(num)
    return None


def _derive_fill_values(record: dict) -> dict[str, str]:
    """
    Derive score, ratings_count, days_since_released from installs.
    - ratings_count: 1% of installs (fallback 1000)
    - score: random 3.0–4.3
    - days_since_released: more ratings = older app
    """
    installs = _parse_installs(record)
    ratings_count = max(1, int((installs or 100_000) * 0.01))

    score = round(random.uniform(3.0, 4.3), 1)

    days = int(90 + 180 * math.log10(max(1, ratings_count)))
    days_since_released = min(1825, max(90, days))

    return {
        "score": str(score),
        "ratings_count": str(ratings_count),
        "days_since_released": str(days_since_released),
    }


def _is_empty(val) -> bool:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return True
    return str(val).strip() == ""


def run(input_path: Path, output_path: Path) -> None:
    log.info("Loading %s …", input_path.resolve())
    df = pd.read_csv(input_path, low_memory=False)
    df.columns = [c.lower() for c in df.columns]

    for col in FILLABLE_COLS:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].astype(object)

    total = len(df)
    filled_count = 0

    for idx in df.index:
        record = df.loc[idx].to_dict()
        derived = _derive_fill_values(record)
        for col in FILLABLE_COLS:
            if _is_empty(record.get(col)):
                df.at[idx, col] = derived[col]
                filled_count += 1

    df.to_csv(output_path, index=False)

    log.info("Input records  : %d", total)
    log.info("Values filled  : %d", filled_count)
    log.info("Output → %s  (%d rows)", output_path.resolve(), len(df))


def _parse_args():
    p = argparse.ArgumentParser(
        description="Fill missing score, ratings_count, days_since_released with derived values. Output has all input rows."
    )
    p.add_argument(
        "--input", type=Path, default=Path("invalid_records.csv"),
        help="Input CSV (default: invalid_records.csv)",
    )
    p.add_argument(
        "--output", type=Path, default=Path("generic_filled_records.csv"),
        help="Output CSV (default: generic_filled_records.csv)",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    if not args.input.exists():
        log.error("Input file not found: %s", args.input.resolve())
        sys.exit(1)

    run(args.input, args.output)
