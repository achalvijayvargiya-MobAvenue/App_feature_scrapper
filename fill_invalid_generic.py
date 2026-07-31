"""
fill_invalid_generic.py
======================
Fills missing required fields with default values so that otherwise-good
rows aren't dropped just because Google Play didn't return everything.

Static defaults:
    genreid          → "UNKNOWN"
    content_rating   → "Rated for 3+"
    developerid      → "UNKNOWN"
    developer        → "UNKNOWN"
    free             → "false"
    offers_iap       → "false"

Statistical defaults (queried live from Athena prod.app_features):
    score            → MODE
    ratings_count    → MODE
    installs         → MEDIAN

Derived defaults (dependency chain — each step uses the previous one's
already-resolved value, so they stay mutually consistent):
    launch_date            → today − days estimated from installs/ratings
                              (format "Jun 14, 2026", same as scraped data)
    days_since_released    → today − launch_date
    months_since_launch    → days_since_released / 30.44

Every row that had at least one of the above fields defaulted is flagged
with a new boolean column: ``default`` (True/False).

Usage:
    python fill_invalid_generic.py
    python fill_invalid_generic.py --input invalid_records.csv --output generic_filled_records.csv
"""

import argparse
import logging
import math
import os
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Columns filled with a fixed static default
# ---------------------------------------------------------------------------
STATIC_DEFAULTS: dict[str, str] = {
    "genreid": "UNKNOWN",
    "content_rating": "Rated for 3+",
    "developerid": "UNKNOWN",
    "developer": "UNKNOWN",
    "free": "false",
    "offers_iap": "false",
}

# ---------------------------------------------------------------------------
# Columns filled with a statistic (MODE / MEDIAN) queried from Athena
# ---------------------------------------------------------------------------
STAT_DEFAULT_COLS = ["score", "ratings_count", "installs"]

# Columns considered when deciding whether a row gets default=True
DEFAULTABLE_COLS = list(STATIC_DEFAULTS) + STAT_DEFAULT_COLS

# Derived columns, filled in dependency order (each depends on the previous)
DERIVED_COLS = ["launch_date", "days_since_released", "months_since_launch"]

FILLABLE_COLS = DEFAULTABLE_COLS + DERIVED_COLS

LAUNCH_DATE_FORMAT = "%b %d, %Y"

_STRIP = re.compile(r"[,+\s]")
_SUFFIX_MULTIPLIER = {"K": 1_000, "M": 1_000_000, "B": 1_000_000_000}
_SUFFIX_RE = re.compile(r"^([\d.]+)([KMB])$", re.IGNORECASE)

ATHENA_S3_STAGING_DIR = "s3://mobavenue-simplismart-aws-s3-apse-sg/rtb/data/shabbir/athena_results/"

ATHENA_STATS_QUERY = """
SELECT
    APPROX_PERCENTILE(installs, 0.5)                                   AS installs_median,
    ARBITRARY(score_mode.score)                                        AS score_mode,
    ARBITRARY(ratings_mode.ratings_count)                              AS ratings_count_mode
FROM prod.app_features
CROSS JOIN (
    SELECT score
    FROM prod.app_features
    WHERE score IS NOT NULL
    GROUP BY score
    ORDER BY COUNT(*) DESC
    LIMIT 1
) AS score_mode
CROSS JOIN (
    SELECT ratings_count
    FROM prod.app_features
    WHERE ratings_count IS NOT NULL
    GROUP BY ratings_count
    ORDER BY COUNT(*) DESC
    LIMIT 1
) AS ratings_mode
""".strip()


def _fetch_athena_stats() -> dict[str, float]:
    """
    Query prod.app_features via Athena for the MODE of score/ratings_count
    and the MEDIAN of installs.

    Credentials are resolved via the standard AWS credential chain (same as
    the AWS CLI): environment variables, ~/.aws/credentials, SSO, or an IAM
    role — whichever is active. No explicit access key/secret is passed.
    """
    from pyathena import connect
    from pyathena.pandas.cursor import PandasCursor

    conn = connect(
        region_name=os.environ.get("AWS_REGION"),
        s3_staging_dir=ATHENA_S3_STAGING_DIR,
        schema_name="prod",
        cursor_class=PandasCursor,
    )
    cursor = conn.cursor()
    df = cursor.execute(ATHENA_STATS_QUERY).as_pandas()
    row = df.iloc[0]

    log.info(
        "Athena stats — score_mode=%s  ratings_count_mode=%s  installs_median=%s",
        row["score_mode"], row["ratings_count_mode"], row["installs_median"],
    )
    return {
        "score": row["score_mode"],
        "ratings_count": row["ratings_count_mode"],
        "installs": row["installs_median"],
    }


def _parse_install_value(s: str) -> float | None:
    """
    Parse a single installs string. Handles:
        "1,000,000+"    (Western grouping)
        "1,00,00,000+"  (Indian grouping)
        "10M+", "500K+", "1.2B+"  (letter-suffix shorthand)
    """
    cleaned = re.sub(_STRIP, "", s)
    m = _SUFFIX_RE.match(cleaned)
    if m:
        num_part, suffix = m.groups()
        num = pd.to_numeric(num_part, errors="coerce")
        if pd.isna(num):
            return None
        return float(num) * _SUFFIX_MULTIPLIER[suffix.upper()]

    num = pd.to_numeric(cleaned, errors="coerce")
    return float(num) if pd.notna(num) else None


def _parse_installs(record: dict) -> float | None:
    """Parse installs from record (installs, real_installs, or min_installs)."""
    for col in ("installs", "real_installs", "min_installs"):
        val = record.get(col)
        if val is None or (isinstance(val, float) and pd.isna(val)):
            continue
        s = str(val).strip()
        if not s:
            continue
        num = _parse_install_value(s)
        if num is not None and num > 0:
            return num
    return None


def _estimate_days_from_installs(record: dict) -> int:
    """More ratings implies an older app (log-scaled, clamped 90–1825 days)."""
    installs = _parse_installs(record)
    ratings_count = max(1, int((installs or 100_000) * 0.01))
    days = int(90 + 180 * math.log10(max(1, ratings_count)))
    return min(1825, max(90, days))


def _derive_launch_date(record: dict) -> str:
    """launch_date = today − days estimated from (already-resolved) installs/ratings."""
    days = _estimate_days_from_installs(record)
    launch = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=days)
    return launch.strftime(LAUNCH_DATE_FORMAT)


def _derive_days_since_released(record: dict) -> str:
    """days_since_released = today − launch_date (kept consistent with launch_date)."""
    launch_date = record.get("launch_date")
    if not _is_empty(launch_date):
        try:
            delta = (
                datetime.now(timezone.utc).replace(tzinfo=None)
                - datetime.strptime(str(launch_date), LAUNCH_DATE_FORMAT)
            )
            return str(max(0, delta.days))
        except ValueError:
            pass
    return str(_estimate_days_from_installs(record))


def _derive_months_since_launch(record: dict) -> str:
    """months_since_launch = days_since_released / 30.44 (same formula as orchestrate.py)."""
    days = pd.to_numeric(record.get("days_since_released"), errors="coerce")
    if pd.isna(days):
        return ""
    return str(int(days / 30.44))


def _is_empty(val) -> bool:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return True
    return str(val).strip() == ""


def run(input_path: Path, output_path: Path) -> None:
    log.info("Loading %s …", input_path.resolve())
    df = pd.read_csv(input_path, low_memory=False)
    df.columns = [c.lower() for c in df.columns]
    fill(df)
    df.to_csv(output_path, index=False)
    log.info("Output → %s  (%d rows)", output_path.resolve(), len(df))


def fill(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing required fields in-place with default values and add a
    row-level ``default`` boolean column. Returns the same DataFrame.
    """
    for col in FILLABLE_COLS:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].astype(object)

    stats = _fetch_athena_stats()

    total = len(df)
    defaulted_mask = pd.Series(False, index=df.index)
    filled_count = 0

    for idx in df.index:
        record = df.loc[idx].to_dict()
        row_defaulted = False

        for col, default_val in STATIC_DEFAULTS.items():
            if _is_empty(record.get(col)):
                df.at[idx, col] = default_val
                row_defaulted = True
                filled_count += 1

        for col in STAT_DEFAULT_COLS:
            if _is_empty(record.get(col)):
                df.at[idx, col] = str(stats[col])
                row_defaulted = True
                filled_count += 1

        # Dependency chain: installs/ratings (just resolved above) → launch_date
        # → days_since_released → months_since_launch. Re-read the row so each
        # step sees the values the previous step just wrote.
        record = df.loc[idx].to_dict()

        if _is_empty(record.get("launch_date")):
            df.at[idx, "launch_date"] = _derive_launch_date(record)
            row_defaulted = True
            filled_count += 1
            record = df.loc[idx].to_dict()

        if _is_empty(record.get("days_since_released")):
            df.at[idx, "days_since_released"] = _derive_days_since_released(record)
            filled_count += 1
            record = df.loc[idx].to_dict()

        if _is_empty(record.get("months_since_launch")):
            df.at[idx, "months_since_launch"] = _derive_months_since_launch(record)
            filled_count += 1

        defaulted_mask.at[idx] = row_defaulted

    df["default"] = defaulted_mask

    log.info("Input records  : %d", total)
    log.info("Values filled  : %d", filled_count)
    log.info("Rows defaulted : %d  (%.1f%%)", int(defaulted_mask.sum()), defaulted_mask.mean() * 100 if total else 0)
    return df


def _parse_args():
    p = argparse.ArgumentParser(
        description="Fill missing required fields with default values "
                    "(static defaults + Athena MODE/MEDIAN stats). "
                    "Output has all input rows plus a 'default' flag column."
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
