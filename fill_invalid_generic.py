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
    installs         → MEDIAN for Android rows; always 0 for iOS rows
                        (os_type == "IOS"), since Apple never exposes
                        install counts.


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
def get_static_defaults(is_ios: bool) -> dict[str, str]:
    return {
        "genreid": "UNKNOWN",
        "content_rating": "4+" if is_ios else "Rated for 3+",
        "developerid": "UNKNOWN",
        "developer": "UNKNOWN",
        "free": "false",
        "offers_iap": "false",
    }

STATIC_DEFAULT_KEYS = ["genreid", "content_rating", "developerid", "developer", "free", "offers_iap"]

# ---------------------------------------------------------------------------
# Columns filled with a statistic (MODE / MEDIAN) queried from Athena.
# installs is handled separately (see IOS_INSTALLS_DEFAULT) since iOS rows
# always default to 0 rather than the Android MEDIAN.
# ---------------------------------------------------------------------------
STAT_DEFAULT_COLS = ["score", "ratings_count"]

IOS_INSTALLS_DEFAULT = 0

# Columns considered when deciding whether a row gets default=True
DEFAULTABLE_COLS = STATIC_DEFAULT_KEYS + STAT_DEFAULT_COLS + ["installs"]

# Derived columns, filled in dependency order (each depends on the previous)
DERIVED_COLS = ["launch_date", "days_since_released", "months_since_launch"]

FILLABLE_COLS = DEFAULTABLE_COLS + DERIVED_COLS

LAUNCH_DATE_FORMAT = "%b %d, %Y"

_STRIP = re.compile(r"[,+\s]")
_SUFFIX_MULTIPLIER = {"K": 1_000, "M": 1_000_000, "B": 1_000_000_000}
_SUFFIX_RE = re.compile(r"^([\d.]+)([KMB])$", re.IGNORECASE)

ATHENA_S3_STAGING_DIR = "s3://mobavenue-simplismart-aws-s3-apse-sg/rtb/data/shabbir/athena_results/"

ATHENA_STATS_QUERY = """
WITH installs_median AS (
    SELECT UPPER(os_type) as os, UPPER(country_code) as cc, APPROX_PERCENTILE(installs, 0.5) as installs_median
    FROM prod.app_features
    GROUP BY 1, 2
),
score_counts AS (
    SELECT UPPER(os_type) as os, UPPER(country_code) as cc, score, COUNT(*) as cnt
    FROM prod.app_features
    WHERE score IS NOT NULL
    GROUP BY 1, 2, 3
),
score_mode AS (
    SELECT os, cc, score as score_mode
    FROM (SELECT os, cc, score, ROW_NUMBER() OVER(PARTITION BY os, cc ORDER BY cnt DESC) as rn FROM score_counts)
    WHERE rn = 1
),
ratings_counts AS (
    SELECT UPPER(os_type) as os, UPPER(country_code) as cc, ratings_count, COUNT(*) as cnt
    FROM prod.app_features
    WHERE ratings_count IS NOT NULL
    GROUP BY 1, 2, 3
),
ratings_mode AS (
    SELECT os, cc, ratings_count as ratings_count_mode
    FROM (SELECT os, cc, ratings_count, ROW_NUMBER() OVER(PARTITION BY os, cc ORDER BY cnt DESC) as rn FROM ratings_counts)
    WHERE rn = 1
)
SELECT 
    COALESCE(i.os, s.os, r.os) as os,
    COALESCE(i.cc, s.cc, r.cc) as cc,
    i.installs_median,
    s.score_mode,
    r.ratings_count_mode
FROM installs_median i
FULL OUTER JOIN score_mode s ON i.os = s.os AND i.cc = s.cc
FULL OUTER JOIN ratings_mode r ON COALESCE(i.os, s.os) = r.os AND COALESCE(i.cc, s.cc) = r.cc
""".strip()


def _fetch_athena_stats() -> dict[tuple[str, str], dict[str, float]]:
    """
    Query prod.app_features via Athena for the MODE of score/ratings_count
    and the MEDIAN of installs, grouped by os_type and country_code.

    Credentials are resolved via the standard AWS credential chain.
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
    
    stats_dict = {}
    for _, row in df.iterrows():
        os_val = str(row["os"]).strip().upper()
        cc_val = str(row["cc"]).strip().upper()
        stats_dict[(os_val, cc_val)] = {
            "score": row["score_mode"],
            "ratings_count": row["ratings_count_mode"],
            "installs": row["installs_median"],
        }
        log.info(
            "Athena stats [%s/%s] — score=%s  ratings=%s  installs=%s",
            os_val, cc_val, row["score_mode"], row["ratings_count_mode"], row["installs_median"]
        )

    return stats_dict


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


def _is_ios(record: dict) -> bool:
    return str(record.get("os_type", "")).strip().upper() == "IOS"


def run(input_path: Path, output_path: Path) -> None:
    log.info("Loading %s …", input_path.resolve())
    df = pd.read_csv(input_path, low_memory=False)
    df.columns = [c.lower() for c in df.columns]
    fill(df)
    df.to_csv(output_path, index=False)
    log.info("Output → %s  (%d rows)", output_path.resolve(), len(df))



def fill(df: pd.DataFrame, stats: dict[tuple[str, str], dict[str, float]] | None = None) -> pd.DataFrame:
    """
    Fill missing required fields in-place with default values and add a
    row-level ``default`` boolean column. Returns the same DataFrame.

    ``stats`` (score/ratings_count/installs MODE-MEDIAN from Athena) is
    fetched once via _fetch_athena_stats() if not supplied. Callers that
    invoke fill() multiple times in one run (e.g. once per batch, or across
    single-app calls in a loop) should fetch stats once up front and pass
    the same dict in each time, rather than letting each call re-query.

    """
    for col in FILLABLE_COLS:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].astype(object)


    if stats is None:
        stats = _fetch_athena_stats()

    total = len(df)
    defaulted_mask = pd.Series(False, index=df.index)
    filled_count = 0
    
    # Fallback to the first available stats if a specific (OS, CC) combination is completely missing in DB
    fallback_stats = next(iter(stats.values())) if stats else {"score": 0.0, "ratings_count": 0.0, "installs": 0.0}

    for idx in df.index:
        record = df.loc[idx].to_dict()
        row_defaulted = False
        
        is_ios = _is_ios(record)
        os_val = str(record.get("os_type", "ANDROID")).strip().upper()
        cc_val = str(record.get("country_code", "USA")).strip().upper()
        
        row_stats = stats.get((os_val, cc_val), fallback_stats)
        static_defaults = get_static_defaults(is_ios)

        for col, default_val in static_defaults.items():
            if _is_empty(record.get(col)):
                df.at[idx, col] = default_val
                row_defaulted = True
                filled_count += 1

        for col in STAT_DEFAULT_COLS:
            if _is_empty(record.get(col)):
                df.at[idx, col] = str(row_stats.get(col, 0.0))
                row_defaulted = True
                filled_count += 1


        if _is_empty(record.get("installs")):
            df.at[idx, "installs"] = (
                str(IOS_INSTALLS_DEFAULT) if is_ios else str(row_stats.get("installs", 0.0))
            )
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
                    "(static defaults + Athena MODE/MEDIAN stats; iOS installs default to 0). "
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
