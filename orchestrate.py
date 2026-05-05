"""
orchestrate.py
==============
Validates and enriches a pre-scraped Play Store dataset, producing two output
files:

  1. <output>          — valid records with all enriched feature columns.
  2. <invalid-output>  — records that failed required-field validation.
                         Pass this file to llm_fallback.py to recover those rows.

Assumes the input CSV was already scraped (e.g. by a teammate).
If required fields are missing, this script can scrape Google Play bundle-wise
to fill those fields before validation. This script does NOT call the LLM.

Target schema columns (test.txt):
    bundle_id, description, summary, genreid, content_rating, score,
    ratings_count, installs, developerid, developer, free, offers_iap,
    days_since_released, months_since_launch,
    is_investments … is_arcade_game  (category flags from category_mapper)
    score_45plus … score_missing  (5 score bins)
    installs_10m_plus … installs_missing  (5 install bins)
    apps_0_3months … app_age_missing  (5 age bins)
    is_free_app … no_iap_flag  (4 pricing flags)
    rating_everyone … rating_other  (12 rating flags)

Pipeline steps
--------------
  1. Load & normalise columns
  2. Resolve installs  (real_installs → installs)
  3. Filter invalid records  →  valid_df  +  invalid_df
  4. Extract base passthrough columns
  5. Run all enrichers  (category, score, installs, age, pricing, content-rating)
  6. Merge enrichments
  7. Select & order final columns
  8. Coverage report
  9. Save valid enriched CSV  +  invalid records CSV

Usage:
    python orchestrate.py --input scraped_data.csv
    python orchestrate.py --input scraped_data.csv --output valid_enriched.csv --invalid-output invalid_records.csv
"""

import argparse
import logging
import sys
from pathlib import Path
from functools import reduce
from datetime import datetime, timezone

import pandas as pd
from google_play_scraper import app as gps_app
from google_play_scraper.exceptions import NotFoundError

# ── Enrichers ────────────────────────────────────────────────────────────────
from enrichers import (
    category_mapper,
    score_binner,
    install_binner,
    app_age_binner,
    pricing_flagger,
    content_rating_flagger,
)
from enrichers.category_mapper import CATEGORY_COLS

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_INPUT   = Path("scraped_data.csv")
DEFAULT_OUTPUT  = Path("valid_enriched_records.csv")
DEFAULT_INVALID = Path("invalid_records.csv")

# ---------------------------------------------------------------------------
# Columns that must be non-null and non-empty to keep a record
# ---------------------------------------------------------------------------
REQUIRED_COLS: list[str] = [
    "bundle_id", "description", "summary", "genreid",
    "content_rating", "score", "ratings_count", "installs",
    "developerid", "developer", "free", "offers_iap",
    "days_since_released",
]

# ---------------------------------------------------------------------------
# Final column order  (matches test.txt schema exactly)
# ---------------------------------------------------------------------------
PASSTHROUGH_COLS: list[str] = [
    "bundle_id", "description", "summary", "genreid",
    "content_rating", "score", "ratings_count", "installs",
    "developerid", "developer", "free", "offers_iap",
    "days_since_released", "months_since_launch",
]

FINAL_COLS: list[str] = PASSTHROUGH_COLS + [
    # category flags (from category_mapper.CATEGORY_COLS)
    *CATEGORY_COLS,
    # score bins
    "score_45plus", "score_40_45", "score_30_40", "score_below30", "score_missing",
    # install bins
    "installs_10m_plus", "installs_1m_10m", "installs_100k_1m",
    "installs_below100k", "installs_missing",
    # age bins
    "apps_0_3months", "apps_3_12months", "apps_1_2years",
    "apps_2plus_years", "app_age_missing",
    # pricing flags
    "is_free_app", "is_paid_app", "offers_iap_flag", "no_iap_flag",
    # content rating flags
    "rating_everyone", "rating_teen", "rating_mature", "rating_everyone10plus",
    "rating_rated3plus", "rating_rated7plus", "rating_rated12plus",
    "rating_rated16plus", "rating_rated18plus", "rating_adults18plus",
    "rating_missing", "rating_other",
]

SEPARATOR = "=" * 70


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Lowercase all column names so lookups work regardless of source casing.
    Also normalises alternate bundle-id column names:
        pub_bundle        → bundle_id
        single-column CSV → bundle_id
    """
    df.columns = [c.lower() for c in df.columns]
    if "bundle_id" not in df.columns:
        if "pub_bundle" in df.columns:
            df = df.rename(columns={"pub_bundle": "bundle_id"})
            log.info("Renamed column 'pub_bundle' → 'bundle_id'.")
        elif len(df.columns) == 1:
            original = df.columns[0]
            df = df.rename(columns={original: "bundle_id"})
            log.info("Single-column input — renamed '%s' → 'bundle_id'.", original)
    return df


def _is_empty(val) -> bool:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return True
    return str(val).strip() == ""


def _released_to_days(released_str: str | None) -> int | None:
    if not released_str:
        return None
    for fmt in ("%b %d, %Y", "%Y-%m-%d", "%B %d, %Y"):
        try:
            delta = (
                datetime.now(timezone.utc).replace(tzinfo=None)
                - datetime.strptime(str(released_str), fmt)
            )
            return max(0, delta.days)
        except ValueError:
            continue
    return None


def _scrape_bundle(bundle_id: str) -> dict:
    data = gps_app(bundle_id, lang="en", country="in")
    return {
        "bundle_id": bundle_id,
        "description": data.get("description"),
        "summary": data.get("summary"),
        "genreid": data.get("genreId"),
        "content_rating": data.get("contentRating"),
        "score": data.get("score"),
        "ratings_count": data.get("ratings"),
        "installs": data.get("realInstalls") or data.get("installs"),
        "developerid": data.get("developerId"),
        "developer": data.get("developer"),
        "free": data.get("free"),
        "offers_iap": data.get("offersIAP"),
        "days_since_released": _released_to_days(data.get("released")),
        "real_installs": data.get("realInstalls"),
    }


def _hydrate_missing_required_fields(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing required fields by scraping Google Play per bundle ID.
    Only empty required values are overwritten; existing values are preserved.
    """
    df = df.copy()
    for col in REQUIRED_COLS:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].astype(object)

    needs_scrape = pd.Series(False, index=df.index)
    for col in REQUIRED_COLS:
        needs_scrape |= df[col].isna() | (df[col].astype(str).str.strip() == "")

    needs_scrape &= ~(df["bundle_id"].isna() | (df["bundle_id"].astype(str).str.strip() == ""))
    targets = df.index[needs_scrape]
    if len(targets) == 0:
        return df

    log.info("Hydrating missing required fields via Google Play scrape for %d rows …", len(targets))
    success_count = 0

    if "real_installs" not in df.columns:
        df["real_installs"] = None

    for idx in targets:
        bundle_id = str(df.at[idx, "bundle_id"]).strip()
        try:
            scraped = _scrape_bundle(bundle_id)
        except NotFoundError:
            log.warning("Bundle not found on Play Store: %s", bundle_id)
            continue
        except Exception as exc:
            log.warning("Scrape failed for %s: %s", bundle_id, exc)
            continue

        for col in REQUIRED_COLS:
            if _is_empty(df.at[idx, col]) and not _is_empty(scraped.get(col)):
                df.at[idx, col] = scraped[col]
        if _is_empty(df.at[idx, "real_installs"]) and not _is_empty(scraped.get("real_installs")):
            df.at[idx, "real_installs"] = scraped.get("real_installs")
        success_count += 1

    log.info("Scrape hydrate complete: %d/%d rows fetched.", success_count, len(targets))
    return df


def _resolve_installs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prefer ``real_installs`` over ``installs`` when available.
    Drops the ``real_installs`` column after merging.
    """
    if "real_installs" not in df.columns:
        return df

    real = pd.to_numeric(df["real_installs"], errors="coerce")
    has_real = real.notna() & (real > 0)

    if has_real.any():
        log.info(
            "install resolution — using real_installs for %d rows, "
            "installs (bucket) fallback for %d rows.",
            int(has_real.sum()),
            int((~has_real).sum()),
        )
        df = df.copy()
        df["installs"] = df["installs"].astype(object)  # allow str or int
        vals = real[has_real].astype("Int64")
        df.loc[has_real, "installs"] = vals.astype(str)

    df = df.drop(columns=["real_installs"])
    return df


def _derive_months(df: pd.DataFrame) -> pd.Series:
    """CAST(days_since_released / 30.44 AS INTEGER)."""
    if "months_since_launch" in df.columns:
        m = pd.to_numeric(df["months_since_launch"], errors="coerce")
        if m.notna().any():
            return m
    days = pd.to_numeric(
        df.get("days_since_released", pd.Series(dtype=float)), errors="coerce"
    )
    return (days / 30.44).apply(lambda x: int(x) if pd.notna(x) else None)


def _select_passthrough(df: pd.DataFrame) -> pd.DataFrame:
    """Extract only the base columns we need to pass through to the output."""
    base = pd.DataFrame()
    base["bundle_id"]           = df.get("bundle_id", "")
    base["description"]         = df.get("description", "")
    base["summary"]             = df.get("summary", "")
    base["genreid"]             = df.get("genreid", "")
    base["content_rating"]      = df.get("content_rating", "")
    base["score"]               = df.get("score", "")
    base["ratings_count"]       = df.get("ratings_count", "")
    base["installs"]            = df.get("installs", "")
    base["developerid"]         = df.get("developerid", "")
    base["developer"]           = df.get("developer", "")
    base["free"]                = df.get("free", "")
    base["offers_iap"]          = df.get("offers_iap", "")
    base["days_since_released"] = df.get("days_since_released", "")
    base["months_since_launch"] = _derive_months(df)
    return base.reset_index(drop=True)


def _merge_enrichments(parts: list[pd.DataFrame]) -> pd.DataFrame:
    """Left-join all enrichment DataFrames on bundle_id."""
    return reduce(
        lambda left, right: left.merge(right, on="bundle_id", how="left"),
        parts,
    )


def _filter_invalid(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split rows into (valid_df, invalid_df) based on REQUIRED_COLS.
    A row is invalid if any required column is null or empty string.
    """
    cols_present = [c for c in REQUIRED_COLS if c in df.columns]
    missing_def  = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing_def:
        log.warning(
            "Required columns not in source (every row treated as invalid): %s",
            missing_def,
        )

    mask_valid = pd.Series(True, index=df.index)

    if missing_def:
        mask_valid &= False
    else:
        for col in cols_present:
            not_null  = df[col].notna()
            not_empty = df[col].astype(str).str.strip() != ""
            mask_valid &= (not_null & not_empty)

    valid   = df[mask_valid].reset_index(drop=True)
    dropped = df[~mask_valid].reset_index(drop=True)
    return valid, dropped


def _validation_report(total: int, dropped: pd.DataFrame) -> None:
    """Log a per-column breakdown of why records were dropped."""
    log.info(SEPARATOR)
    log.info("VALIDATION REPORT")
    log.info(SEPARATOR)
    log.info("  Total loaded    : %d", total)
    log.info(
        "  Invalid (dropped): %d  (%.1f%%)",
        len(dropped), len(dropped) / total * 100 if total else 0,
    )
    log.info(
        "  Valid (kept)     : %d  (%.1f%%)",
        total - len(dropped),
        (total - len(dropped)) / total * 100 if total else 0,
    )
    if len(dropped):
        log.info("  -- Null/empty counts per required column (in dropped rows) --")
        for col in REQUIRED_COLS:
            if col in dropped.columns:
                n = (
                    dropped[col].isna()
                    | (dropped[col].astype(str).str.strip() == "")
                ).sum()
                if n:
                    log.info("    %-22s  %d rows", col, n)
    log.info(SEPARATOR)


def coverage_report(df: pd.DataFrame, enriched_cols: list[str]) -> None:
    log.info(SEPARATOR)
    log.info("COVERAGE REPORT")
    log.info(SEPARATOR)
    total = len(df)
    for col in enriched_cols:
        if col in df.columns:
            n   = df[col].sum()
            pct = n / total * 100 if total else 0
            log.info("  %-28s  %7.0f  (%5.1f%%)", col, n, pct)
    log.info(SEPARATOR)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run(
    input_path: Path,
    output_path: Path,
    *,
    invalid_output: Path,
) -> None:
    log.info(SEPARATOR)
    log.info("ORCHESTRATE — Validate & Enrich Pipeline")
    log.info(SEPARATOR)
    log.info("Input          : %s", input_path.resolve())
    log.info("Valid output   : %s", output_path.resolve())
    log.info("Invalid output : %s", invalid_output.resolve())

    # ── 1. Load ───────────────────────────────────────────────────────────────
    log.info("Loading data …")
    raw = pd.read_csv(input_path, low_memory=False)
    raw = _normalise_columns(raw)
    raw = _hydrate_missing_required_fields(raw)
    raw = _resolve_installs(raw)
    log.info("Loaded %d rows, %d columns.", len(raw), len(raw.columns))

    # ── 2. Filter invalid records ─────────────────────────────────────────────
    log.info("Validating records …")
    valid, dropped = _filter_invalid(raw)
    _validation_report(len(valid) + len(dropped), dropped)

    # Save invalid records immediately so they are available for llm_fallback.py
    if len(dropped):
        dropped.to_csv(invalid_output, index=False)
        log.info(
            "Invalid records (%d) saved → %s",
            len(dropped), invalid_output.resolve(),
        )
        log.info("  → Run:  python llm_fallback.py --input %s", invalid_output)

    if len(valid) == 0:
        log.warning("No valid records to enrich. Exiting.")
        return

    # ── 3. Base passthrough columns ───────────────────────────────────────────
    log.info("Extracting base columns …")
    base = _select_passthrough(valid)

    # ── 4. Run enrichers ──────────────────────────────────────────────────────
    enrichment_parts: list[pd.DataFrame] = [
        category_mapper.enrich(valid),
        score_binner.enrich(valid),
        install_binner.enrich(valid),
        app_age_binner.enrich(valid),
        pricing_flagger.enrich(valid),
        content_rating_flagger.enrich(valid),
    ]

    # ── 5. Merge all enrichments onto base ────────────────────────────────────
    log.info("Merging enrichments …")
    enriched = _merge_enrichments([base] + enrichment_parts)

    # ── 6. Select & order final columns ───────────────────────────────────────
    missing_cols = [c for c in FINAL_COLS if c not in enriched.columns]
    if missing_cols:
        log.warning("Expected columns not found (will be blank): %s", missing_cols)
        for c in missing_cols:
            enriched[c] = None

    final = enriched[FINAL_COLS]

    # ── 7. Coverage report ────────────────────────────────────────────────────
    flag_cols = [c for c in FINAL_COLS if c not in PASSTHROUGH_COLS]
    coverage_report(final, flag_cols)

    # ── 8. Save valid enriched output ─────────────────────────────────────────
    final.to_csv(output_path, index=False)
    log.info(
        "Saved %d valid enriched rows × %d columns → %s",
        len(final), len(final.columns), output_path.resolve(),
    )
    log.info(SEPARATOR)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Validate and enrich pre-scraped Play Store data. "
            "Produces a valid enriched CSV and an invalid records CSV."
        )
    )
    p.add_argument(
        "--input", type=Path, default=DEFAULT_INPUT,
        help=f"Source CSV with scraped Play Store data (default: {DEFAULT_INPUT})",
    )
    p.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT,
        help=f"Output CSV for valid enriched records (default: {DEFAULT_OUTPUT})",
    )
    p.add_argument(
        "--invalid-output", type=Path, default=DEFAULT_INVALID,
        help=(
            f"Output CSV for records that fail validation (default: {DEFAULT_INVALID}). "
            "Pass this file to llm_fallback.py to attempt recovery."
        ),
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    if not args.input.exists():
        log.error("Input file not found: %s", args.input.resolve())
        sys.exit(1)

    run(
        args.input,
        args.output,
        invalid_output=args.invalid_output,
    )
