"""
ios/pipeline.py
================
Entry points for executing the iOS scrape + enrich pipeline.

Pipeline steps (mirrors android/pipeline.py)
---------------------------------------------
  1. Load & normalise columns
  2. (installs resolution — no-op for iOS, Apple exposes no install counts)
  3. Fill missing required fields with defaults (fill_invalid_generic) so
     fewer rows are dropped for gaps that have a sane fallback
  4. Filter invalid records  ->  valid_df  +  invalid_df
  5. Extract base passthrough columns
  6. Run all enrichers  (category, score, installs, age, pricing, content-rating)
  7. Merge enrichments
  8. Select & order final columns
  9. Coverage report
  10. Save valid enriched CSV  +  invalid records CSV
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from functools import reduce

import pandas as pd
from tqdm import tqdm

import fill_invalid_generic
from ios.constants import FINAL_COLUMNS
from ios.scraper import scrape_ios_bundle
from ios.enrichers import (
    category_mapper,
    score_binner,
    install_binner,
    app_age_binner,
    pricing_flagger,
    content_rating_flagger,
)

log = logging.getLogger(__name__)
SEPARATOR = "=" * 70

# Apple's iTunes Lookup API allows ~20 requests/minute per IP before returning
# 429s (undocumented, "approximately", subject to change). 3.5s/request keeps
# us at ~17/min, safely under that ceiling.
_REQUEST_INTERVAL_SECONDS = 3.5

REQUIRED_COLS: list[str] = [
    "bundle_id", "app_name", "description", "summary", "genreid",
    "content_rating", "score", "ratings_count",
    "developerid", "developer", "free",
    "launch_date",
]


def _compute_days_since_released(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "days_since_released" not in df.columns:
        df["days_since_released"] = None
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    for idx, row in df.iterrows():
        launch = row.get("launch_date")
        if pd.notna(launch) and launch:
            for fmt in ("%b %d, %Y", "%Y-%m-%d"):
                try:
                    launch_dt = datetime.strptime(str(launch)[:10] if fmt == "%Y-%m-%d" else str(launch), fmt)
                    df.at[idx, "days_since_released"] = max(0, (now - launch_dt).days)
                    break
                except ValueError:
                    continue
    return df


def _filter_invalid(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    cols_present = [c for c in REQUIRED_COLS if c in df.columns]
    missing_def = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing_def:
        log.warning("Required columns not in source (every row treated as invalid): %s", missing_def)

    mask_valid = pd.Series(True, index=df.index)
    if missing_def:
        mask_valid &= False
    else:
        for col in cols_present:
            mask_valid &= df[col].notna() & (df[col].astype(str).str.strip() != "")

    return df[mask_valid].reset_index(drop=True), df[~mask_valid].reset_index(drop=True)


def _merge_enrichments(parts: list[pd.DataFrame]) -> pd.DataFrame:
    return reduce(lambda left, right: left.merge(right, on="bundle_id", how="left"), parts)


def enrich_ios_data(df: pd.DataFrame) -> pd.DataFrame:
    df = _compute_days_since_released(df)
    df["months_since_launch"] = pd.to_numeric(df["days_since_released"], errors="coerce") / 30.0

    enrichment_parts = [
        category_mapper.enrich(df),
        score_binner.enrich(df),
        install_binner.enrich(df),
        app_age_binner.enrich(df),
        pricing_flagger.enrich(df),
        content_rating_flagger.enrich(df),
    ]

    enriched = _merge_enrichments([df] + enrichment_parts)

    for col in FINAL_COLUMNS:
        if col not in enriched.columns:
            enriched[col] = ""

    return enriched[FINAL_COLUMNS]


def run_single_ios(app_id: str, output_path: str = "single_bundle_output_ios.csv",
                    athena_stats: dict | None = None):
    log.info("Scraping iOS App Store for %s...", app_id)
    try:
        data = scrape_ios_bundle(app_id)
        df = pd.DataFrame([data])
    except Exception as e:
        log.error("Failed to scrape %s: %s", app_id, e)
        return

    df = fill_invalid_generic.fill(df, stats=athena_stats)
    valid, dropped = _filter_invalid(df)
    if valid.empty:
        log.warning("App %s failed validation, no output written.", app_id)
        return

    log.info("Enriching iOS data...")
    df_enriched = enrich_ios_data(valid)

    log.info("Saving to %s...", output_path)
    df_enriched.to_csv(output_path, index=False)
    log.info("Done.")


def run_batch_ios(input_csv: str, output_path: str, id_col: str = "bundle_id",
                   invalid_output: str = "invalid_records_ios.csv",
                   athena_stats: dict | None = None):
    log.info(SEPARATOR)
    log.info("IOS PIPELINE - Scrape & Enrich")
    log.info(SEPARATOR)

    df_in = pd.read_csv(input_csv)
    if id_col not in df_in.columns:
        log.error("Input CSV must contain '%s' column.", id_col)
        return

    records = []
    fail_count = 0
    app_ids = df_in[id_col].dropna().unique()
    with tqdm(total=len(app_ids), desc="Scraping (App Store)", unit="app") as bar:
        for i, app_id in enumerate(app_ids):
            if i > 0:
                time.sleep(_REQUEST_INTERVAL_SECONDS)
            app_id_str = str(app_id).strip()
            try:
                records.append(scrape_ios_bundle(app_id_str))
            except Exception as e:
                fail_count += 1
                tqdm.write(f"Failed to scrape {app_id_str}: {e}")
            bar.set_postfix(ok=len(records), failed=fail_count)
            bar.update(1)

    if not records:
        log.warning("No records were successfully scraped.")
        return

    df_raw = pd.DataFrame(records)
    df_raw = fill_invalid_generic.fill(df_raw, stats=athena_stats)
    valid, dropped = _filter_invalid(df_raw)
    log.info("Valid: %d  Invalid: %d", len(valid), len(dropped))

    if len(dropped):
        dropped.to_csv(invalid_output, index=False)
        log.info("Invalid records saved -> %s", invalid_output)

    if valid.empty:
        log.warning("No valid records to enrich. Exiting.")
        return

    df_enriched = enrich_ios_data(valid)
    df_enriched.to_csv(output_path, index=False)
    log.info("Batch processing complete. Output saved to %s", output_path)
