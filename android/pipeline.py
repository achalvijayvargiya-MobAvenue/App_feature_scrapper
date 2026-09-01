"""
android/pipeline.py
====================
Entry points for executing the Android scrape + validate + enrich pipeline.

Pipeline steps
--------------
  1. Load & normalise columns
  2. Resolve installs  (real_installs -> installs)
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

import concurrent.futures
import logging
from functools import reduce
from pathlib import Path

import pandas as pd
from tqdm import tqdm
# pyrefly: ignore [missing-import]
from google_play_scraper.exceptions import NotFoundError

import fill_invalid_generic
from android.constants import FINAL_COLUMNS, PASSTHROUGH_COLS, REQUIRED_COLS
from android.scraper import scrape_android_bundle
from android.enrichers import (
    category_mapper,
    score_binner,
    install_binner,
    app_age_binner,
    pricing_flagger,
    content_rating_flagger,
)

log = logging.getLogger(__name__)
SEPARATOR = "=" * 70


def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [c.lower() for c in df.columns]
    if "bundle_id" not in df.columns:
        if "pub_bundle" in df.columns:
            df = df.rename(columns={"pub_bundle": "bundle_id"})
            log.info("Renamed column 'pub_bundle' -> 'bundle_id'.")
        elif len(df.columns) == 1:
            original = df.columns[0]
            df = df.rename(columns={original: "bundle_id"})
            log.info("Single-column input - renamed '%s' -> 'bundle_id'.", original)
    return df


def _is_empty(val) -> bool:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return True
    return str(val).strip() == ""


def _hydrate_missing_required_fields(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing required fields by scraping Play Store per bundle ID."""
    df = df.copy()
    for col in REQUIRED_COLS:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].astype(object)

    needs_scrape = pd.Series(False, index=df.index)
    for col in REQUIRED_COLS:
        needs_scrape |= df[col].isna() | (df[col].astype(str).str.strip() == "")
    needs_scrape &= ~(df["bundle_id"].isna() | (df["bundle_id"].astype(str).str.strip() == ""))

    targets = df.index[needs_scrape].tolist()
    if not targets:
        return df

    log.info("Hydrating missing required fields via Play Store scrape for %d rows ...", len(targets))
    if "real_installs" not in df.columns:
        df["real_installs"] = None

    def fetch(idx):
        bundle_id = str(df.at[idx, "bundle_id"]).strip()
        try:
            return idx, scrape_android_bundle(bundle_id), None
        except NotFoundError:
            return idx, None, f"Bundle not found on Play Store: {bundle_id}"
        except Exception as exc:
            return idx, None, f"Scrape failed for {bundle_id}: {exc}"

    success_count = 0
    fail_count = 0
    REQUEST_TIMEOUT_SEC = 30  # google_play_scraper uses urlopen() with no timeout of its own
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(fetch, idx): idx for idx in targets}
        pending = set(futures)
        with tqdm(total=len(targets), desc="Hydrating (Play Store)", unit="bundle") as bar:
            while pending:
                # timeout bounds how long we wait per poll, not per-request: a
                # request stuck in urlopen() with no library-level timeout would
                # otherwise never appear in as_completed() at all, silently
                # stalling the whole batch (see 2026-08-25 incident).
                newly_done, pending = concurrent.futures.wait(
                    pending, timeout=REQUEST_TIMEOUT_SEC,
                    return_when=concurrent.futures.FIRST_COMPLETED,
                )
                if not newly_done:
                    still_running = [str(df.at[futures[f], "bundle_id"]).strip() for f in pending]
                    tqdm.write(f"No scrapes completed in the last {REQUEST_TIMEOUT_SEC}s — "
                               f"{len(pending)} still in flight (e.g. {still_running[:5]})")
                    continue

                for future in newly_done:
                    idx = futures[future]
                    idx, scraped, err = future.result()
                    if err:
                        fail_count += 1
                        bar.set_postfix(ok=success_count, failed=fail_count)
                        bar.update(1)
                        continue
                    for col in REQUIRED_COLS:
                        if _is_empty(df.at[idx, col]) and not _is_empty(scraped.get(col)):
                            df.at[idx, col] = scraped[col]
                    if _is_empty(df.at[idx, "real_installs"]) and not _is_empty(scraped.get("real_installs")):
                        df.at[idx, "real_installs"] = scraped.get("real_installs")
                    success_count += 1
                    bar.set_postfix(ok=success_count, failed=fail_count)
                    bar.update(1)

    log.info("Scrape hydrate complete: %d/%d rows fetched (%d failed/not found).",
              success_count, len(targets), fail_count)
    return df


def _resolve_installs(df: pd.DataFrame) -> pd.DataFrame:
    """Prefer real_installs over installs when available."""
    if "real_installs" not in df.columns:
        return df

    real = pd.to_numeric(df["real_installs"], errors="coerce")
    has_real = real.notna() & (real > 0)

    if has_real.any():
        df = df.copy()
        df["installs"] = df["installs"].astype(object)
        vals = real[has_real].astype("Int64")
        df.loc[has_real, "installs"] = vals.astype(str)

    return df.drop(columns=["real_installs"])


def _derive_months(df: pd.DataFrame) -> pd.Series:
    if "months_since_launch" in df.columns:
        m = pd.to_numeric(df["months_since_launch"], errors="coerce")
        if m.notna().any():
            return m
    days = pd.to_numeric(df.get("days_since_released", pd.Series(dtype=float)), errors="coerce")
    return (days / 30.44).apply(lambda x: int(x) if pd.notna(x) else None)


def _select_passthrough(df: pd.DataFrame) -> pd.DataFrame:
    base = pd.DataFrame()
    for col in PASSTHROUGH_COLS:
        if col == "months_since_launch":
            continue
        base[col] = df.get(col, "")
    base["months_since_launch"] = _derive_months(df)
    return base.reset_index(drop=True)


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


def enrich_android_data(valid: pd.DataFrame) -> pd.DataFrame:
    valid = valid.copy()
    valid["country_code"] = "IND"
    valid["os_type"] = "ANDROID"
    valid["default"] = valid.get("default", None)

    base = _select_passthrough(valid)

    enrichment_parts = [
        category_mapper.enrich(valid),
        score_binner.enrich(valid),
        install_binner.enrich(valid),
        app_age_binner.enrich(valid),
        pricing_flagger.enrich(valid),
        content_rating_flagger.enrich(valid),
    ]

    enriched = _merge_enrichments([base] + enrichment_parts)

    for col in FINAL_COLUMNS:
        if col not in enriched.columns:
            enriched[col] = None

    return enriched[FINAL_COLUMNS]


def run_single_android(bundle_id: str, output_path: str = "single_bundle_output_android.csv",
                        athena_stats: dict | None = None):
    log.info("Scraping Play Store for %s ...", bundle_id)
    try:
        data = scrape_android_bundle(bundle_id.strip())
        df = pd.DataFrame([data])
    except Exception as e:
        log.error("Failed to scrape %s: %s", bundle_id, e)
        return

    df = _hydrate_missing_required_fields(df)
    df = _resolve_installs(df)
    df["months_since_launch"] = _derive_months(df)
    df["os_type"] = "ANDROID"
    df = fill_invalid_generic.fill(df, stats=athena_stats)
    valid, dropped = _filter_invalid(df)
    if valid.empty:
        log.warning("Bundle %s failed validation, no output written.", bundle_id)
        return

    log.info("Enriching Android data...")
    df_enriched = enrich_android_data(valid)

    log.info("Saving to %s...", output_path)
    df_enriched.to_csv(output_path, index=False)
    log.info("Done.")


def run_batch_android(input_csv: str, output_path: str, id_col: str = "bundle_id",
                       invalid_output: str = "invalid_records_android.csv",
                       athena_stats: dict | None = None):
    log.info(SEPARATOR)
    log.info("ANDROID PIPELINE - Validate & Enrich")
    log.info(SEPARATOR)

    raw = pd.read_csv(input_csv, low_memory=False)
    raw = _normalise_columns(raw)
    if id_col != "bundle_id" and id_col in raw.columns:
        raw = raw.rename(columns={id_col: "bundle_id"})

    raw = _hydrate_missing_required_fields(raw)
    raw = _resolve_installs(raw)
    raw["months_since_launch"] = _derive_months(raw)
    raw["os_type"] = "ANDROID"
    raw = fill_invalid_generic.fill(raw, stats=athena_stats)
    log.info("Loaded %d rows, %d columns.", len(raw), len(raw.columns))

    valid, dropped = _filter_invalid(raw)
    log.info("Valid: %d  Invalid: %d", len(valid), len(dropped))

    if len(dropped):
        dropped.to_csv(invalid_output, index=False)
        log.info("Invalid records saved -> %s", invalid_output)

    if valid.empty:
        log.warning("No valid records to enrich. Exiting.")
        return

    df_enriched = enrich_android_data(valid)

    df_enriched.to_csv(output_path, index=False)
    log.info("Batch processing complete. Output saved to %s", output_path)
