"""
run_pipeline.py
================
End-to-end daily automation:

  1. Run the EXCEPT query in Athena: bundle_ids seen in rtb_bids for the
     configured day(s) that are NOT already in the features table
     (imp_tables.app_feature_raw_test is read directly by Athena — nothing
     is downloaded locally for this comparison).
  2. Save the new bundle_ids to
         s3://<bucket>/<base_prefix>/<YYYYMMDD>/extracted_bundles.csv
     (YYYYMMDD = today, always).
  3. Feed extracted_bundles.csv into both android.pipeline and ios.pipeline
     (each scrapes, validates, and enriches independently), then combine
     the two outputs into one CSV via combine.py -> app_data_<YYYYMMDD>.csv,
     saved under the same dated folder.
  4. Clean/standardize that CSV and upload it as a NEW file inside the
     "latest" S3 folder. Nothing already in "latest" is read, merged, or
     overwritten — each run just adds one more file.

Requires android/, ios/, combine.py, and pipeline/ (this file's siblings)
to be importable — run from the App_feature_scrapper directory.

Usage:
    python run_pipeline.py --config config.yaml
    python run_pipeline.py --config config.yaml --date 20260714   # override "today"
"""

from __future__ import annotations

import argparse
import logging
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline.s3_utils import make_s3_client, upload_file
from pipeline.athena_utils import run_query_to_dataframe
from pipeline.clean_output import clean_dataframe

import fill_invalid_generic
from android.pipeline import run_batch_android
from ios.pipeline import run_batch_ios
from combine import combine as combine_outputs

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("run_pipeline")
SEPARATOR = "=" * 78


def load_config(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_except_query(cfg: dict) -> str:
    q = cfg["query"]
    days_sql = ", ".join(str(d) for d in q["days"])
    return f"""
SELECT DISTINCT LOWER(TRIM(pub_bundle)) AS bundle_id
FROM {cfg['athena']['database']}.{q['rtb_bids_table']}
WHERE year = {q['year']}
  AND month = {q['month']}
  AND day IN ({days_sql})
  AND device_id IS NOT NULL
  AND device_id NOT IN ('', 'null')
  AND country_code = 'IND'

EXCEPT

SELECT DISTINCT LOWER(TRIM(bundle_id)) AS bundle_id
FROM {cfg['athena']['existing_features_table']}
""".strip()


# ---------------------------------------------------------------------------
# Step 1
# ---------------------------------------------------------------------------
def step1_run_except_query(cfg: dict) -> pd.DataFrame:
    log.info(SEPARATOR)
    log.info("STEP 1 — Athena EXCEPT query (new bundle_ids)")
    log.info(SEPARATOR)
    sql = build_except_query(cfg)
    df = run_query_to_dataframe(cfg, sql)
    df.columns = [c.lower() for c in df.columns]
    log.info("New bundle_ids found: %d", len(df))
    return df


# ---------------------------------------------------------------------------
# Step 2
# ---------------------------------------------------------------------------
def step2_save_extracted(cfg: dict, s3, bundles_df: pd.DataFrame, workdir: Path, date_str: str) -> Path:
    log.info(SEPARATOR)
    log.info("STEP 2 — Save extracted_bundles.csv to dated S3 folder (%s)", date_str)
    log.info(SEPARATOR)
    local_path = workdir / "extracted_bundles.csv"
    bundles_df.to_csv(local_path, index=False)
    bucket = cfg["s3"]["bucket"]
    key = f"{cfg['s3']['base_prefix'].rstrip('/')}/{date_str}/extracted_bundles.csv"
    upload_file(s3, local_path, bucket, key)
    return local_path


def _segregate_bundles(extracted_csv: Path, workdir: Path, id_col: str = "bundle_id") -> tuple[Path, Path]:
    """
    Split a combined bundle_id list into Android and iOS candidate files
    before either pipeline scrapes anything.

    Apple App Store IDs are purely numeric (e.g. "310633997"); Play Store
    bundle IDs and iOS reverse-domain bundle IDs (e.g. "com.whatsapp",
    "net.whatsapp.WhatsApp") share the same string shape and can't be told
    apart without an API call, so the rule is: numeric -> iOS, everything
    else -> Android. rtb_bids logs iOS impressions by numeric App Store ID
    in practice, so this covers the real split without throwing the whole
    combined list at both APIs (Apple's ~20 req/min limit makes that
    especially wasteful when iOS is the minority of traffic).
    """
    df = pd.read_csv(extracted_csv, low_memory=False)
    df.columns = [c.lower() for c in df.columns]

    # Reindex back to df's full index so rows with a null bundle_id (dropped
    # by dropna()) don't shrink the boolean mask relative to df — those rows
    # are treated as "not iOS" (they fall into Android, same as any other
    # unresolvable bundle_id, rather than crashing df.loc[mask]).
    ids = df[id_col].dropna().astype(str).str.strip()
    is_ios = ids.str.isdigit().reindex(df.index, fill_value=False)

    android_path = workdir / "extracted_bundles_android.csv"
    ios_path = workdir / "extracted_bundles_ios.csv"
    df.loc[~is_ios].to_csv(android_path, index=False)
    df.loc[is_ios].to_csv(ios_path, index=False)

    log.info("Segregated bundle_ids — android: %d  ios: %d", int((~is_ios).sum()), int(is_ios.sum()))
    return android_path, ios_path


def _cap_bundles(csv_path: Path, limit: int | None, label: str) -> Path:
    """Trim a segregated bundle_id file to at most `limit` rows, in place."""
    if not limit:
        return csv_path
    df = pd.read_csv(csv_path, low_memory=False)
    if len(df) > limit:
        log.info("Limiting %s run to %d of %d discovered bundle_ids (scraping.max_bundles_per_platform).",
                  label, limit, len(df))
        df = df.head(limit)
        df.to_csv(csv_path, index=False)
    return csv_path


# ---------------------------------------------------------------------------
# Step 3
# ---------------------------------------------------------------------------
def step3_scrape_enrich(cfg: dict, s3, extracted_local: Path, workdir: Path, date_str: str) -> Path:
    log.info(SEPARATOR)
    log.info("STEP 3 — Scrape / validate / enrich (Android + iOS) and combine")
    log.info(SEPARATOR)

    android_input, ios_input = _segregate_bundles(extracted_local, workdir)

    per_platform_limit = cfg["scraping"].get("max_bundles_per_platform")
    android_input = _cap_bundles(android_input, per_platform_limit, "android")
    ios_input = _cap_bundles(ios_input, per_platform_limit, "ios")

    android_csv = workdir / "android_output.csv"
    ios_csv = workdir / "ios_output.csv"
    android_invalid = workdir / "android_invalid.csv"
    ios_invalid = workdir / "ios_invalid.csv"

    log.info("Fetching Athena fill-default stats (score/ratings_count/installs MODE-MEDIAN) once for this run ...")
    athena_stats = fill_invalid_generic._fetch_athena_stats()

    run_batch_android(str(android_input), str(android_csv), invalid_output=str(android_invalid),
                       athena_stats=athena_stats)
    run_batch_ios(str(ios_input), str(ios_csv), invalid_output=str(ios_invalid),
                  athena_stats=athena_stats)

    # Every CSV that reaches S3 must be newline-safe (embedded \n/\r inside a
    # text field like description breaks Athena's CSV SerDe and other
    # line-based readers downstream) — clean before each upload, not just the
    # final "latest" publish in step 4.
    _clean_csv_in_place(android_csv)
    _clean_csv_in_place(ios_csv)

    for invalid_csv in (android_invalid, ios_invalid):
        if invalid_csv.exists():
            _clean_csv_in_place(invalid_csv)
            bucket = cfg["s3"]["bucket"]
            key = f"{cfg['s3']['base_prefix'].rstrip('/')}/{date_str}/{invalid_csv.name}"
            upload_file(s3, invalid_csv, bucket, key)
            log.info("Unscrapable / invalid bundles archived to s3://%s/%s", bucket, key)

    output_csv = workdir / f"app_data_{date_str}.csv"
    combine_outputs(android_csv, ios_csv, output_csv)

    if not output_csv.exists():
        raise RuntimeError("combine() produced no output — nothing to publish.")

    _clean_csv_in_place(output_csv)

    bucket = cfg["s3"]["bucket"]
    key = f"{cfg['s3']['base_prefix'].rstrip('/')}/{date_str}/{output_csv.name}"
    upload_file(s3, output_csv, bucket, key)
    return output_csv


def _clean_csv_in_place(csv_path: Path) -> None:
    """Run clean_dataframe() over a CSV already on disk and overwrite it."""
    if not csv_path.exists():
        return
    df = pd.read_csv(csv_path, low_memory=False)
    df = clean_dataframe(df)
    df.to_csv(csv_path, index=False)


# ---------------------------------------------------------------------------
# Step 4
# ---------------------------------------------------------------------------
def step4_clean_and_publish(cfg: dict, s3, enriched_csv: Path, workdir: Path, date_str: str) -> str:
    """
    Clean the newly-scraped/enriched rows and upload them as a NEW file
    inside the 'latest' folder. Existing files under latest_prefix are never
    read, merged, or overwritten — this only adds a file.
    """
    log.info(SEPARATOR)
    log.info("STEP 4 — Clean and publish as a new file in 'latest' (append-only)")
    log.info(SEPARATOR)

    new_df = pd.read_csv(enriched_csv, low_memory=False)
    new_df = clean_dataframe(new_df)
    log.info("Cleaned rows ready to publish: %d", len(new_df))

    filename = cfg["output"]["latest_filename_pattern"].format(date=date_str)
    local_out = workdir / filename
    new_df.to_csv(local_out, index=False)

    bucket = cfg["s3"]["bucket"]
    key = f"{cfg['s3']['latest_prefix'].rstrip('/')}/{filename}"
    return upload_file(s3, local_out, bucket, key)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Daily bundle-discovery + enrichment pipeline.")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--date", type=str, default=None,
                         help="Override today's date, format YYYYMMDD (default: actual today).")
    args = parser.parse_args()

    cfg = load_config(args.config)
    date_str = args.date or datetime.now(timezone.utc).strftime("%Y%m%d")

    log.info(SEPARATOR)
    log.info("RUN_PIPELINE — date=%s", date_str)
    log.info(SEPARATOR)

    s3 = make_s3_client(cfg)

    with tempfile.TemporaryDirectory(prefix="pipeline_") as tmp:
        workdir = Path(tmp)

        bundles_df = step1_run_except_query(cfg)
        if bundles_df.empty:
            log.info("No new bundle_ids found — nothing to scrape. Exiting.")
            return

        limit = cfg["scraping"].get("max_bundles_per_run")
        if limit and len(bundles_df) > limit:
            log.info("Limiting this run to %d of %d discovered bundle_ids (scraping.max_bundles_per_run).",
                      limit, len(bundles_df))
            bundles_df = bundles_df.head(limit)

        extracted_local = step2_save_extracted(cfg, s3, bundles_df, workdir, date_str)
        enriched_csv = step3_scrape_enrich(cfg, s3, extracted_local, workdir, date_str)
        final_uri = step4_clean_and_publish(cfg, s3, enriched_csv, workdir, date_str)

    log.info(SEPARATOR)
    log.info("DONE. Published -> %s", final_uri)
    log.info(SEPARATOR)


if __name__ == "__main__":
    main()
