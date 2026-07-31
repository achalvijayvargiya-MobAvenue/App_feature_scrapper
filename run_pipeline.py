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
  3. Feed extracted_bundles.csv into orchestrate.py's pipeline (which
     scrapes any missing required fields bundle-by-bundle, validates, and
     enriches) -> app_data_<YYYYMMDD>.csv, saved under the same dated
     folder.
  4. Clean/standardize that CSV and upload it as a NEW file inside the
     "latest" S3 folder. Nothing already in "latest" is read, merged, or
     overwritten — each run just adds one more file.

Requires orchestrate.py (and the `enrichers` package it imports) to be
importable — either drop this file + pipeline/ next to your existing
orchestrate.py, or add its directory to PYTHONPATH.

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

try:
    from orchestrate import run as orchestrate_run  # your existing script
except ImportError:
    orchestrate_run = None  # validated at runtime with a clear error

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


# ---------------------------------------------------------------------------
# Step 3
# ---------------------------------------------------------------------------
def step3_scrape_enrich(cfg: dict, s3, extracted_local: Path, workdir: Path, date_str: str) -> Path:
    log.info(SEPARATOR)
    log.info("STEP 3 — Scrape / validate / enrich via orchestrate.run()")
    log.info(SEPARATOR)
    if orchestrate_run is None:
        raise RuntimeError(
            "Could not import orchestrate.run — make sure orchestrate.py "
            "(and its `enrichers` package) are on PYTHONPATH, next to this file."
        )

    output_csv = workdir / f"app_data_{date_str}.csv"
    invalid_csv = workdir / f"app_data_{date_str}_invalid.csv"

    # orchestrate.run() will bundle-wise scrape any required fields that are
    # missing — extracted_bundles.csv only needs a bundle_id column.
    orchestrate_run(extracted_local, output_csv, invalid_output=invalid_csv)

    if invalid_csv.exists():
        bucket = cfg["s3"]["bucket"]
        key = f"{cfg['s3']['base_prefix'].rstrip('/')}/{date_str}/{invalid_csv.name}"
        upload_file(s3, invalid_csv, bucket, key)
        log.info("Unscrapable / invalid bundles archived to s3://%s/%s", bucket, key)

    if not output_csv.exists():
        raise RuntimeError("orchestrate.run() produced no valid enriched output — nothing to publish.")

    bucket = cfg["s3"]["bucket"]
    key = f"{cfg['s3']['base_prefix'].rstrip('/')}/{date_str}/{output_csv.name}"
    upload_file(s3, output_csv, bucket, key)
    return output_csv


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
