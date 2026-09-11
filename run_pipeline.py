"""
run_pipeline.py
================
End-to-end daily automation:

  1. Run the EXCEPT query in Athena: (bundle_id, country_code) pairs seen in
     rtb_bids for the configured day(s) and countries (query.countries in
     config.yaml, e.g. IND + USA) that are NOT already in the features table
     for that same (bundle_id, country_code) pair — an app already scraped
     for one country still gets pulled for a country it's missing.
  2. Save the new (bundle_id, country_code) pairs to
         s3://<bucket>/<base_prefix>/<YYYYMMDD>/<HHMMSS>/extracted_bundles.csv
     (YYYYMMDD = today, HHMMSS = this run's start time — every run gets its
     own subfolder so same-day reruns never overwrite each other's files).
  3. Split that list into 4 buckets (android/ios x each configured country)
     and feed each into android.pipeline / ios.pipeline with the matching
     store country param, so score/content_rating/ratings_count/launch_date
     are scraped from the correct storefront per country rather than reused
     across countries. Each bucket's enriched CSV is uploaded independently.
  4. Clean/standardize each bucket's CSV and upload it as its own NEW file
     inside the "latest" S3 folder (e.g. app_data_android_ind_<date>.csv,
     app_data_android_usa_<date>.csv, app_data_ios_ind_<date>.csv,
     app_data_ios_usa_<date>.csv). bundle_id + country_code together are the
     table's grain, so countries are never combined/deduped into one row.
     Nothing already in "latest" is read, merged, or overwritten — each run
     just adds more files.

Requires android/, ios/, and pipeline/ (this file's siblings) to be
importable — run from the App_feature_scrapper directory.

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
    countries_sql = ", ".join(f"'{c}'" for c in q["countries"])
    return f"""
SELECT DISTINCT LOWER(TRIM(pub_bundle)) AS bundle_id, country_code
FROM {cfg['athena']['database']}.{q['rtb_bids_table']}
WHERE year = {q['year']}
  AND month = {q['month']}
  AND day IN ({days_sql})
  AND device_id IS NOT NULL
  AND device_id NOT IN ('', 'null')
  AND country_code IN ({countries_sql})

EXCEPT

SELECT DISTINCT LOWER(TRIM(bundle_id)) AS bundle_id, country_code
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
def step2_save_extracted(cfg: dict, s3, bundles_df: pd.DataFrame, workdir: Path, date_str: str, run_id: str) -> Path:
    log.info(SEPARATOR)
    log.info("STEP 2 — Save extracted_bundles.csv to dated S3 folder (%s/%s)", date_str, run_id)
    log.info(SEPARATOR)
    local_path = workdir / "extracted_bundles.csv"
    bundles_df.to_csv(local_path, index=False)
    bucket = cfg["s3"]["bucket"]
    key = f"{cfg['s3']['base_prefix'].rstrip('/')}/{date_str}/{run_id}/extracted_bundles.csv"
    upload_file(s3, local_path, bucket, key)
    return local_path


# rtb_bids country_code -> scraper `country` param (google_play_scraper / iTunes lookup).
COUNTRY_CODE_TO_PARAM = {"IND": "in", "USA": "us"}


def _segregate_bundles(extracted_csv: Path, workdir: Path, countries: list[str],
                        id_col: str = "bundle_id") -> dict[tuple[str, str], Path]:
    """
    Split a combined (bundle_id, country_code) list into one candidate file
    per (platform, country) pair before any pipeline scrapes anything.

    Apple App Store IDs are purely numeric (e.g. "310633997"); Play Store
    bundle IDs and iOS reverse-domain bundle IDs (e.g. "com.whatsapp",
    "net.whatsapp.WhatsApp") share the same string shape and can't be told
    apart without an API call, so the rule is: numeric -> iOS, everything
    else -> Android. rtb_bids logs iOS impressions by numeric App Store ID
    in practice, so this covers the real split without throwing the whole
    combined list at both APIs (Apple's ~20 req/min limit makes that
    especially wasteful when iOS is the minority of traffic).

    Returns {(platform, country_code): path}, e.g.
    {("android", "IND"): ..., ("android", "USA"): ..., ("ios", "IND"): ..., ("ios", "USA"): ...}
    """
    df = pd.read_csv(extracted_csv, low_memory=False)
    df.columns = [c.lower() for c in df.columns]
    df["country_code"] = df["country_code"].astype(str).str.upper()

    # Reindex back to df's full index so rows with a null bundle_id (dropped
    # by dropna()) don't shrink the boolean mask relative to df — those rows
    # are treated as "not iOS" (they fall into Android, same as any other
    # unresolvable bundle_id, rather than crashing df.loc[mask]).
    ids = df[id_col].dropna().astype(str).str.strip()
    is_ios = ids.str.isdigit().reindex(df.index, fill_value=False)

    paths: dict[tuple[str, str], Path] = {}
    for platform, platform_mask in (("android", ~is_ios), ("ios", is_ios)):
        for country in countries:
            mask = platform_mask & (df["country_code"] == country)
            subset = df.loc[mask]
            path = workdir / f"extracted_bundles_{platform}_{country.lower()}.csv"
            subset.to_csv(path, index=False)
            paths[(platform, country)] = path
            log.info("Segregated bundle_ids — %s/%s: %d", platform, country, len(subset))

    return paths


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
def step3_scrape_enrich(cfg: dict, s3, extracted_local: Path, workdir: Path, date_str: str, run_id: str) -> dict[tuple[str, str], Path]:
    """
    Scrape / validate / enrich each (platform, country) bucket independently
    and upload its enriched CSV + invalid-records CSV to the dated run
    folder. Returns {(platform, country): enriched_csv_path}.
    """
    log.info(SEPARATOR)
    log.info("STEP 3 — Scrape / validate / enrich (Android + iOS) x (IND + USA)")
    log.info(SEPARATOR)

    countries = cfg["query"]["countries"]
    segregated = _segregate_bundles(extracted_local, workdir, countries)

    per_platform_limit = cfg["scraping"].get("max_bundles_per_platform")
    for key, path in segregated.items():
        platform, _ = key
        segregated[key] = _cap_bundles(path, per_platform_limit, f"{platform}/{key[1]}")

    log.info("Fetching Athena fill-default stats (score/ratings_count/installs MODE-MEDIAN) once for this run ...")
    athena_stats = fill_invalid_generic._fetch_athena_stats()

    runners = {"android": run_batch_android, "ios": run_batch_ios}
    enriched_paths: dict[tuple[str, str], Path] = {}

    for (platform, country), input_path in segregated.items():
        country_param = COUNTRY_CODE_TO_PARAM.get(country, country.lower())
        label = f"{platform}_{country.lower()}"
        output_csv = workdir / f"{label}_output.csv"
        invalid_csv = workdir / f"{label}_invalid.csv"

        runners[platform](str(input_path), str(output_csv), invalid_output=str(invalid_csv),
                           athena_stats=athena_stats, country=country_param)

        # Every CSV that reaches S3 must be newline-safe (embedded \n/\r inside
        # a text field like description breaks Athena's CSV SerDe and other
        # line-based readers downstream) — clean before each upload, not just
        # the final "latest" publish in step 4.
        _clean_csv_in_place(output_csv)
        if invalid_csv.exists():
            _clean_csv_in_place(invalid_csv)
            bucket = cfg["s3"]["bucket"]
            key = f"{cfg['s3']['base_prefix'].rstrip('/')}/{date_str}/{run_id}/{invalid_csv.name}"
            upload_file(s3, invalid_csv, bucket, key)
            log.info("Unscrapable / invalid bundles archived to s3://%s/%s", bucket, key)

        if output_csv.exists():
            bucket = cfg["s3"]["bucket"]
            key = f"{cfg['s3']['base_prefix'].rstrip('/')}/{date_str}/{run_id}/{output_csv.name}"
            upload_file(s3, output_csv, bucket, key)
            enriched_paths[(platform, country)] = output_csv
        else:
            log.warning("No output produced for %s/%s — nothing to publish for this bucket.", platform, country)

    return enriched_paths


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
def step4_clean_and_publish(cfg: dict, s3, enriched_paths: dict[tuple[str, str], Path],
                             workdir: Path, date_str: str, run_id: str) -> list[str]:
    """
    Clean each (platform, country) bucket's newly-scraped/enriched rows and
    upload each as its own NEW file inside the 'latest' folder — one file per
    platform/country combination (e.g. app_data_android_ind_<date>.csv),
    since bundle_id + country_code together are the table's grain and must
    not be collapsed into one row. Existing files under latest_prefix are
    never read, merged, or overwritten — this only adds files.
    """
    log.info(SEPARATOR)
    log.info("STEP 4 — Clean and publish 4 new files in 'latest' (append-only)")
    log.info(SEPARATOR)

    published_uris = []
    for (platform, country), enriched_csv in enriched_paths.items():
        new_df = pd.read_csv(enriched_csv, low_memory=False)
        new_df = clean_dataframe(new_df)
        log.info("Cleaned rows ready to publish for %s/%s: %d", platform, country, len(new_df))

        filename = f"app_data_{platform}_{country.lower()}_{date_str}_{run_id}.csv"
        local_out = workdir / filename
        new_df.to_csv(local_out, index=False)

        bucket = cfg["s3"]["bucket"]
        key = f"{cfg['s3']['latest_prefix'].rstrip('/')}/{filename}"
        published_uris.append(upload_file(s3, local_out, bucket, key))

    return published_uris


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
    now = datetime.now(timezone.utc)
    date_str = args.date or now.strftime("%Y%m%d")
    run_id = now.strftime("%H%M%S")

    log.info(SEPARATOR)
    log.info("RUN_PIPELINE — date=%s run_id=%s", date_str, run_id)
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

        extracted_local = step2_save_extracted(cfg, s3, bundles_df, workdir, date_str, run_id)
        enriched_paths = step3_scrape_enrich(cfg, s3, extracted_local, workdir, date_str, run_id)
        if not enriched_paths:
            log.info("No buckets produced output — nothing to publish. Exiting.")
            return
        final_uris = step4_clean_and_publish(cfg, s3, enriched_paths, workdir, date_str, run_id)

    log.info(SEPARATOR)
    log.info("DONE. Published:")
    for uri in final_uris:
        log.info("  -> %s", uri)
    log.info(SEPARATOR)


if __name__ == "__main__":
    main()
