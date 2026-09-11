"""
test_usa_raw_scrape.py
=======================
One-off test: pull ALL distinct bundle_ids seen for country_code = 'USA' in
mobavenue_dsp.rtb_bids (for the query window in config.yaml), scrape each
with the plain android/ios scraper (country='us', no validation/enrichment),
and save the raw fields to a local CSV for manual inspection of what values
each enricher (score_binner, content_rating_flagger, etc.) would actually see.

Output columns are exactly what scrape_android_bundle()/scrape_ios_bundle()
return — no fill_invalid_generic, no enrichers, no S3 upload. Local file
only, so nothing in the real pipeline's S3 locations is touched.

Usage:
    python test_usa_raw_scrape.py --config config.yaml --output usa_raw_scrape.csv
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import pandas as pd
import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline.athena_utils import run_query_to_dataframe

from android.scraper import scrape_android_bundle, ScraperError as AndroidScraperError
# pyrefly: ignore [missing-import]
from google_play_scraper.exceptions import NotFoundError as AndroidNotFoundError
from ios.scraper import scrape_ios_bundle, ScraperError as IosScraperError, NotFoundError as IosNotFoundError

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("test_usa_raw_scrape")

_IOS_REQUEST_INTERVAL_SECONDS = 3.5  # matches ios/pipeline.py's iTunes rate-limit pacing


def load_config(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_usa_distinct_query(cfg: dict) -> str:
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
  AND country_code = 'USA'
""".strip()


def main():
    parser = argparse.ArgumentParser(description="Raw-scrape all distinct USA bundle_ids from rtb_bids.")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--output", type=Path, default=Path("usa_raw_scrape.csv"))
    parser.add_argument("--limit-per-platform", type=int, default=2000,
                         help="Max bundle_ids to scrape per platform (android/ios). Set to 0 for no limit.")
    args = parser.parse_args()

    cfg = load_config(args.config)

    log.info("Querying distinct USA bundle_ids from rtb_bids …")
    sql = build_usa_distinct_query(cfg)
    bundles_df = run_query_to_dataframe(cfg, sql)
    bundles_df.columns = [c.lower() for c in bundles_df.columns]
    bundle_ids = bundles_df["bundle_id"].dropna().astype(str).str.strip().unique().tolist()
    log.info("Found %d distinct USA bundle_ids.", len(bundle_ids))

    if not bundle_ids:
        log.info("Nothing to scrape. Exiting.")
        return

    # Same numeric-ID rule run_pipeline.py uses: Apple App Store IDs are
    # purely numeric, everything else is treated as an Android bundle_id.
    ios_ids = [b for b in bundle_ids if b.isdigit()]
    android_ids = [b for b in bundle_ids if not b.isdigit()]
    log.info("Segregated — android: %d  ios: %d", len(android_ids), len(ios_ids))

    limit = args.limit_per_platform
    if limit:
        if len(android_ids) > limit:
            log.info("Limiting android to %d of %d discovered bundle_ids.", limit, len(android_ids))
            android_ids = android_ids[:limit]
        if len(ios_ids) > limit:
            log.info("Limiting ios to %d of %d discovered bundle_ids.", limit, len(ios_ids))
            ios_ids = ios_ids[:limit]

    records: list[dict] = []

    log.info("Scraping Android (Play Store, country=us) …")
    for bundle_id in tqdm(android_ids, desc="Android", unit="app"):
        try:
            records.append(scrape_android_bundle(bundle_id, country="us"))
        except AndroidNotFoundError:
            log.warning("Not found on Play Store: %s", bundle_id)
        except AndroidScraperError as e:
            log.warning("Scrape failed for %s: %s", bundle_id, e)

    log.info("Scraping iOS (App Store, country=us) …")
    for i, app_id in enumerate(tqdm(ios_ids, desc="iOS", unit="app")):
        if i > 0:
            time.sleep(_IOS_REQUEST_INTERVAL_SECONDS)
        try:
            records.append(scrape_ios_bundle(app_id, country="us"))
        except IosNotFoundError:
            log.warning("Not found on App Store: %s", app_id)
        except IosScraperError as e:
            log.warning("Scrape failed for %s: %s", app_id, e)

    if not records:
        log.warning("No records scraped successfully. Nothing written.")
        return

    df = pd.DataFrame(records)
    df.to_csv(args.output, index=False)
    log.info("Wrote %d raw scraped rows -> %s", len(df), args.output.resolve())


if __name__ == "__main__":
    main()
