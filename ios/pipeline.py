"""
ios/pipeline.py
===============
Entry points for executing the iOS enrichment pipeline.
"""

import pandas as pd
import logging
from pathlib import Path
from typing import List
from ios.scraper import scrape_ios_bundle, NotFoundError, ScraperError
from ios.enricher import enrich_ios_data

log = logging.getLogger(__name__)

def run_single_ios(app_id: str, output_path: str = "single_bundle_output_ios.csv"):
    log.info(f"Scraping iOS App Store for {app_id}...")
    try:
        data = scrape_ios_bundle(app_id)
        df = pd.DataFrame([data])
    except Exception as e:
        log.error(f"Failed to scrape {app_id}: {e}")
        return

    log.info("Enriching iOS data...")
    df_enriched = enrich_ios_data(df)

    log.info(f"Saving to {output_path}...")
    df_enriched.to_csv(output_path, index=False)
    log.info("Done.")

def run_batch_ios(input_csv: str, output_path: str, id_col: str = "bundle_id"):
    df_in = pd.read_csv(input_csv)
    if id_col not in df_in.columns:
        log.error(f"Input CSV must contain '{id_col}' column.")
        return

    records = []
    for app_id in df_in[id_col].dropna().unique():
        app_id_str = str(app_id).strip()
        log.info(f"Processing {app_id_str}...")
        try:
            data = scrape_ios_bundle(app_id_str)
            records.append(data)
        except Exception as e:
            log.warning(f"Failed to scrape {app_id_str}: {e}")

    if not records:
        log.warning("No records were successfully scraped.")
        return

    df_raw = pd.DataFrame(records)
    df_enriched = enrich_ios_data(df_raw)

    df_enriched.to_csv(output_path, index=False)
    log.info(f"Batch processing complete. Output saved to {output_path}")
