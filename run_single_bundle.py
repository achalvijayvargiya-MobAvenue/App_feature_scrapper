"""
run_single_bundle.py
====================
End-to-end flow for a single bundle ID: scrape (or load from CSV), fill gaps,
validate, enrich, and output the full feature row.

Usage:
    python run_single_bundle.py com.kotak811mobilebankingapp.instantsavingsupiscanandpayrecharge
    python run_single_bundle.py com.example.app --from-csv App_dataOutput_all10_filled.csv --output result.csv
"""

import argparse
import logging
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from google_play_scraper import app as gps_app
from google_play_scraper.exceptions import NotFoundError

from fill_invalid_generic import FILLABLE_COLS, _derive_fill_values, _is_empty
from orchestrate import run as orchestrate_run

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def _scrape_bundle(bundle_id: str) -> pd.DataFrame:
    """Scrape a single bundle ID from Play Store and return a 1-row DataFrame."""
    try:
        data = gps_app(bundle_id, lang="en", country="in")
    except NotFoundError:
        log.error("Bundle not found on Play Store: %s", bundle_id)
        sys.exit(1)
    except Exception as exc:
        log.error("Scrape failed: %s", exc)
        sys.exit(1)

    released_str = data.get("released") or ""
    days = None
    for fmt in ("%b %d, %Y", "%Y-%m-%d", "%B %d, %Y"):
        try:
            delta = (
                datetime.now(timezone.utc).replace(tzinfo=None)
                - datetime.strptime(released_str, fmt)
            )
            days = max(0, delta.days)
            break
        except ValueError:
            continue

    row = {
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
        "days_since_released": days,
        "real_installs": data.get("realInstalls"),
    }
    return pd.DataFrame([row])


def _load_from_csv(csv_path: Path, bundle_id: str) -> pd.DataFrame | None:
    """Load and return the row for bundle_id from CSV, or None if not found."""
    df = pd.read_csv(csv_path, low_memory=False)
    df.columns = [c.lower() for c in df.columns]
    if "bundle_id" not in df.columns:
        log.error("CSV has no bundle_id column.")
        return None
    mask = df["bundle_id"].astype(str).str.strip() == bundle_id.strip()
    if not mask.any():
        return None
    return df[mask].copy()


def _apply_fill_generic(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing score, ratings_count, days_since_released with derived values."""
    df = df.copy()
    for col in FILLABLE_COLS:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].astype(object)

    for idx in df.index:
        record = df.loc[idx].to_dict()
        derived = _derive_fill_values(record)
        for col in FILLABLE_COLS:
            if _is_empty(record.get(col)):
                df.at[idx, col] = derived[col]
    return df


def run_single_bundle(
    bundle_id: str,
    *,
    from_csv: Path | None = None,
    output_path: Path = Path("single_bundle_output.csv"),
) -> None:
    bundle_id = bundle_id.strip()

    if from_csv and from_csv.exists():
        log.info("Loading from %s …", from_csv.resolve())
        df = _load_from_csv(from_csv, bundle_id)
        if df is None:
            log.warning("Bundle not found in CSV. Scraping …")
            df = _scrape_bundle(bundle_id)
    else:
        log.info("Scraping Play Store for %s …", bundle_id)
        df = _scrape_bundle(bundle_id)

    log.info("Applying fill_invalid_generic for any missing values …")
    df = _apply_fill_generic(df)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        temp_path = Path(f.name)
    try:
        df.to_csv(temp_path, index=False)
        log.info("Running orchestrate (validate + enrich) …")
        orchestrate_run(
            temp_path,
            output_path,
            invalid_output=Path(temp_path.parent / "single_bundle_invalid.csv"),
        )
    finally:
        temp_path.unlink(missing_ok=True)

    log.info("Done. Output → %s", output_path.resolve())


def _parse_args():
    p = argparse.ArgumentParser(
        description="Run end-to-end flow for a single bundle ID."
    )
    p.add_argument(
        "bundle_id",
        help="Bundle ID to process (e.g. com.kotak811mobilebankingapp.instantsavingsupiscanandpayrecharge)",
    )
    p.add_argument(
        "--from-csv", type=Path, default=None,
        help="Optional: load bundle from this CSV instead of scraping",
    )
    p.add_argument(
        "--output", type=Path, default=Path("single_bundle_output.csv"),
        help="Output CSV path (default: single_bundle_output.csv)",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_single_bundle(
        args.bundle_id,
        from_csv=args.from_csv,
        output_path=args.output,
    )
