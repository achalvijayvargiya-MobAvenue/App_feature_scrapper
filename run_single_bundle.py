"""
run_single_bundle.py
====================
End-to-end flow for a single bundle ID: look it up in an existing CSV if
given, then hand off to orchestrate.run() — which scrapes any missing
fields, fills defaults, validates, and enriches.

Usage:
    python run_single_bundle.py com.kotak811mobilebankingapp.instantsavingsupiscanandpayrecharge
    python run_single_bundle.py com.example.app --from-csv App_dataOutput_all10_filled.csv --output result.csv
"""

import argparse
import logging
import tempfile
from pathlib import Path

import pandas as pd

from orchestrate import run as orchestrate_run

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


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


def run_single_bundle(
    bundle_id: str,
    *,
    from_csv: Path | None = None,
    output_path: Path = Path("single_bundle_output.csv"),
) -> None:
    bundle_id = bundle_id.strip()

    df = None
    if from_csv and from_csv.exists():
        log.info("Loading from %s …", from_csv.resolve())
        df = _load_from_csv(from_csv, bundle_id)
        if df is None:
            log.warning("Bundle not found in CSV. Will scrape via orchestrate.")

    if df is None:
        df = pd.DataFrame([{"bundle_id": bundle_id}])

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        temp_path = Path(f.name)
    try:
        df.to_csv(temp_path, index=False)
        log.info("Running orchestrate (scrape + validate + enrich) …")
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
