"""
combine.py
==========
Merges the Android and iOS enriched output CSVs into a single combined CSV.
Both pipelines already produce the same column schema; os_type distinguishes
rows ("ANDROID" vs "IOS").

Usage:
    python combine.py --android android_output.csv --ios ios_output.csv --output app_data_combined.csv
"""

import argparse
import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def combine(android_csv: Path, ios_csv: Path, output_path: Path) -> None:
    frames = []
    for label, path in (("android", android_csv), ("ios", ios_csv)):
        if path and path.exists():
            df = pd.read_csv(path, low_memory=False)
            log.info("Loaded %d rows from %s (%s)", len(df), path, label)
            frames.append(df)
        else:
            log.warning("%s output not found, skipping: %s", label, path)

    if not frames:
        log.error("No input files found — nothing to combine.")
        return

    combined = pd.concat(frames, ignore_index=True, sort=False)
    combined.to_csv(output_path, index=False)
    log.info("Combined %d rows -> %s", len(combined), output_path.resolve())


def _parse_args():
    p = argparse.ArgumentParser(description="Combine Android and iOS enriched CSVs into one file.")
    p.add_argument("--android", type=Path, default=Path("batch_output_android.csv"))
    p.add_argument("--ios", type=Path, default=Path("batch_output_ios.csv"))
    p.add_argument("--output", type=Path, default=Path("app_data_combined.csv"))
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    combine(args.android, args.ios, args.output)
