"""
android/run_single_bundle.py
=============================
Entry point for scraping and enriching a single Android app.
"""

import argparse
from android.pipeline import run_single_android

def _parse_args():
    p = argparse.ArgumentParser(description="Run end-to-end flow for a single Android app.")
    p.add_argument("bundle_id", help="Play Store bundle ID to process (e.g. com.whatsapp)")
    p.add_argument("--output", default="single_bundle_output_android.csv", help="Output CSV path")
    return p.parse_args()

if __name__ == "__main__":
    args = _parse_args()
    run_single_android(args.bundle_id, output_path=args.output)
