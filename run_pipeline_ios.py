"""
run_pipeline_ios.py
===================
Entry point for running the iOS enrichment pipeline on a batch of apps.
"""

import argparse
from ios.pipeline import run_batch_ios

def _parse_args():
    p = argparse.ArgumentParser(description="Run iOS pipeline over a CSV of app IDs.")
    p.add_argument("input_csv", help="Path to input CSV containing app IDs.")
    p.add_argument("--id-col", default="bundle_id", help="Name of the column containing the app ID.")
    p.add_argument("--output", default="batch_output_ios.csv", help="Output CSV path")
    return p.parse_args()

if __name__ == "__main__":
    args = _parse_args()
    run_batch_ios(args.input_csv, output_path=args.output, id_col=args.id_col)
