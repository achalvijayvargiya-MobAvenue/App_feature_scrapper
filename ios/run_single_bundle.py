"""
ios/run_single_bundle.py
=========================
Entry point for scraping and enriching a single iOS app.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ios.pipeline import run_single_ios

def _parse_args():
    p = argparse.ArgumentParser(description="Run end-to-end flow for a single iOS app.")
    p.add_argument("app_id", help="Apple App Store ID or bundle ID to process (e.g. 310633997 or net.whatsapp.WhatsApp)")
    p.add_argument("--country", default="in", help="Storefront country code, e.g. 'in' (IND), 'us' (USA) (default: in)")
    p.add_argument("--output", default="single_bundle_output_ios.csv", help="Output CSV path")
    return p.parse_args()

if __name__ == "__main__":
    args = _parse_args()
    run_single_ios(args.app_id, output_path=args.output, country=args.country)
