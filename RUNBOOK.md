# RUNBOOK — App Feature Scraper & Enrichment Pipeline

A simple, practical guide on how to configure and run the scraping and enrichment workflows.

---

## 1. Setup & Environment (Once per terminal session)

Activate the Python virtual environment:

```powershell
# PowerShell:
C:\App_features\venv\Scripts\Activate.ps1

# If script execution is blocked on Windows:
powershell -ExecutionPolicy Bypass -File C:\App_features\venv\Scripts\Activate.ps1
```

```bash
# Git Bash:
source "c:/App_features/venv/Scripts/activate"
```

> [!NOTE]
> **AWS Credentials**: Ensure your AWS CLI credentials (`~/.aws/credentials`) are configured for region `ap-southeast-1`. The Athena queries and S3 uploads will automatically use your active credential chain.

---

## 2. Automated End-to-End Pipeline (`run_pipeline.py`)

This is the primary automated script. It discovers new app bundles from Athena `rtb_bids`, scrapes metadata from Google Play / Apple App Store, validates & enriches the features, and uploads the results directly to S3.

### Workflow:
1. **Step 1 (Athena Discovery)**: Runs an `EXCEPT` query against `mobavenue_dsp.rtb_bids` to find bundle IDs not yet present in `prod.app_features` or `prod.app_features_invalid`.
2. **Step 2 (Archive Extracted)**: Saves the raw discovered bundle list to S3.
3. **Step 3 (Scrape & Enrich)**: Segregates bundles by platform (numeric IDs $\rightarrow$ iOS, package names $\rightarrow$ Android) and scrapes them against the country storefront (`IND` $\rightarrow$ `in`, `USA` $\rightarrow$ `us`).
4. **Step 4 (Publish to S3)**: Cleans and uploads:
   - Valid enriched records to S3 `latest/` prefix: `app_data_<YYYYMMDD>_<HHMMSS>.csv`
   - Unscrapable / invalid bundles to S3 `invalid/` prefix: `invalid_data_<YYYYMMDD>_<HHMMSS>.csv`

### Configuration (`config.yaml`):
Before running, check or update `config.yaml`:
```yaml
query:
  year: 2026
  month: 10
  days: [10, 11, 12, 13]      # Day partition(s) in rtb_bids to scan
  countries: [IND, USA]        # Target country storefronts

scraping:
  allow_defaults: false        # Set true to fill defaults from Athena stats; false to drop missing apps
  max_bundles_per_run: 100000  # Cap on bundles discovered per run
  max_bundles_per_platform: 100000
```

### Execution Commands:

#### Recommended: Run Concurrently in Two Terminals
Running Android and iOS side-by-side in separate terminals prevents log mixing and speeds up execution:

```powershell
# Terminal 1 — Android:
python run_pipeline.py --config config.yaml --platform android

# Terminal 2 — iOS:
python run_pipeline.py --config config.yaml --platform ios
```

#### Run All Platforms in One Terminal:
```powershell
python run_pipeline.py --config config.yaml --platform all
```

#### Optional Flags:
- `--config <path>`: Path to YAML config (default: `config.yaml`).
- `--platform <all|android|ios>`: Restrict run to a specific OS.
- `--date <YYYYMMDD>`: Override today's date for folder naming.

---

## 3. Scraping a Single App (Testing & Ad-hoc Runs)

Use these scripts when you want to scrape and enrich a specific app without running the whole Athena pipeline.

### Android:
```powershell
# Default country is India ('in'):
python android/run_single_bundle.py com.whatsapp

# Specific country (e.g., USA 'us'):
python android/run_single_bundle.py com.getcopper.copper --country us --output copper_android.csv
```

### iOS:
Accepts either numeric Apple App Store IDs (e.g. `1486647561` or `id1486647561`) or bundle IDs:
```powershell
# Default country is India ('in'):
python ios/run_single_bundle.py 310633997

# Specific country (e.g., USA 'us'):
python ios/run_single_bundle.py 1486647561 --country us --output copper_ios.csv
```

> [!TIP]
> Single bundle runs save the output CSV locally (`--output`). If you want to push the file to S3:
> ```powershell
> aws s3 cp .\copper_android.csv s3://mobavenue-simplismart-aws-s3-apse-sg/rtb/data/ml/app_features/latest/
> ```

---

## 4. Batch CSV Scraping (Offline / Custom List)

If you already have a CSV containing app IDs and want to run the full scrape + enrichment flow without Athena:

### Android Batch:
```powershell
python android/run_pipeline.py my_android_apps.csv --id-col bundle_id --output enriched_android.csv
```

### iOS Batch:
```powershell
python ios/run_pipeline.py my_ios_apps.csv --id-col bundle_id --output enriched_ios.csv
```

---

## 5. Standalone Utilities

### `fill_invalid_generic.py` (Default Filling)
Queries live statistics (`score` mode, `ratings_count` mode, `installs` median) from `prod.app_features` in Athena and fills missing fields in any incomplete CSV:
```powershell
python fill_invalid_generic.py --input invalid_records.csv --output generic_filled_records.csv
```

### `merge.py` (S3 Daily Rollup)
Located at project root (`c:\App_features\merge.py`). If multiple pipeline runs occurred on the same day, this combines all timestamped `app_data_<YYYYMMDD>_<HHMMSS>.csv` files into a single consolidated `app_data_merged_<YYYYMMDD>.csv`:
```powershell
# In c:\App_features:
python merge.py
```
*(Make sure `INPUT_PREFIX` and `OUTPUT_PREFIX` in `merge.py` point to your desired S3 directory, e.g. `rtb/data/ml/app_features/...`).*
