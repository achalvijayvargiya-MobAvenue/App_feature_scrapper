# App Feature Scraper

Discovers application bundle IDs, scrapes Google Play and the Apple App Store, enriches features, and publishes CSV files to S3.

`run_pipeline.py` is the daily entry point. `RUNBOOK.md` is the operator guide for single-app runs, batch CSV runs, and the commands below. This README records the repository layout and the values in `config.yaml`.

`requirements.txt` does not declare a Python version.

## Install

From this repository's root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Athena and S3 use the AWS default credential chain. `config.yaml` sets `aws.region` to `ap-southeast-1` and `aws.profile` to null. Do not commit access keys.

`RUNBOOK.md` also shows activation of an existing environment at `C:\App_features\venv`. That path is specific to the machine where the runbook was written.

## Configuration

`config.yaml` is the file `run_pipeline.py` loads with `--config`. Active values in this checkout:

| Key | Value |
| --- | --- |
| `aws.region` | `ap-southeast-1` |
| `s3.bucket` | `mobavenue-simplismart-aws-s3-apse-sg` |
| `s3.latest_prefix` | `rtb/data/ml/app_features/latest/` |
| `s3.base_prefix` | `rtb/data/ml/app_features/` |
| `s3.invalid_prefix` | `rtb/data/ml/app_features/invalid` |
| `athena.database` | `mobavenue_dsp` |
| `athena.workgroup` | `primary` |
| `athena.output_location` | `s3://mobavenue-simplismart-aws-s3-apse-sg/athena-query-results/` |
| `athena.existing_features_tables` | `prod.app_features`, `prod.app_features_invalid` |
| `query.rtb_bids_table` | `rtb_bids` |
| `query.year` / `query.month` / `query.days` | `2026` / `10` / `10, 11, 12, 13` |
| `query.countries` | `IND`, `USA` (`IND` maps to storefront `in`, `USA` to `us`) |
| `scraping.allow_defaults` | `false` |
| `scraping.max_bundles_per_run` | `100000` |
| `scraping.max_bundles_per_platform` | `100000` |
| `output.latest_filename_pattern` | `app_data_{date}.csv` |

`rtb_bids` is filtered on partitions `year`, `month`, and `day`. Partition columns for `prod.app_features` and `prod.app_features_invalid` are not defined in this repository. Commented `rtb/data/shreyansh/` prefixes in `config.yaml` are inactive.

## Run

From the repository root, as written in `RUNBOOK.md`:

```powershell
python run_pipeline.py --config config.yaml --platform android
python run_pipeline.py --config config.yaml --platform ios
python run_pipeline.py --config config.yaml --platform all
```

| Flag | Default | Meaning |
| --- | --- | --- |
| `--config` | `config.yaml` | Path to the YAML config. |
| `--platform` | `all` | `all`, `android`, or `ios`. |
| `--date` | current UTC date | Override the date as `YYYYMMDD`. |

The daily pipeline does not read a local app list. It queries `mobavenue_dsp.rtb_bids` for bundle IDs that are not already in `prod.app_features` or `prod.app_features_invalid`. Numeric IDs are treated as iOS and package names as Android.

Single-app, batch CSV, and `fill_invalid_generic.py` commands are in `RUNBOOK.md`.

`combine.py` merges two local enriched CSVs. Its docstring shows:

```text
python combine.py --android android_output.csv --ios ios_output.csv --output app_data_combined.csv
```

`RUNBOOK.md` also tells you to run `merge.py`. That file is not in this repository.

## Output

Valid rows are uploaded under `s3.latest_prefix` using `output.latest_filename_pattern`. `{date}` becomes `YYYYMMDD_HHMMSS`. Invalid rows are uploaded as `invalid_data_YYYYMMDD_HHMMSS.csv` under `s3.invalid_prefix`, keeping `bundle_id` and `country_code`.

Discovered bundles are archived at `s3://<bucket>/<base_prefix>/<YYYYMMDD>/<HHMMSS>/extracted_bundles.csv`.

Final column order is `FINAL_COLUMNS` in `android/constants.py` and `ios/constants.py`.

Default local files for the helper scripts:

| Script | Default output |
| --- | --- |
| `android/run_single_bundle.py` | `single_bundle_output_android.csv` |
| `ios/run_single_bundle.py` | `single_bundle_output_ios.csv` |
| `android/run_pipeline.py` | `batch_output_android.csv` |
| `ios/run_pipeline.py` | `batch_output_ios.csv` |
| `fill_invalid_generic.py` | `generic_filled_records.csv` |
