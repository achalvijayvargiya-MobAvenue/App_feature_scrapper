# RUNBOOK — App_feature_scrapper

How to run each script in this project.

## Setup (once per shell session)

Activate the venv (adjust path if yours differs):

```powershell
# PowerShell — if script execution is blocked, use this instead of Activate.ps1:
powershell -ExecutionPolicy Bypass -File C:\App_features\venv\Scripts\Activate.ps1

# or call the venv's python directly without activating:
C:\App_features\venv\Scripts\python.exe <script>.py ...
```

```bash
# Git Bash
source "c:/App_features/venv/Scripts/activate"
```

Any time `orchestrate.py` runs, it fills missing `score` / `ratings_count` /
`installs` with MODE/MEDIAN values queried live from Athena
(`prod.app_features`). AWS credentials and region are picked up
automatically from your AWS CLI config (`~/.aws/credentials` / SSO / IAM
role) — nothing to set if `aws sts get-caller-identity` already works in
this shell. The Athena query-results staging bucket is hardcoded as
`ATHENA_S3_STAGING_DIR` in `fill_invalid_generic.py` — update it there
directly if it ever needs to change.

---

## 1. orchestrate.py — main pipeline

Validates and enriches a CSV of bundle data, producing a valid+enriched
output and an invalid-records output.

Pipeline: **load → (scrape missing fields) → resolve installs → fill
defaults → validate → enrich → save**.

### First-time run (input needs scraping)

Use this when `--input` is a fresh list of bundle_ids or partially-scraped
data, and you want Google Play hit for anything missing.

```
python orchestrate.py --input scraped_data.csv --output valid_enriched_records.csv --invalid-output invalid_records.csv
```

All three flags have defaults (`scraped_data.csv` /
`valid_enriched_records.csv` / `invalid_records.csv`), so a bare
`python orchestrate.py` also works if your file is named `scraped_data.csv`.

### Reprocessing an already-scraped invalid CSV (no re-scraping)

Use this when `--input` is a previous run's `invalid_records.csv` — the data
was already scraped once, and you just want default-filling + validation +
enrichment applied, without hitting Google Play again.

```
python orchestrate.py --input invalid_records.csv --output recovered_valid.csv --invalid-output still_invalid.csv --skip-scrape
```

`--skip-scrape` is a bare flag — just including it turns it on, there's no
value to pass (not `--skip-scrape true`, just `--skip-scrape`). Omit it
entirely to scrape.

---

## 2. run_single_bundle.py — one bundle at a time

```
python run_single_bundle.py com.kotak811mobilebankingapp.instantsavingsupiscanandpayrecharge
```

Optional flags:

```
python run_single_bundle.py <bundle_id> --from-csv existing_data.csv --output result.csv
```

- `--from-csv` — look up the bundle in this CSV first; scrapes only if not found there.
- `--output` — output CSV path (default: `single_bundle_output.csv`).

This internally calls `orchestrate.run()`, so it also needs the Athena env
vars set (see Setup above).

---

## 3. fill_invalid_generic.py — standalone default-filler

Same defaulting logic `orchestrate.py` now runs automatically, exposed as
its own script if you want to fill a CSV without running the full
validate+enrich pipeline.

```
python fill_invalid_generic.py --input invalid_records.csv --output generic_filled_records.csv
```

Note: output from this script alone is **not** in the final enriched
schema (no category/score/install/age/pricing/rating columns) — for a
main-table-ready output, use `orchestrate.py --skip-scrape` instead (see
above), which fills defaults *and* runs the enrichers.

---

## 4. analyze_invalid.py — diagnose invalid_records.csv

Reports how many bundles were dropped due to exactly 1 missing field, 2
missing fields, etc., and which columns are the culprits in each bucket.

```
python analyze_invalid.py --input invalid_records.csv
```

---

## 5. run_pipeline.py — daily S3/Athena automation

Requires the `pipeline/` package (`s3_utils.py`, `athena_utils.py`,
`clean_output.py`) which is **not present in this checkout** — this script
will fail to import until that package exists alongside it. Treat it as a
reference for the intended production flow (Athena EXCEPT query → S3 →
`orchestrate.run()` → clean/publish) rather than something runnable here.

```
python run_pipeline.py --config config.yaml
python run_pipeline.py --config config.yaml --date 20260714
```
