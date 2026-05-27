import argparse
import logging
import os
import subprocess
import sys
import time
import io
import re
import csv
import pandas as pd
from datetime import datetime, timedelta
from collections import defaultdict
from urllib.parse import urlparse

import boto3
from dotenv import load_dotenv

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

def generate_date_condition(start_date: str, end_date: str) -> str:
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    
    dates = defaultdict(list)
    curr = start
    while curr <= end:
        dates[(curr.year, curr.month)].append(curr.day)
        curr += timedelta(days=1)
        
    conditions = []
    for (year, month), days in dates.items():
        days_str = ", ".join(map(str, days))
        conditions.append(f"(year = {year} AND month = {month} AND day IN ({days_str}))")
        
    return " OR ".join(conditions)


def main():
    parser = argparse.ArgumentParser(description="Automate App Feature Scraper from Athena")
    parser.add_argument("--start-date", required=True, help="Start date in YYYY-MM-DD format")
    parser.add_argument("--end-date", required=True, help="End date in YYYY-MM-DD format")
    parser.add_argument("--limit", type=int, default=None, help="Limit the number of bundles to scrape for testing")
    args = parser.parse_args()

    # Load environment variables
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    load_dotenv(env_path)

    aws_access_key = os.getenv("AWS_ACCESS_KEY")
    aws_secret_key = os.getenv("AWS_SECRET_KEY")
    aws_region = os.getenv("AWS_REGION", "ap-southeast-1")
    athena_output = os.getenv("ATHENA_S3_OUTPUT")
    database = os.getenv("ATHENA_DATABASE", "mobavenue_dsp")

    if not all([aws_access_key, aws_secret_key, athena_output]):
        log.error("Missing required AWS credentials or S3 output location in .env")
        sys.exit(1)

    # Clean up AWS keys (remove trailing whitespace from .env loading if any)
    aws_access_key = aws_access_key.strip()
    aws_secret_key = aws_secret_key.strip()
    aws_region = aws_region.strip()
    athena_output = athena_output.strip()

    log.info("Generating date conditions for %s to %s", args.start_date, args.end_date)
    date_cond = generate_date_condition(args.start_date, args.end_date)

    query = f"""
    SELECT bundle_id FROM (
        SELECT DISTINCT LOWER(TRIM(pub_bundle)) AS bundle_id
        FROM {database}.rtb_bids
        WHERE ({date_cond})
          AND device_id IS NOT NULL
          AND device_id NOT IN ('', 'null')

        EXCEPT

        SELECT DISTINCT LOWER(TRIM(bundle_id)) AS bundle_id
        FROM imp_tables.app_feature_raw_test
    )
    """
    
    if args.limit:
        query += f"\n    LIMIT {args.limit};"
    else:
        query += ";"
    
    log.info("Connecting to AWS Athena in %s...", aws_region)
    athena_client = boto3.client(
        'athena',
        region_name=aws_region,
        aws_access_key_id=aws_access_key,
        aws_secret_access_key=aws_secret_key
    )

    log.info("Executing query:\n%s", query)
    
    # Start the Athena query
    response = athena_client.start_query_execution(
        QueryString=query,
        QueryExecutionContext={'Database': database},
        ResultConfiguration={'OutputLocation': athena_output}
    )
    
    execution_id = response['QueryExecutionId']
    log.info("Query started. Execution ID: %s", execution_id)
    
    # Poll for status
    while True:
        status_resp = athena_client.get_query_execution(QueryExecutionId=execution_id)
        state = status_resp['QueryExecution']['Status']['State']
        if state in ['SUCCEEDED', 'FAILED', 'CANCELLED']:
            break
        log.info("Query status: %s... waiting 5 seconds.", state)
        time.sleep(5)
        
    if state != 'SUCCEEDED':
        reason = status_resp['QueryExecution']['Status'].get('StateChangeReason', 'Unknown error')
        log.error("Athena query failed with state %s: %s", state, reason)
        sys.exit(1)
        
    log.info("Query succeeded. Downloading results...")
    
    # Parse S3 output location
    parsed_s3 = urlparse(athena_output)
    bucket = parsed_s3.netloc
    prefix = parsed_s3.path.lstrip('/')
    if not prefix.endswith('/') and prefix:
        prefix += '/'
        
    s3_client = boto3.client(
        's3',
        region_name=aws_region,
        aws_access_key_id=aws_access_key,
        aws_secret_access_key=aws_secret_key
    )
    
    s3_key = f"{prefix}{execution_id}.csv"
    
    input_dir = os.path.join(os.path.dirname(__file__), "input_data")
    os.makedirs(input_dir, exist_ok=True)
    csv_filename = f"athena_output_bundles_{args.start_date}_{args.end_date}.csv"
    local_csv = os.path.join(input_dir, csv_filename)
    
    try:
        s3_client.download_file(bucket, s3_key, local_csv)
        log.info("Successfully downloaded results to %s", local_csv)
        
        # Count the records (subtracting 1 for the header)
        with open(local_csv, "r", encoding="utf-8") as f:
            count = sum(1 for _ in f) - 1
        log.info("Total records (bundles) fetched: %d", max(0, count))
        
    except Exception as e:
        log.error("Failed to download CSV from S3: %s", e)
        sys.exit(1)

    log.info("Triggering orchestrate.py...")
    
    output_dir = os.path.join(os.path.dirname(__file__), "output_data")
    os.makedirs(output_dir, exist_ok=True)
    valid_csv = os.path.join(output_dir, f"valid_enriched_bundles_{args.start_date}_{args.end_date}.csv")
    invalid_csv = os.path.join(input_dir, f"invalid_bundles_{args.start_date}_{args.end_date}.csv")
    
    # Execute orchestrate.py
    cmd = [
        sys.executable,
        os.path.join(os.path.dirname(__file__), "orchestrate.py"),
        "--input", local_csv,
        "--output", valid_csv,
        "--invalid-output", invalid_csv
    ]
    
    try:
        subprocess.run(cmd, check=True)
        log.info("Pipeline completed successfully! Now filtering and cleaning the valid records...")
        
        with open(valid_csv, "r", encoding="utf-8", errors="replace") as f:
            raw = f.read()

        def fix_quoted_newlines(text):
            result = []
            in_quotes = False
            i = 0
            while i < len(text):
                char = text[i]
                if char == '"':
                    in_quotes = not in_quotes
                    result.append(char)
                elif in_quotes and char in ("\n", "\r", "\t"):
                    result.append(" ")
                else:
                    result.append(char)
                i += 1
            return "".join(result)

        fixed = fix_quoted_newlines(raw)
        
        df = pd.read_csv(io.StringIO(fixed), engine="python", on_bad_lines="skip")
        
        def clean_text(val):
            if pd.isna(val):
                return val
            val = str(val)
            val = re.sub(r"<[^>]+>", " ", val)
            val = re.sub(r"&[a-zA-Z]+;", " ", val)
            val = val.replace('""', '"')
            val = re.sub(r"[\r\n\t]+", " ", val)
            val = re.sub(r"\s+", " ", val)
            return val.strip()

        for col in df.select_dtypes(include="object").columns:
            df[col] = df[col].apply(clean_text)
            
        for col in df.columns:
            try:
                df[col] = pd.to_numeric(df[col])
            except:
                pass
                
        df.to_csv(valid_csv, index=False, encoding="utf-8-sig", quoting=csv.QUOTE_MINIMAL)
        log.info(f"Data filtering completed. Saved {len(df)} rows and {df.shape[1]} columns. Now uploading to S3...")
        
        # Upload valid_csv to mis.app_feature backing S3 location
        s3_dest_bucket = "mobavenue-simplismart-aws-s3-apse-sg"
        s3_dest_key = f"rtb/data/mis/app_feature/valid_enriched_bundles_{args.start_date}_{args.end_date}.csv"
        
        log.info(f"Uploading {valid_csv} to s3://{s3_dest_bucket}/{s3_dest_key}")
        s3_client.upload_file(valid_csv, s3_dest_bucket, s3_dest_key)
        log.info("Upload to S3 completed successfully.")
        
    except subprocess.CalledProcessError as e:
        log.error("orchestrate.py failed with return code %s", e.returncode)
        sys.exit(1)
    except Exception as e:
        log.error("Failed to upload valid records to S3: %s", e)
        sys.exit(1)

if __name__ == "__main__":
    main()
