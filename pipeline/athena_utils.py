"""
pipeline/athena_utils.py
=========================
Runs a SQL query against Athena and returns the result as a pandas
DataFrame. Uses plain boto3 (no extra deps like awswrangler) so it works
anywhere boto3 already works.
"""

from __future__ import annotations

import logging
import time
from io import StringIO

import boto3
import pandas as pd

log = logging.getLogger(__name__)

_POLL_SECONDS = 2
_TERMINAL_STATES = {"SUCCEEDED", "FAILED", "CANCELLED"}


def make_athena_client(cfg: dict):
    session_kwargs = {}
    if cfg["aws"].get("profile"):
        session_kwargs["profile_name"] = cfg["aws"]["profile"]
    session = boto3.Session(**session_kwargs)
    return session.client("athena", region_name=cfg["aws"]["region"])


def run_query_to_dataframe(cfg: dict, sql: str) -> pd.DataFrame:
    """
    Execute `sql` in Athena, block until it finishes, and return the result
    set as a DataFrame. Raises RuntimeError on FAILED/CANCELLED.
    """
    athena = make_athena_client(cfg)
    a_cfg = cfg["athena"]

    log.info("Submitting Athena query (workgroup=%s, db=%s) …", a_cfg["workgroup"], a_cfg["database"])
    log.debug("SQL:\n%s", sql)

    start = athena.start_query_execution(
        QueryString=sql,
        QueryExecutionContext={"Database": a_cfg["database"]},
        ResultConfiguration={"OutputLocation": a_cfg["output_location"]},
        WorkGroup=a_cfg["workgroup"],
    )
    query_id = start["QueryExecutionId"]
    log.info("Athena QueryExecutionId: %s", query_id)

    while True:
        resp = athena.get_query_execution(QueryExecutionId=query_id)
        state = resp["QueryExecution"]["Status"]["State"]
        if state in _TERMINAL_STATES:
            break
        time.sleep(_POLL_SECONDS)

    if state != "SUCCEEDED":
        reason = resp["QueryExecution"]["Status"].get("StateChangeReason", "unknown")
        raise RuntimeError(f"Athena query {query_id} ended in state {state}: {reason}")

    result_s3_path = resp["QueryExecution"]["ResultConfiguration"]["OutputLocation"]
    log.info("Athena query succeeded. Result CSV: %s", result_s3_path)

    # Result lands at exactly `output_location + query_id + ".csv"`. Read it
    # straight from S3 with pandas (pandas can read s3:// URIs given the
    # s3fs package; fall back to boto3 get_object if that's not installed).
    try:
        return pd.read_csv(result_s3_path)
    except Exception:
        bucket, key = result_s3_path.replace("s3://", "", 1).split("/", 1)
        s3 = boto3.Session(
            profile_name=cfg["aws"].get("profile")
        ).client("s3", region_name=cfg["aws"]["region"]) if cfg["aws"].get("profile") else boto3.client(
            "s3", region_name=cfg["aws"]["region"]
        )
        obj = s3.get_object(Bucket=bucket, Key=key)
        return pd.read_csv(StringIO(obj["Body"].read().decode("utf-8")))
