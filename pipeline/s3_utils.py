"""
pipeline/s3_utils.py
=====================
Small, dependency-light S3 helpers. All functions take an explicit boto3
client so callers control credentials/region/profile in one place.
"""

from __future__ import annotations

import logging
from pathlib import Path

import boto3

log = logging.getLogger(__name__)


def make_s3_client(cfg: dict):
    session_kwargs = {}
    if cfg["aws"].get("profile"):
        session_kwargs["profile_name"] = cfg["aws"]["profile"]
    session = boto3.Session(**session_kwargs)
    return session.client("s3", region_name=cfg["aws"]["region"])


def list_keys(s3, bucket: str, prefix: str) -> list[str]:
    keys = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            keys.append(obj["Key"])
    return keys


def list_csv_keys(s3, bucket: str, prefix: str) -> list[str]:
    return [k for k in list_keys(s3, bucket, prefix) if k.lower().endswith(".csv")]


def download_file(s3, bucket: str, key: str, local_path: Path) -> Path:
    local_path.parent.mkdir(parents=True, exist_ok=True)
    log.info("Downloading s3://%s/%s -> %s", bucket, key, local_path)
    s3.download_file(bucket, key, str(local_path))
    return local_path


def upload_file(s3, local_path: Path, bucket: str, key: str) -> str:
    log.info("Uploading %s -> s3://%s/%s", local_path, bucket, key)
    s3.upload_file(str(local_path), bucket, key)
    return f"s3://{bucket}/{key}"
