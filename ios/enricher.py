"""
ios/enricher.py
===============
Applies feature engineering for iOS apps (age binning, score binning, content rating, IAP, category mapping).
"""

import pandas as pd
from datetime import datetime, timezone
from ios.categories import map_categories
from ios.constants import FINAL_COLUMNS

def enrich_age(df: pd.DataFrame) -> pd.DataFrame:
    df["days_since_released"] = pd.to_numeric(df["days_since_released"], errors="coerce")
    df["months_since_launch"] = df["days_since_released"] / 30.0

    df["apps_0_3months"] = (df["months_since_launch"] <= 3).astype(int)
    df["apps_3_12months"] = ((df["months_since_launch"] > 3) & (df["months_since_launch"] <= 12)).astype(int)
    df["apps_1_2years"] = ((df["months_since_launch"] > 12) & (df["months_since_launch"] <= 24)).astype(int)
    df["apps_2plus_years"] = (df["months_since_launch"] > 24).astype(int)
    df["app_age_missing"] = df["days_since_released"].isna().astype(int)
    
    # If missing, zero out the other flags
    mask = df["app_age_missing"] == 1
    df.loc[mask, ["apps_0_3months", "apps_3_12months", "apps_1_2years", "apps_2plus_years"]] = 0
    return df

def enrich_score(df: pd.DataFrame) -> pd.DataFrame:
    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    df["score_45plus"] = (df["score"] >= 4.5).astype(int)
    df["score_40_45"] = ((df["score"] >= 4.0) & (df["score"] < 4.5)).astype(int)
    df["score_30_40"] = ((df["score"] >= 3.0) & (df["score"] < 4.0)).astype(int)
    df["score_below30"] = (df["score"] < 3.0).astype(int)
    df["score_missing"] = df["score"].isna().astype(int)
    return df

def enrich_content_rating(df: pd.DataFrame) -> pd.DataFrame:
    # Initialize all to 0
    rating_flags = ["rating_everyone", "rating_teen", "rating_mature", "rating_everyone10plus", 
                    "rating_rated3plus", "rating_rated7plus", "rating_rated12plus", 
                    "rating_rated16plus", "rating_rated18plus", "rating_adults18plus", 
                    "rating_missing", "rating_other"]
    for col in rating_flags:
        df[col] = 0

    df["content_rating"] = df["content_rating"].astype(str).str.strip().str.lower()
    
    for idx, row in df.iterrows():
        val = row["content_rating"]
        if pd.isna(val) or val in ("nan", "none", ""):
            df.at[idx, "rating_missing"] = 1
        elif "4+" in val:
            df.at[idx, "rating_rated3plus"] = 1 # Mapping 4+ to 3+
        elif "9+" in val:
            df.at[idx, "rating_rated7plus"] = 1 # Mapping 9+ to 7+ (approximate match to Android)
        elif "12+" in val:
            df.at[idx, "rating_rated12plus"] = 1
        elif "17+" in val:
            df.at[idx, "rating_rated16plus"] = 1
            df.at[idx, "rating_mature"] = 1
        elif "18+" in val:
            df.at[idx, "rating_rated18plus"] = 1
            df.at[idx, "rating_adults18plus"] = 1
        else:
            df.at[idx, "rating_other"] = 1
            
    return df

def enrich_pricing_iap(df: pd.DataFrame) -> pd.DataFrame:
    df["is_free_app"] = df["free"].apply(lambda x: 1 if x is True or str(x).lower() == "true" else 0)
    df["is_paid_app"] = df["free"].apply(lambda x: 1 if x is False or str(x).lower() == "false" else 0)
    
    # Missing IAP data for iOS by default
    df["offers_iap_flag"] = df["offers_iap"].apply(lambda x: 1 if x is True or str(x).lower() == "true" else 0)
    df["no_iap_flag"] = df["offers_iap"].apply(lambda x: 1 if x is False or str(x).lower() == "false" else (1 if pd.isna(x) else 0))
    return df

def enrich_installs(df: pd.DataFrame) -> pd.DataFrame:
    # Apple doesn't expose installs, set missing to 1
    for col in ["installs_10m_plus", "installs_1m_10m", "installs_100k_1m", "installs_below100k"]:
        df[col] = 0
    df["installs_missing"] = 1
    return df

def enrich_ios_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    # Compute days_since_released if launch_date is present
    if "days_since_released" not in df.columns:
        df["days_since_released"] = None
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        for idx, row in df.iterrows():
            launch = row.get("launch_date")
            if pd.notna(launch) and launch:
                try:
                    # Apple usually returns something like "2010-09-17T00:00:00Z"
                    launch_dt = datetime.strptime(str(launch)[:10], "%Y-%m-%d")
                    df.at[idx, "days_since_released"] = max(0, (now - launch_dt).days)
                except Exception:
                    pass

    df = enrich_age(df)
    df = enrich_score(df)
    df = enrich_content_rating(df)
    df = enrich_pricing_iap(df)
    df = enrich_installs(df)
    df = map_categories(df)
    
    # Ensure default columns exist and match schema
    for col in FINAL_COLUMNS:
        if col not in df.columns:
            df[col] = ""
            
    return df[FINAL_COLUMNS]
