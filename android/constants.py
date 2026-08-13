"""
android/constants.py
=====================
Defines the required/final schema columns for the Android pipeline.
"""

from android.enrichers.category_mapper import CATEGORY_COLS

# Columns that must be non-null and non-empty to keep a record
REQUIRED_COLS: list[str] = [
    "bundle_id", "app_name", "description", "summary", "genreid",
    "content_rating", "score", "ratings_count", "installs",
    "developerid", "developer", "free", "offers_iap",
    "launch_date", "days_since_released", "months_since_launch",
]

PASSTHROUGH_COLS: list[str] = REQUIRED_COLS + ["country_code", "os_type", "default"]

FINAL_COLUMNS: list[str] = PASSTHROUGH_COLS + [
    *CATEGORY_COLS,
    "score_45plus", "score_40_45", "score_30_40", "score_below30", "score_missing",
    "installs_10m_plus", "installs_1m_10m", "installs_100k_1m",
    "installs_below100k", "installs_missing",
    "apps_0_3months", "apps_3_12months", "apps_1_2years",
    "apps_2plus_years", "app_age_missing",
    "is_free_app", "is_paid_app", "offers_iap_flag", "no_iap_flag",
    "rating_everyone", "rating_teen", "rating_mature", "rating_everyone10plus",
    "rating_rated3plus", "rating_rated7plus", "rating_rated12plus",
    "rating_rated16plus", "rating_rated18plus", "rating_adults18plus",
    "rating_missing", "rating_other",
]
