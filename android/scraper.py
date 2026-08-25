"""
android/scraper.py
===================
Fetches application metadata using the Google Play Store (via google_play_scraper).
"""

import logging
import socket
from datetime import datetime, timezone
from typing import Any, Dict, Optional

# pyrefly: ignore [missing-import]
from google_play_scraper import app as gps_app
# pyrefly: ignore [missing-import]
from google_play_scraper.exceptions import NotFoundError

log = logging.getLogger(__name__)

# google_play_scraper calls urlopen() with no timeout of its own, so a stalled
# connection blocks its worker thread forever (observed 2026-08-25: 2 of 10
# ThreadPoolExecutor workers wedged on a single request for 12+ minutes,
# stalling the whole hydrate batch since nothing else was left to schedule).
# socket.setdefaulttimeout() only applies to sockets that don't set their own
# timeout, so it won't affect boto3/Athena calls, which configure theirs
# explicitly.
socket.setdefaulttimeout(30)


class ScraperError(Exception):
    pass


def _released_to_days(released_str: Optional[str]) -> Optional[int]:
    if not released_str:
        return None
    for fmt in ("%b %d, %Y", "%Y-%m-%d", "%B %d, %Y"):
        try:
            delta = (
                datetime.now(timezone.utc).replace(tzinfo=None)
                - datetime.strptime(str(released_str), fmt)
            )
            return max(0, delta.days)
        except ValueError:
            continue
    return None


def scrape_android_bundle(bundle_id: str, country: str = "in", lang: str = "en") -> Dict[str, Any]:
    """
    Query the Play Store for an Android app by its bundle ID.
    Returns a dictionary mapping the Play Store fields to our base schema.
    """
    try:
        data = gps_app(bundle_id, lang=lang, country=country)
    except NotFoundError:
        raise
    except Exception as exc:
        raise ScraperError(f"Scrape failed for {bundle_id}: {exc}") from exc

    return {
        "bundle_id": bundle_id,
        "app_name": data.get("title"),
        "description": data.get("description"),
        "summary": data.get("summary"),
        "genreid": data.get("genreId"),
        "content_rating": data.get("contentRating"),
        "score": data.get("score"),
        "ratings_count": data.get("ratings"),
        "installs": data.get("realInstalls") or data.get("installs"),
        "developerid": data.get("developerId"),
        "developer": data.get("developer"),
        "free": data.get("free"),
        "offers_iap": data.get("offersIAP"),
        "launch_date": data.get("released"),
        "days_since_released": _released_to_days(data.get("released")),
        "real_installs": data.get("realInstalls"),
        "country_code": "IND",
        "os_type": "ANDROID",
        "default": None,
    }
