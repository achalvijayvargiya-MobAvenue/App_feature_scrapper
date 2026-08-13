"""
ios/scraper.py
==============
Fetches application metadata using Apple's iTunes Lookup API.
"""

import json
import logging
import time
import urllib.request
import urllib.error
from datetime import datetime
from typing import Dict, Any, Optional

log = logging.getLogger(__name__)

class NotFoundError(Exception):
    pass

class ScraperError(Exception):
    pass

def _fetch_itunes_data(app_id: str, country: str = "in", retries: int = 3) -> Dict[str, Any]:
    """
    Fetch data from iTunes API with simple retry logic.
    """
    if app_id.isdigit():
        url = f"https://itunes.apple.com/lookup?id={app_id}&country={country}"
    else:
        url = f"https://itunes.apple.com/lookup?bundleId={app_id}&country={country}"
        
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(
                url, 
                headers={'User-Agent': 'Mozilla/5.0 (iPhone; CPU iPhone OS 16_0 like Mac OS X)'}
            )
            with urllib.request.urlopen(req, timeout=10) as response:
                data = json.loads(response.read().decode('utf-8'))
                if data.get("resultCount", 0) == 0:
                    raise NotFoundError(f"No results found for app {app_id} in country '{country}'.")
                return data["results"][0]
                
        except urllib.error.HTTPError as exc:
            if exc.code in (429, 500, 502, 503, 504) and attempt < retries:
                time.sleep(2 ** attempt)
                continue
            raise ScraperError(f"HTTP Error {exc.code} for {app_id}: {exc.reason}") from exc
        except urllib.error.URLError as exc:
            if attempt < retries:
                time.sleep(2 ** attempt)
                continue
            raise ScraperError(f"Network Error for {app_id}: {exc.reason}") from exc
        except NotFoundError:
            raise
        except Exception as exc:
            if attempt < retries:
                time.sleep(2 ** attempt)
                continue
            raise ScraperError(f"Unexpected error for {app_id}: {exc}") from exc
            
    raise ScraperError(f"Max retries exceeded for {app_id}")

def _normalize_launch_date(release_date: Optional[str]) -> Optional[str]:
    """
    Apple returns releaseDate as ISO 8601, e.g. "2010-04-01T21:02:20Z".
    Normalize to "%b %d, %Y" (e.g. "Apr 01, 2010") to match the format
    google_play_scraper's 'released' field already uses for Android, so
    launch_date is consistent across both platforms.
    """
    if not release_date:
        return None
    try:
        dt = datetime.strptime(str(release_date)[:10], "%Y-%m-%d")
        return dt.strftime("%b %d, %Y")
    except ValueError:
        return release_date

def scrape_ios_bundle(app_id: str, country: str = "in") -> Dict[str, Any]:
    """
    Query the iTunes API for an iOS app by its numeric ID or bundle ID.
    Returns a dictionary mapping the Apple fields to our base schema.
    """
    app_info = _fetch_itunes_data(app_id, country)

    description = app_info.get("description", "")

    price = app_info.get("price")
    is_free = (price == 0.0) if price is not None else True

    # iTunes API doesn't return IAP reliably in this endpoint, defaulting to None
    offers_iap = None

    return {
        "bundle_id": app_id,
        "app_name": app_info.get("trackName"),
        "description": description,
        # iTunes API has no subtitle/summary field on this endpoint. "Not Available" is
        # used instead of "NA"/"N/A" because pandas treats those as null on CSV re-read.
        "summary": "Not Available",
        "genreid": app_info.get("primaryGenreName"),
        "content_rating": app_info.get("trackContentRating") or app_info.get("contentAdvisoryRating"),
        "score": app_info.get("averageUserRating"),
        "ratings_count": app_info.get("userRatingCount"),
        "installs": None, # Apple does not provide install counts
        "developerid": str(app_info.get("artistId", "")),
        "developer": app_info.get("artistName"),
        "free": is_free,
        "offers_iap": offers_iap,
        "launch_date": _normalize_launch_date(app_info.get("releaseDate")),
        "real_installs": None,
        "country_code": "IND",
        "os_type": "IOS",
        "default": None
    }
