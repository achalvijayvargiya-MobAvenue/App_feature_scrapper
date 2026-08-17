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

def _fetch_itunes_batch_data(app_ids: list[str], country: str = "in", retries: int = 3) -> list[Dict[str, Any]]:
    """
    Fetch data from iTunes API for multiple apps with simple retry logic.
    Apple allows up to 200 IDs per request.
    """
    if not app_ids:
        return []
        
    numeric_ids = [aid for aid in app_ids if str(aid).isdigit()]
    bundle_ids = [aid for aid in app_ids if not str(aid).isdigit()]
    
    results = []
    
    # Process numeric IDs
    if numeric_ids:
        id_str = ",".join(numeric_ids)
        url = f"https://itunes.apple.com/lookup?id={id_str}&country={country}"
        results.extend(_do_fetch(url, retries, id_str))
        
    # Process string bundle IDs
    if bundle_ids:
        bundle_str = ",".join(bundle_ids)
        url = f"https://itunes.apple.com/lookup?bundleId={bundle_str}&country={country}"
        results.extend(_do_fetch(url, retries, bundle_str))
        
    return results

def _do_fetch(url: str, retries: int, context_id: str) -> list[Dict[str, Any]]:
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(
                url, 
                headers={'User-Agent': 'Mozilla/5.0 (iPhone; CPU iPhone OS 16_0 like Mac OS X)'}
            )
            with urllib.request.urlopen(req, timeout=10) as response:
                data = json.loads(response.read().decode('utf-8'))
                return data.get("results", [])
                
        except urllib.error.HTTPError as exc:
            if exc.code in (429, 500, 502, 503, 504) and attempt < retries:
                time.sleep(2 ** attempt)
                continue
            raise ScraperError(f"HTTP Error {exc.code} for {context_id}: {exc.reason}") from exc
        except urllib.error.URLError as exc:
            if attempt < retries:
                time.sleep(2 ** attempt)
                continue
            raise ScraperError(f"Network Error for {context_id}: {exc.reason}") from exc
        except Exception as exc:
            if attempt < retries:
                time.sleep(2 ** attempt)
                continue
            raise ScraperError(f"Unexpected error for {context_id}: {exc}") from exc
            
    raise ScraperError(f"Max retries exceeded for {context_id}")

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

def scrape_ios_bundles(app_ids: list[str], country: str = "in") -> list[Dict[str, Any]]:
    """
    Query the iTunes API for a list of iOS apps by numeric ID or bundle ID.
    Returns a list of dictionaries mapping the Apple fields to our base schema.
    """
    app_info_list = _fetch_itunes_batch_data(app_ids, country)
    
    records = []
    for app_info in app_info_list:
        description = app_info.get("description", "")
        price = app_info.get("price")
        is_free = (price == 0.0) if price is not None else True
        offers_iap = None

        raw_genres = app_info.get("genres") or []
        sorted_genres = sorted(raw_genres)
        formatted_genres = "_".join(sorted_genres) if sorted_genres else None

        # Determine the original ID requested, iTunes returns bundleId and trackId
        # We try trackId first as a string, then bundleId, to match original request
        original_id = str(app_info.get("trackId", "")) or app_info.get("bundleId", "")

        records.append({
            "bundle_id": original_id,
            "app_name": app_info.get("trackName"),
            "description": description,
            "summary": "Not Available",
            "genreid": formatted_genres,
            "content_rating": app_info.get("trackContentRating") or app_info.get("contentAdvisoryRating"),
            "score": app_info.get("averageUserRating"),
            "ratings_count": app_info.get("userRatingCount"),
            "installs": None,
            "developerid": str(app_info.get("artistId", "")),
            "developer": app_info.get("artistName"),
            "free": is_free,
            "offers_iap": offers_iap,
            "launch_date": _normalize_launch_date(app_info.get("releaseDate")),
            "real_installs": None,
            "country_code": "IND",
            "os_type": "IOS",
            "default": None
        })
    return records

def scrape_ios_bundle(app_id: str, country: str = "in") -> Dict[str, Any]:
    """
    Backward-compatible wrapper for single app.
    """
    res = scrape_ios_bundles([app_id], country)
    if not res:
        raise NotFoundError(f"No results found for app {app_id} in country '{country}'.")
    # For single lookup, always force the bundle_id to be what was requested
    res[0]["bundle_id"] = app_id
    return res[0]
