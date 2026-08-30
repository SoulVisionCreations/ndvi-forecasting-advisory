"""
advisory.forecast_client — the ONLY link to the forecaster: a thin HTTP client for the
existing POST /forecast endpoint. We treat the model as a black box (no imports of the
forecasting core), which is what keeps this layer purely additive.
"""
import json
import urllib.request
from datetime import datetime

from . import config as C


def forecast(lat, lon, date, base_url=None, timeout=200, debug=True):
    """POST one (lat,lon,date) to /forecast and return the parsed response.
    date = the as-of / forecast-from date (YYYY-MM-DD); the target is ~3 months ahead."""
    url = (base_url or C.FORECAST_URL) + ("?debug=true" if debug else "")
    payload = json.dumps({"lat": lat, "lon": lon, "date": date}).encode()
    req = urllib.request.Request(url, data=payload,
                                 headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=timeout))


def extract(resp):
    """Pull the fields the advisory layer needs out of a /forecast response, tolerant
    of shape (fields live at top level or under 'debug')."""
    def dig(*keys, default=None):
        for k in keys:
            if isinstance(resp, dict) and k in resp:
                return resp[k]
        dbg = resp.get("debug", {}) if isinstance(resp, dict) else {}
        for k in keys:
            if k in dbg:
                return dbg[k]
        return default

    anomaly = resp.get("anomaly", {}) if isinstance(resp, dict) else {}
    z = anomaly.get("z", resp.get("z"))
    target_date = dig("target_date")
    target_month = None
    if target_date:
        try:
            target_month = datetime.strptime(target_date[:10], "%Y-%m-%d").month
        except ValueError:
            target_month = None
    return {
        "z": z,
        "forecast_ndvi": dig("forecast_ndvi"),
        "baseline_mean": dig("baseline_mean") or anomaly.get("baseline_mean"),
        "baseline_std": dig("baseline_std") or anomaly.get("baseline_std"),
        "target_date": target_date,
        "target_month": target_month,
        # staleness = how many fortnights the anchor NDVI was ffilled; a data-quality clue
        "stale_fortnights": dig("stale_fortnights", "anchor_stale_fortnights"),
        "admin": resp.get("admin", {}) if isinstance(resp, dict) else {},
    }
