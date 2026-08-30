"""
ndvi_core.download — shared HLS quarterly-composite recipe.

Extracted verbatim (behavior-preserving) from scripts/download_quarterly_composite.py
so the training tile-build and the inference live-fetch run the SAME compositing
code (no duplication). Produces a (6, 224, 224) int16 reflectance×10000 tile for a
point, identical to the tiles Prithvi was fine-tuned on.

Inference usage: pick the walk-back (year, quarter) for a target date, fetch the
composite for (lat, lon) into a local cache dir, then encode it with the same
(mean, std) the training TileStore uses.
"""

import os
import threading
import time

import ee
import requests

HIGH_VOLUME = "https://earthengine-highvolume.googleapis.com"
LANDSAT_BANDS = ["B2", "B3", "B4", "B5", "B6", "B7"]
SENTINEL_BANDS = ["B2", "B3", "B4", "B8A", "B11", "B12"]
COMMON_BANDS = ["BLUE", "GREEN", "RED", "NIR", "SWIR1", "SWIR2"]
REFLECTANCE_SCALE = 10000

QUARTERS = ("Q1", "Q2", "Q3", "Q4")
QUARTER_DOY = {"Q1": 46, "Q2": 136, "Q3": 227, "Q4": 319}   # mid-quarter doy (encode-time)


# --- rate-limit (throttle) detection -----------------------------------------
# The advisory fires up to ADVISORY_FETCH_WORKERS concurrent getDownloadURL pulls
# (one composite tile per sampled point). Earth Engine can throttle such bursts.
# We DON'T fail on it — fetch_composite already retries with exponential backoff —
# but we RECORD each throttle, keyed by the request's cache dir (a per-advisory uuid
# dir), so the serving layer can surface a warning telling the operator to lower the
# concurrency. Request-scoped keying => concurrent requests never see each other's.
_RATE_LIMIT_MARKERS = (
    "429", "too many requests", "rate limit", "user rate limit",
    "quota", "resource exhausted", "rate_limit_exceeded")
_throttle_lock = threading.Lock()
_throttle_events = {}          # out_dir -> [short detail strings]


def is_rate_limit(msg):
    """True if an error string looks like an upstream rate-limit / quota throttle."""
    m = str(msg).lower()
    return any(k in m for k in _RATE_LIMIT_MARKERS)


def _note_throttle(out_dir, detail):
    """Record one throttle occurrence for this request's dir (no-op if out_dir falsy)."""
    if not out_dir:
        return
    with _throttle_lock:
        _throttle_events.setdefault(out_dir, []).append(str(detail)[:120])


def pop_throttle_events(out_dir):
    """Return AND clear the throttle events recorded for this request's dir."""
    with _throttle_lock:
        return _throttle_events.pop(out_dir, [])


def quarter_window(year, q):
    return {
        "Q1": (f"{year}-01-01", f"{year}-04-01"),
        "Q2": (f"{year}-04-01", f"{year}-07-01"),
        "Q3": (f"{year}-07-01", f"{year}-10-01"),
        "Q4": (f"{year}-10-01", f"{year + 1}-01-01"),
    }[q]


def season_of_month(month):
    """Quarter label for a calendar month — matches prithvi_embed._season_of
    (quarterly): Jan-Mar=Q1 / Apr-Jun=Q2 / Jul-Sep=Q3 / Oct-Dec=Q4."""
    return "Q" + str((int(month) - 1) // 3 + 1)


# --- masking / harmonization (identical to the training downloader) ------------
def mask_clouds_hls(image):
    fmask = image.select("Fmask")
    mask = (fmask.bitwiseAnd(1 << 1).eq(0)
            .And(fmask.bitwiseAnd(1 << 2).eq(0))
            .And(fmask.bitwiseAnd(1 << 3).eq(0)))
    return image.updateMask(mask)


def rename_landsat(image):
    return image.select(LANDSAT_BANDS + ["Fmask"], COMMON_BANDS + ["Fmask"])


def rename_sentinel(image):
    return image.select(SENTINEL_BANDS + ["Fmask"], COMMON_BANDS + ["Fmask"])


def utm_epsg(lat, lon):
    zone = int((lon + 180) / 6) + 1
    return f"EPSG:{(32600 if lat >= 0 else 32700) + zone}"


def hls_composite(region, start_date, end_date):
    landsat = (ee.ImageCollection("NASA/HLS/HLSL30/v002")
               .filterBounds(region).filterDate(start_date, end_date)
               .map(rename_landsat).map(mask_clouds_hls))
    sentinel = (ee.ImageCollection("NASA/HLS/HLSS30/v002")
                .filterBounds(region).filterDate(start_date, end_date)
                .map(rename_sentinel).map(mask_clouds_hls))
    return landsat.merge(sentinel).select(COMMON_BANDS).median()


def loc_id(lat, lon):
    """Location-keyed tile id for inference: `<lat>_<lon>` (4dp). Used as the tile
    `mws_id` at serve time so the cached filename (composite_<id>_<year>_<Q>.tif)
    encodes the POINT, not a fixed 'req'. Without this, two different locations that
    need the same (year, quarter) collide on one filename and skip-exists reuse would
    hand one point another's tile in a shared cache dir. The dotted form parses back
    cleanly via the folder regex (mws = everything before _<year>[_<Q>])."""
    return f"{lat:.4f}_{lon:.4f}"


def fetch_composite(mws_id, lat, lon, year, quarter, out_dir,
                    patch_radius_m=3360, patch_px=224, retries=5):
    """Download one (key, year, quarter) composite to out_dir.

    Returns (out_path, valid_frac) where valid_frac = (NIR>0).mean() — the same
    coverage metric the training downloader uses to drop gappy tiles. On a cache
    hit returns (out_path, None). `mws_id` only names the file
    (composite_<id>_<year>_<Q>.tif) so the cached tile is consumable by the
    standard TileStore. Raises RuntimeError on persistent failure."""
    file_name = f"composite_{mws_id}_{year}_{quarter}"
    out_path = os.path.join(out_dir, file_name + ".tif")
    if os.path.exists(out_path):
        return out_path, None

    start_date, end_date = quarter_window(year, quarter)
    region = ee.Geometry.Point([lon, lat]).buffer(patch_radius_m).bounds()
    epsg = utm_epsg(lat, lon)
    img = (hls_composite(region, start_date, end_date)
           .multiply(REFLECTANCE_SCALE).toInt16().unmask(0))
    params = {"region": region, "crs": epsg,
              "dimensions": f"{patch_px}x{patch_px}",
              "format": "GEO_TIFF", "bands": COMMON_BANDS}

    last_err = None
    for attempt in range(retries):
        try:
            url = img.getDownloadURL(params)
            resp = requests.get(url, timeout=300)
            if resp.status_code == 200:
                data = resp.content
                from rasterio.io import MemoryFile
                with MemoryFile(data) as mem, mem.open() as src:
                    arr = src.read()
                valid = float((arr[COMMON_BANDS.index("NIR")] > 0).mean())
                os.makedirs(out_dir, exist_ok=True)
                tmp = out_path + ".tmp"
                with open(tmp, "wb") as f:
                    f.write(data)
                os.replace(tmp, out_path)
                return out_path, valid
            last_err = f"HTTP {resp.status_code}"
            if resp.status_code == 429:                 # throttled — record (still retried below)
                _note_throttle(out_dir, f"HTTP 429 on {file_name}")
        except Exception as e:
            last_err = str(e)[:160]
            if is_rate_limit(last_err):                 # rate-limit raised at getDownloadURL
                _note_throttle(out_dir, last_err)
        time.sleep(2 ** attempt)
    raise RuntimeError(f"composite fetch failed for {file_name}: {last_err}")


def fetch_walkback_tile(mws_id, lat, lon, target_date, out_dir, lag=1,
                        min_coverage=0.75, min_year=2016,
                        patch_radius_m=3360, patch_px=224, retries=5):
    """Inference tile selection — matches training prithvi_embed.resolve_row
    (mode='quarterly'): quarter = the TARGET month's quarter; start at
    year(target)-lag and walk BACK year-by-year (SAME quarter) until a composite
    with valid-pixel fraction >= min_coverage is found.

    Returns (tile_path, mws_id, year, quarter). lag=1 guarantees we never read the
    same-year same-quarter tile (overlap = leak), identical to training; the
    walk-back-on-low-coverage mirrors training's walk-back over <min_coverage tiles.
    Raises RuntimeError if no sufficient tile down to min_year."""
    import pandas as pd
    ts = pd.Timestamp(target_date)
    quarter = season_of_month(ts.month)
    start = ts.year - lag
    last_err = None
    for year in range(int(start), int(min_year) - 1, -1):
        try:
            path, valid = fetch_composite(mws_id, lat, lon, year, quarter, out_dir,
                                          patch_radius_m, patch_px, retries)
        except RuntimeError as e:
            last_err = str(e)
            continue
        if valid is None or valid >= min_coverage:   # None = cache hit (already vetted)
            return path, mws_id, year, quarter
    raise RuntimeError(
        f"no quarterly tile >= {min_coverage} coverage for {mws_id} {quarter} "
        f"in years [{min_year}, {start}]: {last_err}")


def actual_ndvi(lat, lon, date_str):
    """Actual MODIS NDVI at (point, date): MOD13Q1/061 16-day median, x1e-4.
    Shared by the batch validation scripts (was duplicated in batch_metrics +
    batch_sanity). Returns None when no valid composite is present at the date."""
    pt = ee.Geometry.Point([lon, lat])
    s = ee.Date(date_str); e = s.advance(16, "day")
    col = (ee.ImageCollection("MODIS/061/MOD13Q1")
           .filterDate(s, e).filterBounds(pt).select("NDVI"))
    img = ee.Image(ee.Algorithms.If(col.size().gt(0), col.median(),
                                    ee.Image.constant(-9999).rename("NDVI")))
    v = img.reduceRegion(ee.Reducer.mean(), pt, 250, bestEffort=True).get("NDVI").getInfo()
    if v is None:
        return None
    v = float(v) * 0.0001
    return None if v <= -0.999 else v
