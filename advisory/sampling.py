"""
advisory.sampling — Steps 1-2: locate the farmer's VILLAGE and sample >= MIN_POINTS
valid cropland points around the plot, using the open CoreStack layers.

Pinned rule (use_case_ndvi.txt): draw cropland cells from within the village polygon;
if fewer than MIN_POINTS, buffer outward (up to BUFFER_KM_MAX) to reach it; still short
-> flag a small, low-confidence sample. Cropland sampling is pure stdlib (urllib); the WorldCover
vegetated FALLBACK (Tier 3) lazily imports ee (Earth Engine, already initialized by the engine).
"""
import json
import math
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request

from . import config as C


class OutOfCoverage(Exception):
    """Raised when neither cropland NOR vegetated land is available within range of the query point."""
    def __init__(self, state, district, tehsil, nearest_km=None):
        self.state, self.district, self.tehsil = state, district, tehsil
        self.nearest_km = nearest_km
        super().__init__(f"no cropland within range of {district}/{tehsil}")


def _getj(url, headers=None, timeout=60, retry_budget=90):
    # CoreStack sits behind a flaky shared proxy that intermittently returns
    # "Tunnel connection failed: 500" for seconds-to-tens-of-seconds at a time.
    # Retry until a total TIME BUDGET is spent (not a fixed count), so a longer
    # blip is ridden out; a healthy call still returns on the first try (~0.5s).
    # A 4xx (e.g. an unmapped WFS layer -> 400) is PERMANENT, so fail FAST — never
    # burn the retry budget on it (that used to hang out-of-coverage requests ~90s).
    last = None
    start = time.monotonic()
    attempt = 0
    while True:
        attempt += 1
        try:
            req = urllib.request.Request(url, headers=headers or {})
            return json.load(urllib.request.urlopen(req, timeout=timeout))
        except urllib.error.HTTPError as e:
            if 400 <= e.code < 500 and e.code != 429:
                raise                                    # permanent client error -> fail fast
            last = e
        except Exception as e:
            last = e
        if time.monotonic() - start >= retry_budget:
            break
        time.sleep(min(1.0 + 1.5 * attempt, 8))
    raise last if last else RuntimeError(f"request failed: {url}")


def _admin(lat, lon, key):
    url = C.CORESTACK_ADMIN + "?" + urllib.parse.urlencode(
        {"latitude": lat, "longitude": lon})
    r = _getj(url, {"X-API-Key": key})
    return (r.get("State", "").strip(), r.get("District", "").strip(),
            r.get("Tehsil", "").strip())


def _wfs(layer, count=4000):
    q = urllib.parse.urlencode({
        "service": "WFS", "version": "2.0.0", "request": "GetFeature",
        "outputFormat": "application/json", "srsName": "EPSG:4326",
        "typeNames": layer, "count": count})
    return _getj(C.CORESTACK_WFS + "?" + q).get("features", [])


def _slug(s):
    return s.strip().lower().replace(" ", "_")


def _rings(g):
    return [g["coordinates"][0]] if g["type"] == "Polygon" else [p[0] for p in g["coordinates"]]


def _centroid(g):
    """Return (lon, lat) of the first ring's centroid."""
    r = _rings(g)[0]
    xs = [p[0] for p in r[:-1]]
    ys = [p[1] for p in r[:-1]]
    return (sum(xs) / len(xs), sum(ys) / len(ys))


def _point_in_poly(pt, g):
    """pt = (lon, lat)."""
    x, y = pt
    inside = False
    for r in _rings(g):
        n = len(r)
        j = n - 1
        for i in range(n):
            xi, yi = r[i]
            xj, yj = r[j]
            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / ((yj - yi) or 1e-12) + xi):
                inside = not inside
            j = i
    return inside


def _km(a_latlon, b_latlon):
    (la1, lo1), (la2, lo2) = a_latlon, b_latlon
    R = 6371.0
    p1, p2 = math.radians(la1), math.radians(la2)
    dphi = math.radians(la2 - la1)
    dlmb = math.radians(lo2 - lo1)
    h = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * R * math.asin(math.sqrt(h))


# --- crop-tile catalog: which district_tehsil grids exist + their centroids ----------
# The cropland grids are per-tehsil WFS tiles (~464 across ~15 states). When the plot's
# OWN tehsil tile is missing we fall back to the nearest AVAILABLE tile, so build a small
# catalog {district_tehsil: (lat, lon)} from GeoServer capabilities (cached on disk).
_MAX_PROXY_KM = 50.0            # a proxy tile's cells must be within this of the plot
_CATALOG = None
_CATALOG_TTL = 7 * 24 * 3600


def _catalog_path():
    base = os.environ.get("CACHE_DIR") or "/tmp"
    return os.path.join(base, "crop_tile_catalog.json")


def _fetch_tile_catalog():
    caps = C.CORESTACK_WFS + "?service=WFS&version=2.0.0&request=GetCapabilities"
    xml = urllib.request.urlopen(urllib.request.Request(caps), timeout=180).read().decode(
        "utf-8", "ignore")
    out = []
    for blk in xml.split("<FeatureType"):
        m = re.search(r"crop_grid_layers:([a-z_]+)_grid", blk)
        if not m:
            continue
        lc = re.search(r"LowerCorner>([-\d.]+)\s+([-\d.]+)</", blk)
        uc = re.search(r"UpperCorner>([-\d.]+)\s+([-\d.]+)</", blk)
        if not (lc and uc):
            continue
        lon = (float(lc.group(1)) + float(uc.group(1))) / 2
        lat = (float(lc.group(2)) + float(uc.group(2))) / 2
        out.append([m.group(1), lat, lon])
    return out


def _load_tile_catalog():
    global _CATALOG
    if _CATALOG is not None:
        return _CATALOG
    p = _catalog_path()
    try:
        if time.time() - os.stat(p).st_mtime < _CATALOG_TTL:
            _CATALOG = json.load(open(p))
            return _CATALOG
    except Exception:
        pass
    cat = _fetch_tile_catalog()
    try:
        os.makedirs(os.path.dirname(p), exist_ok=True)
        json.dump(cat, open(p, "w"))
    except Exception:
        pass
    _CATALOG = cat
    return _CATALOG


def _cropland_cells(dt):
    """[( (clat,clon), frac ), ...] for cropland cells (frac >= threshold) of this tile;
    [] if the tile is not mapped (WFS 400) so the caller can fall back."""
    try:
        grid = _wfs(f"crop_grid_layers:{dt}_grid")
    except urllib.error.HTTPError as e:
        if 400 <= e.code < 500:
            return []
        raise
    cells = []
    for f in grid:
        try:
            frac = float(f["properties"].get("fraction", 0))
        except (TypeError, ValueError):
            frac = 0.0
        if frac < C.SAMPLE_FRACTION:
            continue
        clon, clat = _centroid(f["geometry"])
        cells.append(((clat, clon), frac))
    return cells


def _worldcover_vegetated_cells(lat, lon, radius_km, min_points):
    """FALLBACK candidate-gen: sample nearby VEGETATED land via ESA WorldCover (GEE).
    Returns [(dist_km, (clat, clon)), ...] sorted nearest-first; [] if GEE is unavailable or no
    vegetated pixel is found. The z-score is SELF-relative per point, so a non-cropland vegetated
    point still gives a valid 'vs its own normal' read — but the caller FLAGS it (veg_fallback) and
    CAPS its confidence (the forecaster is cropland-trained, so a non-cropland read is out-of-domain)."""
    try:
        import ee                                            # available once the engine has ee.Initialize()d
    except Exception:
        return []
    try:
        pt = ee.Geometry.Point([lon, lat])
        region = pt.buffer(radius_km * 1000.0).bounds()
        wc = ee.ImageCollection(C.WORLDCOVER_ASSET).first()  # one static global 10 m land-cover image
        classes = list(C.WORLDCOVER_VEG_CLASSES)
        veg = wc.remap(classes, [1] * len(classes), 0).selfMask()   # 1 on vegetated pixels, masked elsewhere
        fc = veg.sample(region=region, scale=10, numPixels=C.VEG_SAMPLE_PIXELS,
                        geometries=True, seed=1)             # only vegetated pixels survive the mask
        feats = fc.getInfo().get("features", [])
    except Exception:
        return []
    cells = []
    for f in feats:
        try:
            clon, clat = f["geometry"]["coordinates"]
        except (KeyError, TypeError, ValueError):
            continue
        cells.append((_km((lat, lon), (clat, clon)), (clat, clon)))
    cells.sort(key=lambda c: c[0])
    return cells


def locate_and_sample(lat, lon, key=None, min_points=None, buffer_km=None, max_proxy_km=None):
    """Return (meta, points) where meta = {state,district,tehsil,village,...,proxy?,veg_fallback?}
    and points = [(lat,lon), ...]. Tiered so it does NOT hard-fail:
      1) the plot's OWN tehsil CROPLAND tile (village-preferred, small buffer);
      2) else the NEAREST reference points, cropland-preferred: nearby cropland (<= max_proxy_km) and
         local WorldCover vegetation (<= VEG_SEARCH_KM) are MERGED and ranked by distance, with veg
         handicapped (VEG_DIST_PENALTY_KM) so near cropland wins and local vegetation only beats FAR
         cropland. Any veg point in the chosen set -> flagged veg_fallback (caller caps confidence);
         all-cropland -> flagged proxy;
      3) nothing within range -> raise OutOfCoverage (caller returns a graceful message)."""
    key = key if key is not None else C.CORESTACK_KEY
    min_points = min_points or C.MIN_POINTS
    buffer_km = buffer_km or C.BUFFER_KM_MAX
    max_proxy_km = max_proxy_km or _MAX_PROXY_KM

    state, district, tehsil = _admin(lat, lon, key)
    dt = f"{_slug(district)}_{_slug(tehsil)}"

    # ---- Tier 1: the plot's OWN tehsil tile ----
    exact = _cropland_cells(dt)
    if exact:
        try:
            villages = _wfs(f"panchayat_boundaries:{dt}")
        except Exception:
            villages = []
        village = next((v for v in villages if _point_in_poly((lon, lat), v["geometry"])), None)
        vname = village["properties"].get("vill_name") if village else None
        cand = [((clat, clon), frac,
                 bool(village) and _point_in_poly((clon, clat), village["geometry"]))
                for (clat, clon), frac in exact]
        inside = sorted([c for c in cand if c[2]], key=lambda c: -c[1])   # in-village, highest frac
        chosen = inside[:min_points]
        if len(chosen) < min_points:                                     # buffer outward
            have = {c[0] for c in chosen}
            near = sorted([c for c in cand if c[0] not in have and _km((lat, lon), c[0]) <= buffer_km],
                          key=lambda c: _km((lat, lon), c[0]))
            chosen += near[:(min_points - len(chosen))]
        if chosen:
            return ({"state": state, "district": district, "tehsil": tehsil, "village": vname,
                     "n_points": len(chosen), "small_sample": len(chosen) < min_points,
                     "proxy": False},
                    [c[0] for c in chosen])

    # ---- Tier 2 (MERGED cropland + vegetation, NEAREST first, cropland-preferred) ----
    #   Pool nearby CROPLAND cells (<= max_proxy_km) AND local WorldCover VEGETATION (<= VEG_SEARCH_KM),
    #   then take the min_points NEAREST. Veg distances are handicapped (+VEG_DIST_PENALTY_KM) so nearby
    #   cropland wins when comparable and local vegetation only beats cropland that is MUCH farther. z is
    #   SELF-relative per point, so a non-cropland point still gives a valid "vs its own normal" read — but
    #   any veg point in the chosen set flags veg_fallback so the caller caps the confidence.
    catalog = _load_tile_catalog()
    ranked = sorted(catalog, key=lambda t: _km((lat, lon), (t[1], t[2])))
    pool = []                                    # (effective_dist, actual_dist, (clat,clon), kind, tile_dt)
    for tdt, tlat, tlon in ranked:
        if tdt == dt:
            continue
        if _km((lat, lon), (tlat, tlon)) > max_proxy_km + 60:            # centroid too far even w/ tile radius
            break
        for (clat, clon), frac in _cropland_cells(tdt):
            d = _km((lat, lon), (clat, clon))
            if d <= max_proxy_km:
                pool.append((d, d, (clat, clon), "cropland", tdt))
    for d, cell in _worldcover_vegetated_cells(lat, lon, C.VEG_SEARCH_KM, min_points):
        pool.append((d + C.VEG_DIST_PENALTY_KM, d, cell, "veg", None))
    if pool:
        pool.sort(key=lambda c: c[0])
        chosen = pool[:min_points]
        meta = {"state": state, "district": district, "tehsil": tehsil, "village": None,
                "n_points": len(chosen), "small_sample": len(chosen) < min_points, "proxy": False}
        if any(c[3] == "veg" for c in chosen):        # any non-cropland point -> veg-fallback (capped + noted)
            meta["veg_fallback"] = True
            meta["veg_km"] = round(max(c[1] for c in chosen), 1)
        else:                                          # all cropland -> the nearby-cropland proxy
            crop = chosen[0]
            meta["proxy"] = True
            meta["proxy_tile"] = crop[4]
            meta["proxy_km"] = round(crop[1], 1)
        return (meta, [c[2] for c in chosen])

    # ---- last resort: no cropland AND no vegetation nearby -> graceful out-of-coverage ----
    nearest_km = round(_km((lat, lon), (ranked[0][1], ranked[0][2])), 1) if ranked else None
    raise OutOfCoverage(state, district, tehsil, nearest_km)
