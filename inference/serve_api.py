"""
serve_api.py — FastAPI wrapper over the TFT + Prithvi NDVI inference.

Thin HTTP layer: NO new inference logic. It reuses the exact same path as the
CLI — ndvi_core.download.fetch_walkback_tile, ndvi_core.inference.forecast_point,
ndvi_core.indicators.vegetation_indicators — plus fetch_all for the numerical
history. The heavy model (frozen Prithvi backbone + LoRA + TFT) is loaded ONCE at
startup; each request only fetches its data and refreshes the tile table
(ndvi_core.model_io.refresh_tile_store) before the forward pass.

Run:
    uvicorn serve_api:app --host 0.0.0.0 --port 8000
    # or:  python serve_api.py

Request:
    POST /forecast  {"lat": 25.44, "lon": 91.71, "date": "2024-08-15"}
    add ?debug=true for the verbose payload.
Response 200 (canonical dict from ndvi_core.results, SAME as the CLI):
    { location, admin, target_date, forecast_ndvi, relative_vegetation_condition,
      anomaly{z, baseline_mean, baseline_std} }
    ?debug=true also adds: as_of, anchor_date, prithvi_tile{quarter,year,used},
      anomaly.baseline_source, anomaly.baseline_n (baseline sample size), anomaly.ndvi
      (raw anomaly) + anomaly.pct, and a staleness `note` when the anchor NDVI was carried forward.
Errors (faithful to the CLI failure points):
    502  upstream data fetch failed (GEE/CoreStack/Open-Meteo) or composite fetch
    422  insufficient NDVI history for the window
"""

import os
import sys
import shutil
import threading
from uuid import uuid4
from datetime import datetime, timedelta

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)                       # inference/: shared_data_layer, infer_cli
sys.path.insert(0, os.path.dirname(BASE))      # repo root: ndvi_core (works run-in-place)

from ndvi_core import config as ncfg
from ndvi_core import download as ncd
from ndvi_core import dates as ncdates       # shared anchor/target-date geometry (C1)
from ndvi_core import model_io as ncm_io
from ndvi_core import inference as ncf_inf
from ndvi_core import indicators as ncf_ind
from ndvi_core import results as ncres       # shared canonical result dict (C2)
from infer_cli import build_static_lookup, nearest_static_profile   # reuse, no dup

# --- configuration (env-overridable, same defaults as the CLI) --------------
RUN_DIR = os.environ.get("RUN_DIR", "weights")   # self-contained model asset (download target)
MODEL_PATH = os.path.join(RUN_DIR, os.environ.get("MODEL_PATH", "tft_temporal_production_ft.pt"))
SCALER_PATH = os.path.join(RUN_DIR, os.environ.get("SCALER_PATH", "standard_scaler_temporal_tft_ft.pkl"))
ENCODER_PATH = os.path.join(RUN_DIR, os.environ.get("ENCODER_PATH", "label_encoders_temporal_tft_ft.pkl"))
LOOKUP_CSV = os.environ.get("LOOKUP_CSV", "weights/mws_static_lookup_UNSCALED.tsv")  # reader sniffs .csv/.tsv
MODEL_DIR = os.environ.get("MODEL_DIR", "weights")   # Prithvi arch/config vendored into the model asset
CACHE_DIR = os.environ.get("CACHE_DIR", "inference_cache_api")
CORESTACK_KEY = os.environ.get("CORESTACK_KEY", "")   # set via env; no key shipped
YEARS_LOOKBACK = int(os.environ.get("YEARS_LOOKBACK", 10))
DEVICE = os.environ.get("DEVICE", "cuda")
KEEP_TILES = os.environ.get("KEEP_TILES", "0") == "1"

# The model + projector are shared and mutated per request (tile refresh), so the
# request-critical section is serialized. GPU-bound anyway.
_MODEL_LOCK = threading.Lock()

app = FastAPI(title="NDVI TFT+Prithvi forecast", version="1.0")

# Populated at startup.
_STATE = {}


class ForecastRequest(BaseModel):
    lat: float = Field(..., ge=-90, le=90)
    lon: float = Field(..., ge=-180, le=180)
    date: str | None = Field(None, description="anchor / forecast-from (current) date "
                             "YYYY-MM-DD; default = today. Forecast target = date + lead. "
                             "Must be today or earlier -- future dates are rejected (422).")
    # NOTE: the seasonal-baseline knobs (baseline_years, baseline_month_window) and the
    # serve-time `bias` correction are intentionally NOT exposed on the API request —
    # they confuse callers and their production defaults (all years, target month only,
    # bias = 0) are what we serve. They stay tunable on the CLI (--baseline_years /
    # --baseline_month_window / --bias) and are explained in the docs. Fixed defaults
    # are applied in forecast() below.


@app.on_event("startup")
def _load():
    """Load EE, artifacts, lookup tree, and the model — ONCE."""
    import ee
    ee.Initialize(opt_url=ncd.HIGH_VOLUME)          # credentials' own project
    os.makedirs(CACHE_DIR, exist_ok=True)

    scaler = joblib.load(SCALER_PATH)
    encoders = joblib.load(ENCODER_PATH)
    lk, tree = build_static_lookup(LOOKUP_CSV)

    # The model's OWN window/lead (train_config.json in RUN_DIR; else config.py).
    tcfg = ncm_io.load_train_config(RUN_DIR)

    # Build the model once over an (initially empty) cache; per request we refresh
    # the tile table with that request's freshly-fetched composite.
    model, _, _ = ncm_io.build_model(
        MODEL_DIR, MODEL_PATH, encoders, CACHE_DIR, latlon={}, device=DEVICE, tcfg=tcfg)

    _STATE.update(model=model, scaler=scaler, encoders=encoders, lk=lk, tree=tree,
                  window=int(tcfg["window"]), lead=int(tcfg["lead"]))
    print(f"[serve_api] ready — model + artifacts loaded (window={tcfg['window']} lead={tcfg['lead']}).")


@app.get("/health")
def health():
    return {"status": "ok" if _STATE.get("model") is not None else "loading",
            "lead_fortnights": _STATE.get("lead", ncfg.LEAD),
            "window": _STATE.get("window", ncfg.WINDOW)}


@app.post("/forecast")
def forecast(req: ForecastRequest, debug: bool = False):
    if _STATE.get("model") is None:
        raise HTTPException(status_code=503, detail="model still loading")

    # `date` = the as-of / forecast-from date (default today). C1 geometry: anchor
    # s0 = nearest 14-day grid fortnight <= date; forecast TARGET = anchor + lead
    # (mirrors training's target = anchor_slot + lead). A missing anchor reading is
    # ffilled in build_window, never slid back. Resolved via the SHARED ndvi_core.dates
    # helper so the API and CLI produce identical anchor/target for the same input.
    try:
        ff_dt, anchor_dt, target_dt = ncdates.resolve_forecast_dates(req.date, lead=_STATE["lead"])
        forecast_from = ff_dt.strftime("%Y-%m-%d")
        anchor = anchor_dt.strftime("%Y-%m-%d")
        target = target_dt.strftime("%Y-%m-%d")
    except ValueError as e:
        # bad format OR a future date (both raised with a user-facing message).
        raise HTTPException(status_code=422, detail=str(e))

    from shared_data_layer import fetch_all

    # 1) numerical history + admin + forecast statics (upstream fetch).
    # Client gets a clean message; the technical cause is logged server-side.
    try:
        shared = fetch_all(lat=req.lat, lon=req.lon, target_date=target,
                           corestack_key=CORESTACK_KEY, years_lookback=YEARS_LOOKBACK,
                           lead_fortnights=_STATE["lead"])
    except Exception as e:
        print(f"[serve_api] upstream fetch exception ({req.lat},{req.lon},{target}): {e}")
        raise HTTPException(status_code=502,
                            detail="Upstream data fetch failed (satellite / weather / admin). "
                                   "Please retry shortly.")
    if any("ERROR:" in q for q in shared.quality_issues):
        print(f"[serve_api] data-fetch ERROR ({req.lat},{req.lon},{target}): {shared.quality_issues}")
        raise HTTPException(status_code=502,
                            detail="No usable satellite/weather data for this location and date. "
                                   "The point may be over water, outside the supported coverage area, "
                                   "or hit a transient upstream issue — check the coordinates and retry.")

    # 2) fresh composite into a per-request tile dir (isolated -> no collisions).
    # lid keys the tile FILE by location too (defense-in-depth beyond the isolated dir).
    lid = ncd.loc_id(req.lat, req.lon)
    req_cache = os.path.join(CACHE_DIR, f"req_{uuid4().hex[:8]}")
    os.makedirs(req_cache, exist_ok=True)
    try:
        try:
            _, _, tyear, tq = ncd.fetch_walkback_tile(
                lid, req.lat, req.lon, target, req_cache, lag=1)
        except Exception as e:
            print(f"[serve_api] composite fetch failed ({req.lat},{req.lon},{target}): {e}")
            raise HTTPException(status_code=502,
                                detail="Could not fetch a cloud-free satellite image tile for this "
                                       "location in the lookback window. Try a nearby point or another date.")

        prof = nearest_static_profile(req.lat, req.lon, lk=_STATE["lk"], tree=_STATE["tree"])

        # 3) refresh tile table + forward (shared model -> serialize)
        with _MODEL_LOCK:
            key_to_row, zero_idx = ncm_io.refresh_tile_store(
                _STATE["model"], req_cache, {lid: (req.lat, req.lon)})
            try:
                ndvi, emb_idx = ncf_inf.forecast_point(
                    _STATE["model"], shared, lid, req.lat, req.lon, prof, tyear, tq,
                    key_to_row, zero_idx, _STATE["scaler"], _STATE["encoders"],
                    window=_STATE["window"], device=DEVICE)
            except ValueError as e:                 # build_window: too few history rows
                print(f"[serve_api] insufficient history ({req.lat},{req.lon},{target}): {e}")
                raise HTTPException(status_code=422,
                                    detail="Not enough NDVI history at this location to build the "
                                           "model input window. Try a point with a longer observation record.")
    finally:
        if not KEEP_TILES:
            shutil.rmtree(req_cache, ignore_errors=True)

    # 4) NDVI -> vegetation output indicators (same module as the CLI).
    # Seasonal-baseline knobs + bias are fixed to their production defaults (all years,
    # target month only, no bias); they are not exposed on the request (see ForecastRequest).
    ind = ncf_ind.vegetation_indicators(
        ndvi, shared.history_df, target, prof=prof,
        month_window=0, years=None, bias=0.0)

    # 5) canonical result dict (SAME builder the CLI uses). ?debug=true adds the
    # as-of/anchor dates, the Prithvi tile, and the baseline source.
    stale_fn, last_obs = ncf_inf.anchor_staleness(shared, anchor)
    tile = {"quarter": tq, "year": int(tyear), "used": emb_idx != zero_idx}
    return ncres.build_result(
        req.lat, req.lon, shared.admin, forecast_from, anchor, target,
        ndvi, ind, tile, verbose=debug,
        stale_fortnights=stale_fn, last_obs=last_obs)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("serve_api:app", host=os.environ.get("HOST", "0.0.0.0"),
                port=int(os.environ.get("PORT", 8000)), workers=1)
