"""
shared_data_layer.py
====================
Single source of truth for all data fetching.
Both LSTM and XGBoost workers consume from here.
"""
# Current (missing as_completed):
from __future__ import annotations
# Fix — just add as_completed:
from concurrent.futures import ThreadPoolExecutor, as_completed

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
import time
import ee
import numpy as np
import pandas as pd
import requests
from dateutil.relativedelta import relativedelta

logger = logging.getLogger(__name__)
# ---------------------------------------------------------------------------
# GEE Batch Logger — separate file handler so batch progress is always
# readable regardless of what the root logger is doing
# ---------------------------------------------------------------------------
_gee_logger = logging.getLogger("gee_batch")
_gee_logger.setLevel(logging.DEBUG)

if not _gee_logger.handlers:
    _gee_handler = logging.FileHandler("gee_batch.log", mode="a")
    _gee_handler.setLevel(logging.DEBUG)
    _gee_handler.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    ))
    _gee_logger.addHandler(_gee_handler)
    _gee_logger.propagate = False   # don't bubble up to root logger

# ---------------------------------------------------------------------------
# Open-Meteo client (built once, shared)
# ---------------------------------------------------------------------------
# Open-Meteo forecast weather via the JSON API + a PLAIN retry session.
# NOT requests_cache (its cattrs serializer crashes on save -> "RequestsCookieJar
# not defined" -> the whole request failed -> silent climatology fallback), and NOT
# the flatbuffer SDK (the seasonal endpoint returns ~100 ensemble-member variables,
# so the old index-based parse read a RAIN member as temperature). The JSON endpoint
# returns clean named keys (rain_sum, temperature_2m_mean = absolute degC).
try:
    from retry_requests import retry
    _http = retry(requests.Session(), retries=3, backoff_factor=0.3)
except Exception:
    _http = requests.Session()
OPENMETEO_AVAILABLE = True


# ---------------------------------------------------------------------------
# Shared fetch result container
# ---------------------------------------------------------------------------
@dataclass
class SharedFetchResult:
    """All raw data needed by both LSTM and XGBoost workers."""

    # GEE history — full 10yr flat DataFrame (LSTM slices last 18)
    history_df: pd.DataFrame = field(default_factory=pd.DataFrame)

    # Forecast aggregates (Kelvin for both engines)
    forecast_precip_sum: float = 0.0
    forecast_temp_mean_k: float = 298.0
    forecast_source: str = "UNKNOWN"
    forecast_fallback_reason: Optional[str] = None
    
    # NEW: raw per-fortnight arrays for LSTM scaling
    forecast_precip_steps_raw: list = field(default_factory=list)
    forecast_temp_steps_raw:   list = field(default_factory=list)
    max_consecutive_gap: int = 0
    feature_completeness: dict = field(default_factory=dict)

    # Admin details
    admin: dict = field(default_factory=lambda: {
        "state": "Unknown", "district": "Unknown", "tehsil": "Unknown"
    })

    # Data quality
    coverage_pct: float = 0.0
    valid_steps: int = 0
    total_steps: int = 0
    quality_issues: list = field(default_factory=list)

    # Prediction datetime (derived from target_date)
    prediction_dt: Optional[datetime] = None
    data_cutoff_dt: Optional[datetime] = None
    days_ahead: int = 0
    gee_steps_fetched: int = 0
    prediction_mode: str = "historical"  # "historical" | "near_future" | "far_future"


# ---------------------------------------------------------------------------
# GEE fetch (7-fortnight lead, both models share the same window)
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# GEE fetch (TRAINING-EXACT REPLICA)
# ---------------------------------------------------------------------------

_LEAD_FORTNIGHTS = 7




#     return df
def _fetch_gee_raw(
    lat: float,
    lon: float,
    data_cutoff_str: str,
    steps: int,
    batch_size: int = 100
) -> pd.DataFrame:

    point = ee.Geometry.Point([lon, lat])
    cutoff_date = ee.Date(data_cutoff_str)
    start_date = cutoff_date.advance(
        ee.Number(steps - 1).multiply(-14), "day"
    )

    # unique run tag so interleaved log lines from concurrent
    # calls (e.g. 3 workers) are always traceable
    run_tag = f"({lat:.4f},{lon:.4f})@{data_cutoff_str}"

    _gee_logger.info(
        "[%s] START — steps=%d  batch_size=%d  total_batches=%d",
        run_tag, steps, batch_size, -(-steps // batch_size)
    )

    def safe_image(ic, band_names, reducer="mean", fallback=-9999):
        fallback_img = (
            ee.Image.constant([fallback] * len(band_names))
            .rename(band_names)
        )
        def build():
            if reducer == "sum":      return ic.sum()
            elif reducer == "median": return ic.median()
            else:                     return ic.mean()
        return ee.Image(
            ee.Algorithms.If(ic.size().gt(0), build(), fallback_img)
        )

    def safe_sample(img, fc, scale):
        return img.sampleRegions(
            collection=fc, properties=["row_id"],
            scale=scale, geometries=False
        )

    def sample_step(i):
        i = ee.Number(i)
        step_date = start_date.advance(i.multiply(14), "day")
        end      = step_date.advance(14, "day")
        ndvi_end = step_date.advance(16, "day")

        fc = ee.FeatureCollection([ee.Feature(point, {"row_id": 1})])

        era5_ic = ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR").filterDate(step_date, end)
        era5_states = safe_image(era5_ic.select([
            "temperature_2m", "dewpoint_temperature_2m",
            "u_component_of_wind_10m", "v_component_of_wind_10m",
            "surface_solar_radiation_downwards_sum", "surface_net_thermal_radiation_sum"
        ]), ["temperature_2m", "dewpoint_temperature_2m", "u_component_of_wind_10m",
             "v_component_of_wind_10m", "surface_solar_radiation_downwards_sum",
             "surface_net_thermal_radiation_sum"], reducer="mean")
        precip = safe_image(era5_ic.select("total_precipitation_sum")
            .map(lambda img: img.multiply(1000).rename("precipitation")),
            ["precipitation"], reducer="sum")
        runoff = safe_image(era5_ic.select("runoff_sum")
            .map(lambda img: img.multiply(1000).rename("runoff")),
            ["runoff"], reducer="sum")
        et = safe_image(era5_ic.select("total_evaporation_sum")
            .map(lambda img: img.multiply(-1000).rename("et")),
            ["et"], reducer="sum")

        smap_ic = ee.ImageCollection("NASA/SMAP/SPL4SMGP/008").filterDate(
            step_date, step_date.advance(3, "day"))
        smap = safe_image(smap_ic.select(["sm_surface", "sm_rootzone"]),
            ["sm_surface", "sm_rootzone"], reducer="mean", fallback=0.2)

        ndvi_ic = ee.ImageCollection("MODIS/061/MOD13Q1").filterDate(step_date, ndvi_end)
        ndvi = safe_image(ndvi_ic.select("NDVI"), ["NDVI"],
            reducer="median", fallback=0.4).multiply(0.0001)

        s_precip = safe_sample(precip,      fc, 5000)
        s_runoff = safe_sample(runoff,      fc, 5000)
        s_et     = safe_sample(et,          fc, 5000)
        s_era5   = safe_sample(era5_states, fc, 9000)
        s_smap   = safe_sample(smap,        fc, 11000)
        s_ndvi   = safe_sample(ndvi,        fc, 250)

        joined = s_precip
        for s in [s_runoff, s_et, s_era5, s_smap, s_ndvi]:
            joined = ee.Join.inner().apply(
                joined, s,
                ee.Filter.equals(leftField="row_id", rightField="row_id")
            ).map(lambda f: ee.Feature(f.get("primary"))
                .copyProperties(ee.Feature(f.get("secondary"))))

        stats = ee.FeatureCollection(joined).first().toDictionary()

        ndvi_available   = ndvi_ic.size().gt(0)
        smap_available   = smap_ic.size().gt(0)
        era5_available   = era5_ic.size().gt(0)
        precip_available = era5_ic.select("total_precipitation_sum").size().gt(0)
        coverage_fraction = (
            ee.Number(ndvi_available).multiply(0.4)
            .add(ee.Number(precip_available).multiply(0.2))
            .add(ee.Number(smap_available).multiply(0.2))
            .add(ee.Number(era5_available).multiply(0.2))
        )
        return ee.Feature(None, stats).set({
            "step": i, "date_str": step_date.format("YYYY-MM-dd"),
            "is_real_ndvi":      ee.Number(ndvi_available),
            "is_real_smap":      ee.Number(smap_available),
            "is_real_era5":      ee.Number(era5_available),
            "is_real_chirps":    ee.Number(precip_available),
            "coverage_fraction": coverage_fraction
        })

    # ── Build batches ─────────────────────────────────────────
    step_indices = list(range(steps))
    batches = [
        step_indices[i : i + batch_size]
        for i in range(0, steps, batch_size)
    ]
    total_batches = len(batches)
    results_dict  = {}

    def fetch_batch(batch_idx: int, batch: list) -> tuple[int, pd.DataFrame]:
        label = f"[{run_tag}] Batch {batch_idx+1:02d}/{total_batches}"

        for attempt in range(3):
            _gee_logger.info(
                "%s | attempt=%d | steps %d–%d | submitting to GEE",
                label, attempt + 1, batch[0], batch[-1]
            )
            try:
                T1 = time.time()
                raw = ee.FeatureCollection(
                    ee.List(batch).map(sample_step)
                ).getInfo()
                elapsed = round(time.time() - T1, 1)
                rows = len(raw.get("features", []))
                _gee_logger.info(
                    "%s | attempt=%d | SUCCESS | rows=%d | %.1fs",
                    label, attempt + 1, rows, elapsed
                )
                df = pd.DataFrame([f["properties"] for f in raw["features"]])
                return batch_idx, df

            except Exception as e:
                err_short = str(e)[:120].replace("\n", " ")
                _gee_logger.warning(
                    "%s | attempt=%d | FAILED — %s",
                    label, attempt + 1, err_short
                )
                if attempt < 2:
                    wait = 2 ** attempt
                    time.sleep(wait)
                    try:
                        if os.environ.get("GEE_PROJECT"):
                            ee.Initialize(project=os.environ["GEE_PROJECT"])
                        else:
                            ee.Initialize()   # credentials' own default project (no hardcoded project)
                    except Exception as init_err:
                        _gee_logger.warning(
                            "%s | GEE re-init failed: %s", label, str(init_err)[:60]
                        )
                # loop continues to next attempt

        # all 3 attempts exhausted
        _gee_logger.error(
            "%s | PERMANENT FAILURE — filling NaN", label
        )
        return batch_idx, pd.DataFrame([{"step": s, "date_str": None} for s in batch])

    # ── Submit all batches simultaneously ─────────────────────
    _gee_logger.info(
        "[%s] Submitting all %d batches simultaneously via ThreadPoolExecutor",
        run_tag, total_batches
    )

    with ThreadPoolExecutor(max_workers=total_batches) as pool:
        future_to_idx = {
            pool.submit(fetch_batch, idx, batch): idx
            for idx, batch in enumerate(batches)
        }
        for future in as_completed(future_to_idx):
            try:
                batch_idx, df = future.result()
            except Exception as e:
                # find which batch this was
                batch_idx = future_to_idx[future]
                batch = batches[batch_idx]
                _gee_logger.error(
                    "[%s] Batch %02d/%d future raised uncaught exception: %s — filling NaN",
                    run_tag, batch_idx + 1, total_batches, str(e)[:120]
                )
                df = pd.DataFrame([{"step": s, "date_str": None} for s in batch])
            
            results_dict[batch_idx] = df
            _gee_logger.info(
                "[%s] Batch %02d/%d received | running total: %d/%d batches done",
                run_tag, batch_idx + 1, total_batches,
                len(results_dict), total_batches
            )

    # ── Reassemble in order ───────────────────────────────────
    all_dfs = [results_dict[i] for i in sorted(results_dict.keys())]
    df = (
        pd.concat(all_dfs, ignore_index=True)
        .sort_values("step")
        .reset_index(drop=True)
    )

    nan_steps = int(df["date_str"].isna().sum())
    _gee_logger.info(
        "[%s] DONE — total_rows=%d  nan_steps=%d  failed_batches=%d",
        run_tag, len(df), nan_steps,
        sum(1 for d in all_dfs if d["date_str"].isna().all())
    )

    # At the end of _fetch_gee_raw, before return df:
    required_cols = [
        "NDVI", "precipitation", "temperature_2m", "et", "runoff",
        "sm_rootzone", "sm_surface", "dewpoint_temperature_2m",
        "u_component_of_wind_10m", "v_component_of_wind_10m",
        "surface_solar_radiation_downwards_sum",
        "surface_net_thermal_radiation_sum",
    ]
    for col in required_cols:
        if col not in df.columns:
            _gee_logger.error(
                "[%s] Column '%s' missing from GEE output — filling with NaN", run_tag, col
            )
            df[col] = np.nan

    nan_pct = df[required_cols].isna().mean().mean()
    if nan_pct > 0.9:
        _gee_logger.error(
            "[%s] GEE returned >90%% NaN across all critical columns. "
            "Likely an access-denied or quota error.", run_tag
        )
        # Return the df anyway — fetch_all will catch it via coverage_pct < 20

    return df


# ---------------------------------------------------------------------------
# BATCHED GEE fetch (M2) — sample ALL points in ONE FeatureCollection per step,
# instead of one _fetch_gee_raw call per point. The per-step reductions / scales /
# fallbacks are copied VERBATIM from _fetch_gee_raw above, so every numeric column
# is byte-identical to the per-point path (proven for NDVI; the exactness harness
# checks full-column parity). ADDS one column: SummaryQA (MOD13Q1 pixel reliability)
# — the Level-2 clear-view signal — sampled SEPARATELY so NDVI is untouched. The new
# column rides in history_df without touching the model (not in DYNAMIC_FEATURES) or
# the missingness enrichment (not in its numeric_cols).
# ---------------------------------------------------------------------------
def _fetch_gee_raw_batch(points, data_cutoff_str, steps, batch_size=100):
    """points: list of (lat, lon). Returns {row_id -> DataFrame}; each df matches
    _fetch_gee_raw(lat, lon, data_cutoff_str, steps) column-for-column, plus SummaryQA."""
    cutoff_date = ee.Date(data_cutoff_str)
    start_date = cutoff_date.advance(ee.Number(steps - 1).multiply(-14), "day")
    N = len(points)
    fc = ee.FeatureCollection([
        ee.Feature(ee.Geometry.Point([lon, lat]), {"row_id": k})
        for k, (lat, lon) in enumerate(points)])
    run_tag = f"BATCH[N={N}]@{data_cutoff_str}"
    _gee_logger.info("[%s] START — steps=%d batch_size=%d", run_tag, steps, batch_size)

    def safe_image(ic, band_names, reducer="mean", fallback=-9999):
        fallback_img = ee.Image.constant([fallback] * len(band_names)).rename(band_names)
        def build():
            if reducer == "sum":      return ic.sum()
            elif reducer == "median": return ic.median()
            else:                     return ic.mean()
        return ee.Image(ee.Algorithms.If(ic.size().gt(0), build(), fallback_img))

    def safe_sample(img, scale):
        return img.sampleRegions(collection=fc, properties=["row_id"],
                                 scale=scale, geometries=False)

    def sample_step(i):
        i = ee.Number(i)
        step_date = start_date.advance(i.multiply(14), "day")
        end      = step_date.advance(14, "day")
        ndvi_end = step_date.advance(16, "day")

        era5_ic = ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR").filterDate(step_date, end)
        era5_states = safe_image(era5_ic.select([
            "temperature_2m", "dewpoint_temperature_2m",
            "u_component_of_wind_10m", "v_component_of_wind_10m",
            "surface_solar_radiation_downwards_sum", "surface_net_thermal_radiation_sum"
        ]), ["temperature_2m", "dewpoint_temperature_2m", "u_component_of_wind_10m",
             "v_component_of_wind_10m", "surface_solar_radiation_downwards_sum",
             "surface_net_thermal_radiation_sum"], reducer="mean")
        precip = safe_image(era5_ic.select("total_precipitation_sum")
            .map(lambda img: img.multiply(1000).rename("precipitation")),
            ["precipitation"], reducer="sum")
        runoff = safe_image(era5_ic.select("runoff_sum")
            .map(lambda img: img.multiply(1000).rename("runoff")),
            ["runoff"], reducer="sum")
        et = safe_image(era5_ic.select("total_evaporation_sum")
            .map(lambda img: img.multiply(-1000).rename("et")),
            ["et"], reducer="sum")

        smap_ic = ee.ImageCollection("NASA/SMAP/SPL4SMGP/008").filterDate(
            step_date, step_date.advance(3, "day"))
        smap = safe_image(smap_ic.select(["sm_surface", "sm_rootzone"]),
            ["sm_surface", "sm_rootzone"], reducer="mean", fallback=0.2)

        ndvi_ic = ee.ImageCollection("MODIS/061/MOD13Q1").filterDate(step_date, ndvi_end)
        ndvi = safe_image(ndvi_ic.select("NDVI"), ["NDVI"],
            reducer="median", fallback=0.4).multiply(0.0001)
        # NEW: pixel reliability (0 good, 1 marginal, 2 snow/ice, 3 cloudy). Missing
        # composite -> fallback 3 (treat as not-clear). Sampled separately (NDVI stays
        # byte-identical); same collection/window/scale so it tracks the NDVI composite.
        summ = safe_image(ndvi_ic.select("SummaryQA"), ["SummaryQA"],
            reducer="median", fallback=3)

        s_precip = safe_sample(precip,      5000)
        s_runoff = safe_sample(runoff,      5000)
        s_et     = safe_sample(et,          5000)
        s_era5   = safe_sample(era5_states, 9000)
        s_smap   = safe_sample(smap,        11000)
        s_ndvi   = safe_sample(ndvi,        250)
        s_summ   = safe_sample(summ,        250)

        joined = s_precip
        for s in [s_runoff, s_et, s_era5, s_smap, s_ndvi, s_summ]:
            joined = ee.Join.inner().apply(
                joined, s,
                ee.Filter.equals(leftField="row_id", rightField="row_id")
            ).map(lambda f: ee.Feature(f.get("primary"))
                .copyProperties(ee.Feature(f.get("secondary"))))

        ndvi_available   = ndvi_ic.size().gt(0)
        smap_available   = smap_ic.size().gt(0)
        era5_available   = era5_ic.size().gt(0)
        precip_available = era5_ic.select("total_precipitation_sum").size().gt(0)
        coverage_fraction = (
            ee.Number(ndvi_available).multiply(0.4)
            .add(ee.Number(precip_available).multiply(0.2))
            .add(ee.Number(smap_available).multiply(0.2))
            .add(ee.Number(era5_available).multiply(0.2))
        )
        # step-level props are identical for every point at this step
        return ee.FeatureCollection(joined).map(lambda f: f.set({
            "step": i, "date_str": step_date.format("YYYY-MM-dd"),
            "is_real_ndvi":      ee.Number(ndvi_available),
            "is_real_smap":      ee.Number(smap_available),
            "is_real_era5":      ee.Number(era5_available),
            "is_real_chirps":    ee.Number(precip_available),
            "coverage_fraction": coverage_fraction,
        }))

    step_indices = list(range(steps))
    batches = [step_indices[i:i + batch_size] for i in range(0, steps, batch_size)]
    results = {}

    def fetch_batch(bi, batch):
        for attempt in range(3):
            try:
                raw = ee.FeatureCollection(ee.List(batch).map(sample_step)).flatten().getInfo()
                return bi, pd.DataFrame([f["properties"] for f in raw["features"]])
            except Exception as e:
                _gee_logger.warning("[%s] batch %d attempt %d failed: %s",
                                    run_tag, bi, attempt + 1, str(e)[:100])
                if attempt < 2:
                    time.sleep(2 ** attempt)
        # exhausted -> NaN rows for these steps, all points
        rows = [{"step": s, "date_str": None, "row_id": k} for s in batch for k in range(N)]
        return bi, pd.DataFrame(rows)

    with ThreadPoolExecutor(max_workers=len(batches)) as pool:
        futs = {pool.submit(fetch_batch, bi, b): bi for bi, b in enumerate(batches)}
        for fu in as_completed(futs):
            bi, df = fu.result()
            results[bi] = df

    big = pd.concat([results[i] for i in sorted(results)], ignore_index=True)

    required_cols = [
        "NDVI", "precipitation", "temperature_2m", "et", "runoff",
        "sm_rootzone", "sm_surface", "dewpoint_temperature_2m",
        "u_component_of_wind_10m", "v_component_of_wind_10m",
        "surface_solar_radiation_downwards_sum",
        "surface_net_thermal_radiation_sum",
    ]
    out = {}
    for k in range(N):
        dfk = (big[big["row_id"] == k].drop(columns=["row_id"])
               .sort_values("step").reset_index(drop=True))
        for col in required_cols:
            if col not in dfk.columns:
                dfk[col] = np.nan
        out[k] = dfk
    _gee_logger.info("[%s] DONE — %d points, %d rows each (nan_steps varies)",
                     run_tag, N, steps)
    return out


# ---------------------------------------------------------------------------
# Missingness / realism enrichment
# ---------------------------------------------------------------------------
def _enrich_missingness_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds realism-aware features without destroying raw missingness.
    """

    df = df.copy()

    numeric_cols = [
        "NDVI",
        "precipitation",
        "temperature_2m",
        "sm_rootzone",
        "sm_surface",
        "et",
        "runoff",
    ]

    existing_numeric = [
        c for c in numeric_cols
        if c in df.columns
    ]

    # ======================================================
    # S1 — PRESERVE MISSINGNESS FLAGS
    # ======================================================

    for col in existing_numeric:

        df[f"{col}_is_missing"] = (
            df[col].isna().astype(int)
        )

    # ======================================================
    # S4 — SOURCE PROVENANCE
    # ======================================================

    for col in existing_numeric:

        source_col = f"{col}_source"

        df[source_col] = "observed"

        df.loc[
            df[col].isna(),
            source_col
        ] = "missing"

    # ======================================================
    # S2 — OBSERVATION FRESHNESS
    # ======================================================

    freshness_targets = [
        c for c in [
            "NDVI",
            "precipitation",
            "sm_rootzone",
        ]
        if c in df.columns
    ]

    for col in freshness_targets:

        observed = (~df[col].isna()).astype(int)

        freshness = []
        counter = 0

        for v in observed:

            if v == 1:
                counter = 0
            else:
                counter += 1

            freshness.append(counter)

        df[f"{col}_steps_since_observed"] = freshness

    # ======================================================
    # S6 — GAP LENGTH FEATURES
    # ======================================================

    def compute_gap_lengths(series):

        gaps = []
        gap = 0

        for v in series.isna():

            if v:
                gap += 1
            else:
                gap = 0

            gaps.append(gap)

        return gaps

    if "NDVI" in df.columns:
        df["ndvi_gap_length"] = compute_gap_lengths(df["NDVI"])

    # ======================================================
    # S7 — TEMPORAL DENSITY
    # ======================================================

    if "date_str" in df.columns:

        df["date_str"] = pd.to_datetime(df["date_str"])

        diffs = (
            df["date_str"]
            .diff()
            .dt.days
        )

        df["days_since_last_obs"] = diffs.fillna(0)

    # ======================================================
    # S8 — FLATLINE DETECTION
    # ======================================================

    flatline_targets = [
        c for c in [
            "NDVI",
            "sm_rootzone",
            "et",
        ]
        if c in df.columns
    ]

    for col in flatline_targets:

        rolling_std = (
            df[col]
            .rolling(6, min_periods=4)
            .std()
        )

        df[f"{col}_is_flatlined"] = (
            rolling_std < 1e-5
        ).astype(int)

    # ======================================================
    # LIMITED FILL ONLY
    # ======================================================

    MAX_FILL = 2

    fill_cols = [
        c for c in existing_numeric
        if c != "NDVI"
    ]

    df[fill_cols] = (
        df[fill_cols]
        .ffill(limit=MAX_FILL)
        .bfill(limit=1)
    )

    # ======================================================
    # UPDATE SOURCE LABELS AFTER FILL
    # ======================================================

    for col in existing_numeric:

        source_col = f"{col}_source"

        filled_mask = (
            (df[f"{col}_is_missing"] == 1)
            & (df[col].notna())
        )

        df.loc[filled_mask, source_col] = "forward_fill"

    # ======================================================
    # SENSOR RELIABILITY
    # ======================================================

    if "NDVI" in df.columns:

        df["sensor_reliability"] = 1.0

        df.loc[
            df["NDVI_is_missing"] == 1,
            "sensor_reliability"
        ] -= 0.3

        if "ndvi_gap_length" in df.columns:

            df.loc[
                df["ndvi_gap_length"] > 2,
                "sensor_reliability"
            ] -= 0.3

        if "NDVI_steps_since_observed" in df.columns:

            df.loc[
                df["NDVI_steps_since_observed"] > 3,
                "sensor_reliability"
            ] -= 0.2

        df["sensor_reliability"] = (
            df["sensor_reliability"]
            .clip(0, 1)
        )

    return df
"""
Patch: granular fallback reasons in _fetch_forecast
=====================================================
Replace _fetch_forecast and _climatology_fallback in shared_data_layer.py
with these versions.

Every pathway that triggers the climatology fallback now carries a specific
reason string. That reason is returned as the third element of the tuple
and surfaces in the API response under metadata.forecast.fallback_reason.

Possible source/reason values
------------------------------
SEASONAL_API                          — fully successful, no fallback
CLIMATOLOGY_FALLBACK:NO_LIBRARY       — openmeteo_requests not installed
CLIMATOLOGY_FALLBACK:SEASONAL_FAILED  — seasonal API network/HTTP error
CLIMATOLOGY_FALLBACK:ARCHIVE_FAILED   — archive API network/HTTP error
CLIMATOLOGY_FALLBACK:PARSE_FAILED     — date/array length mismatch on parse
CLIMATOLOGY_FALLBACK:TEMP_NAN         — temp_K computed as NaN
CLIMATOLOGY_FALLBACK:TEMP_IMPLAUSIBLE — temp_K < 250 K
CLIMATOLOGY_FALLBACK:EXCEPTION        — unexpected error (message appended)
CLIMATOLOGY_FALLBACK:NO_HISTORY       — history df empty for target months
"""


import logging
from datetime import datetime

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta

logger = logging.getLogger(__name__)

_FALLBACK_PREFIX = "CLIMATOLOGY_FALLBACK"


def _parse_daily(resp, col_names: list) -> pd.DataFrame:
    """Parse Open-Meteo daily response into a clean naive-datetime DataFrame."""
    daily = resp.Daily()
    dates = pd.date_range(
        start=pd.to_datetime(daily.Time(), unit="s", utc=True),
        end=pd.to_datetime(daily.TimeEnd(), unit="s", utc=True),
        freq=pd.Timedelta(seconds=daily.Interval()),
        inclusive="left",
    ).tz_localize(None)

    data = {"date": dates}
    for i, name in enumerate(col_names):
        arr = daily.Variables(i).ValuesAsNumpy()
        if len(arr) != len(dates):
            raise ValueError(
                f"Length mismatch for column '{name}': "
                f"dates={len(dates)}, values={len(arr)}"
            )
        data[name] = arr

    return pd.DataFrame(data)
def _safe_aggregate(steps: list, agg: str, fallback: float) -> float:
    """Aggregate a possibly-empty list, returning fallback instead of nan."""
    if not steps:
        return fallback
    try:
        arr = np.array(steps, dtype=float)
        arr = arr[~np.isnan(arr)]
        if len(arr) == 0:
            return fallback
        return float(np.sum(arr) if agg == "sum" else np.mean(arr))
    except Exception:
        return fallback
def _fetch_forecast(
    lat: float,
    lon: float,
    prediction_dt: datetime,
    data_cutoff_dt: datetime,
    history_df: pd.DataFrame,
    mode: str = "historical",

) -> tuple[float, float, str, list, list]:
    """
    Returns (precip_sum_raw, temp_mean_k, source_label,
             precip_steps_raw, temp_steps_raw).
    The last two are per-fortnight raw arrays for LSTM scaling.
    """
    final_precip       = None
    final_temp_k       = None
    precip_steps_raw   = []   # raw per-step values for LSTM
    temp_steps_raw     = []
    reasons            = []

    if OPENMETEO_AVAILABLE:
        try:
            # Seasonal (CFS) forecast over the lead window, via the JSON API. Named
            # keys -> no ensemble-member index confusion. rain_sum = mm/day,
            # temperature_2m_mean = ABSOLUTE degC (NOT an anomaly). The seasonal API
            # only serves a ~9-month window around now, so a PAST target returns
            # HTTP 400 -> handled below -> climatology (correct: can't forecast the past).
            r = _http.get(
                "https://seasonal-api.open-meteo.com/v1/seasonal",
                params={
                    "latitude": lat, "longitude": lon,
                    "daily": "rain_sum,temperature_2m_mean",
                    "start_date": data_cutoff_dt.strftime("%Y-%m-%d"),
                    "end_date": prediction_dt.strftime("%Y-%m-%d"),
                }, timeout=30)
            if r.status_code != 200:
                raise ValueError(f"HTTP {r.status_code}: {r.json().get('reason', '')}")
            d = r.json().get("daily", {})
            rain   = pd.Series(d.get("rain_sum") or [], dtype="float64")
            temp_c = pd.Series(d.get("temperature_2m_mean") or [], dtype="float64")

            if len(rain) >= 98 and rain.notna().all():
                final_precip     = float(rain.sum())
                fn = np.arange(len(rain)) // 14
                precip_steps_raw = rain.groupby(fn).sum().head(7).tolist()
            else:
                reasons.append("PRECIP_INCOMPLETE")

            if len(temp_c) >= 98 and temp_c.notna().all():
                temp_k = temp_c + 273.15                 # absolute degC -> K
                mean_t = float(temp_k.mean())
                if 250.0 < mean_t < 330.0:
                    final_temp_k   = mean_t
                    fn = np.arange(len(temp_k)) // 14
                    temp_steps_raw = temp_k.groupby(fn).mean().head(7).tolist()
                else:
                    reasons.append("TEMP_IMPLAUSIBLE")
            else:
                reasons.append("TEMP_INCOMPLETE")

        except Exception as exc:
            reasons.append(f"API_ERROR:{str(exc)[:40]}")

    # Climatology fallback
    if final_precip is None or final_temp_k is None:
        clim_p_steps, clim_t_steps = _climatology_fallback(history_df, prediction_dt,mode=mode)

        if final_precip is None:
            precip_steps_raw = clim_p_steps
            final_precip = _safe_aggregate(clim_p_steps, "sum", 0.0)
            p_src = "CLIMATOLOGY"
        else:
            p_src = "API"

        if final_temp_k is None:
            temp_steps_raw = clim_t_steps
            final_temp_k = _safe_aggregate(clim_t_steps, "mean", 298.0)
            t_src = "CLIMATOLOGY"
        else:
            t_src = "API"
    else:
        p_src, t_src = "API", "API"

    source_label = f"P:{p_src}|T:{t_src}"
    if reasons:
        source_label += f"|REASONS:{','.join(reasons)}"

    return final_precip, final_temp_k, source_label, precip_steps_raw, temp_steps_raw


# def _climatology_fallback(
#     history_df: pd.DataFrame,
#     prediction_dt: datetime,
# ):

#     try:

#         df = history_df.copy()

#         df["date_str"] = pd.to_datetime(df["date_str"])
#         df["year"] = df["date_str"].dt.year

#         target_doy = prediction_dt.timetuple().tm_yday

#         best_window = None
#         best_diff = np.inf

#         for year, group in df.groupby("year"):

#             group = (
#                 group
#                 .sort_values("date_str")
#                 .reset_index(drop=True)
#             )

#             doy_values = (
#                 group["date_str"]
#                 .dt.dayofyear
#                 .values
#             )

#             doy_diff = np.abs(
#                 doy_values - target_doy
#             )

#             doy_diff = np.minimum(
#                 doy_diff,
#                 365 - doy_diff
#             )

#             target_pos = int(
#                 np.argmin(doy_diff)
#             )

#             start = max(
#                 0,
#                 target_pos-6
#             )

#             window = group.iloc[
#                 start:target_pos+1
#             ]

#             if len(window) != 7:
#                 continue

#             current_diff = doy_diff[target_pos]

#             if current_diff < best_diff:

#                 best_diff = current_diff

#                 best_window = window

#         if best_window is None:
#             return [], []

#         precip_steps = (
#             best_window[
#                 "precipitation"
#             ]
#             .tolist()
#         )

#         temp_steps = (
#             best_window[
#                 "temperature_2m"
#             ]
#             .tolist()
#         )
#         print("\nSELECTED CLIMATOLOGY WINDOW")
#         print("="*60)

#         print(
#             "Year:",
#             best_window["year"].iloc[0]
#         )

#         print(
#             best_window[
#                 ["date_str",
#                 "precipitation",
#                 "temperature_2m"]
#             ]
#         )
#         return (
#             precip_steps,
#             temp_steps
#         )

#     except Exception:
#         return [], []
def _climatology_fallback(
    history_df: pd.DataFrame,
    prediction_dt: datetime,
    mode: str = "historical",   # ADD THIS

):
    if mode == "historical":

        try:

            df = history_df.copy()

            df["date_str"] = pd.to_datetime(
                df["date_str"]
            )

            df = df.sort_values(
                "date_str"
            )

            # use latest 7 steps before prediction
            window = df[
                df["date_str"] <= prediction_dt
            ].tail(7)
            latest = df["date_str"].max()

            gap_days = (
                prediction_dt - latest
            ).days

            if gap_days > 30:

                logger.warning(
                    f"History ends {gap_days} days before prediction"
                )

                return [],[]
            if len(window) != 7:
                return [], []

            precip_steps = (
                window["precipitation"]
                .tolist()
            )

            temp_steps = (
                window["temperature_2m"]
                .tolist()
            )
            return (
                precip_steps,
                temp_steps
            )

        except Exception:
            return [], []
    else:
        df = history_df.copy()
        df["date_str"] = pd.to_datetime(df["date_str"])
        df["year"] = df["date_str"].dt.year
        target_doy = prediction_dt.timetuple().tm_yday
        
        # exclude current/future year from averaging
        df = df[df["year"] < prediction_dt.year]
        
        best_window = None
        best_diff = np.inf
        for year, group in df.groupby("year"):
            group = group.sort_values("date_str").reset_index(drop=True)
            doy_values = group["date_str"].dt.dayofyear.values
            doy_diff = np.abs(doy_values - target_doy)
            doy_diff = np.minimum(doy_diff, 365 - doy_diff)
            target_pos = int(np.argmin(doy_diff))
            start = max(0, target_pos - 6)
            window = group.iloc[start:target_pos+1]
            if len(window) != 7:
                continue
            current_diff = doy_diff[target_pos]
            if current_diff < best_diff:
                best_diff = current_diff
                best_window = window
        
        if best_window is None:
            return [], []
        
        precip_steps = best_window["precipitation"].tolist()
        temp_steps = best_window["temperature_2m"].tolist()
        return precip_steps, temp_steps

# ---------------------------------------------------------------------------
# Admin fetch
# ---------------------------------------------------------------------------
# CoreStack is flaky (transient 500 / proxy errors). A single failed call would
# collapse admin to "Unknown", which is not a trained category -> it encodes to
# index 0 -> a shifted forecast. Hardened two ways:
#   1. retry with exponential backoff (most 500s succeed on the next try);
#   2. cache SUCCESSFUL lookups by (lat, lon) so repeats / a later outage reuse a
#      known-good answer. The all-failed "Unknown" fallback is NOT cached.
# In-memory, per-process: persists across requests in serve_api; per-run in the CLI.
_ADMIN_CACHE: dict = {}


def _fetch_admin(lat: float, lon: float, corestack_key: str,
                 retries: int = 3, backoff: float = 0.5) -> dict:
    key = (round(float(lat), 4), round(float(lon), 4))
    if key in _ADMIN_CACHE:
        return _ADMIN_CACHE[key]

    url = (
        f"https://geoserver.core-stack.org/api/v1/get_admin_details_by_latlon/"
        f"?latitude={lat}&longitude={lon}"
    )
    headers = {"X-API-Key": corestack_key, "Content-Type": "application/json"}
    for attempt in range(retries):
        try:
            resp = requests.get(url, headers=headers, timeout=15)
            resp.raise_for_status()
            res = resp.json()
            admin = {
                "state": str(res.get("State", "Unknown")).strip().title(),
                "district": str(res.get("District", "Unknown")).strip().title(),
                "tehsil": str(res.get("Tehsil", "Unknown")).strip().title(),
            }
            _ADMIN_CACHE[key] = admin           # cache successful lookups only
            return admin
        except Exception as exc:
            logger.warning("CoreStack attempt %d/%d failed: %s", attempt + 1, retries, exc)
            if attempt < retries - 1:
                time.sleep(backoff * (2 ** attempt))   # 0.5s, 1s, ...
    return {"state": "Unknown", "district": "Unknown", "tehsil": "Unknown"}


# ---------------------------------------------------------------------------
# Main entry point — called once per request
# ---------------------------------------------------------------------------
def fetch_all(
    lat: float,
    lon: float,
    target_date: str,
    corestack_key: str,
    years_lookback: int = 10,
    lead_fortnights: int = _LEAD_FORTNIGHTS,
    prefetched_history_df=None,
) -> SharedFetchResult:
    """
    Serial phase:
      1. GEE fetch (10yr, 7-fortnight cutoff)
      2. Admin fetch (CoreStack)
      3. Forecast fetch (Open-Meteo seasonal → climatology)

    Returns SharedFetchResult consumed by both LSTM and XGBoost workers.
    """
    today = pd.Timestamp.now().normalize()
    prediction_dt = pd.to_datetime(target_date)
    days_ahead = (prediction_dt - today).days

    if days_ahead <= 0:
        mode = "historical"
    elif days_ahead <= 98:  # up to 7 fortnights ahead
        mode = "near_future"
    else:
        mode = "far_future"  # beyond Open-Meteo seasonal range
    result = SharedFetchResult()
    prediction_dt = pd.to_datetime(target_date)
    data_cutoff_dt = prediction_dt - pd.Timedelta(days=lead_fortnights * 14)
    result.prediction_dt = prediction_dt.to_pydatetime()
    result.data_cutoff_dt = data_cutoff_dt.to_pydatetime()

    steps = int(years_lookback * 26) + _LEAD_FORTNIGHTS  # e.g. 267
    result.total_steps = steps


    # --- 1. GEE -----------------------------------------------------------------
    logger.info("Fetching GEE data: %d steps, cutoff=%s", steps, data_cutoff_dt.date())
    # Change data_cutoff_str to target_date and add _LEAD_FORTNIGHTS to steps
    if days_ahead <= 0:
        # fully historical — fetch all steps ending at prediction_dt
        gee_anchor = prediction_dt
        gee_steps = steps  # full 85
    else:
        # future — anchor at today, only fetch what exists
        # how many fortnights back from prediction_dt does today fall?
        days_from_prediction_to_today = days_ahead
        fortnights_missing = days_from_prediction_to_today // 14
        gee_steps = max(steps - fortnights_missing, 1)
        gee_anchor = today
    result.days_ahead = days_ahead
    result.gee_steps_fetched = gee_steps
    result.prediction_mode = mode
    T1 = time.time()
    # --- hardened GEE fetch --------------------------------------------------
    # The high-volume endpoint intermittently returns DEGRADED responses for
    # minutes at a time: it drops the coverage_fraction band and/or returns
    # all-NaN NDVI/precip/temp, then recovers. Retry at the source with
    # exponential backoff before falling through to the guards below. Benefits
    # both serving (transparent recovery from short blips) and batch eval.
    _anchor_str = gee_anchor.strftime("%Y-%m-%d")
    _GEE_TRIES, _GEE_BACKOFF = 3, 4.0

    def _gee_degraded(df):
        if df is None or len(df) == 0 or "date_str" not in df.columns:
            return True
        if "coverage_fraction" not in df.columns:
            return True
        crit = [c for c in ("NDVI", "precipitation", "temperature_2m")
                if c in df.columns]
        if not crit or all(df[c].isna().all() for c in crit):
            return True
        return False

    if prefetched_history_df is not None:
        # M2: the batched fetch already produced this point's rows (identical columns
        # to _fetch_gee_raw, plus SummaryQA). Skip the per-point fetch + retry entirely;
        # every downstream step below is unchanged. Serve path (prefetched=None) is
        # byte-identical to before.
        history_df = prefetched_history_df.copy()
    else:
        history_df = _fetch_gee_raw(lat, lon, _anchor_str, gee_steps)
        _gee_try = 1
        while _gee_degraded(history_df) and _gee_try < _GEE_TRIES:
            _delay = _GEE_BACKOFF * (2 ** (_gee_try - 1))
            logger.warning("degraded GEE response (try %d/%d) — retrying in %.0fs",
                           _gee_try, _GEE_TRIES, _delay)
            time.sleep(_delay)
            history_df = _fetch_gee_raw(lat, lon, _anchor_str, gee_steps)
            _gee_try += 1

    # Guard 1: empty df
    if len(history_df) == 0 or "date_str" not in history_df.columns:
        result.quality_issues.append(
            "ERROR: GEE returned no data for this location. "
            "Possible causes: GEE access denied, project quota exceeded, "
            "or coordinates are outside supported coverage area."
        )
        return result

    # Guard 2: missing columns
    REQUIRED_COLS = [
        "date_str",
        "NDVI",
        "precipitation",
        "temperature_2m",
        "dewpoint_temperature_2m",
        "et",
        "runoff",
        "sm_rootzone",
        "sm_surface",
        "u_component_of_wind_10m",
        "v_component_of_wind_10m",
        "surface_solar_radiation_downwards_sum",
        "surface_net_thermal_radiation_sum",
        "coverage_fraction",
    ]
    missing_cols = [c for c in REQUIRED_COLS if c not in history_df.columns]
    if missing_cols:
        logger.warning("GEE response missing columns: %s — filling with NaN", missing_cols)
        result.quality_issues.append(
            f"WARNING: GEE missing columns {missing_cols} — "
            "likely a partial fetch failure. Predictions may be unreliable."
        )
        for col in missing_cols:
            history_df[col] = np.nan

    # Guard 3: if ALL critical columns are NaN (not just missing)
    CRITICAL_COLS = ["NDVI", "precipitation", "temperature_2m"]
    all_nan = all(
        history_df[c].isna().all()
        for c in CRITICAL_COLS
        if c in history_df.columns
    )
    if all_nan:
        result.quality_issues.append(
            "ERROR: GEE returned all-NaN values for critical features "
            "(NDVI, precipitation, temperature). "
            "This usually means GEE access was denied mid-batch or the "
            "location has no satellite coverage."
        )
        return result
    T2 = time.time()
    print("Size of Data:",gee_steps,len(history_df),gee_anchor)
    print("Time for GEE Data Extraction:",T2-T1)
    # ======================================================
    # REALISM / MISSINGNESS ENRICHMENT
    # ======================================================
    history_df = _enrich_missingness_features(history_df)
    result.history_df = history_df
    result.total_steps = len(history_df)  # fix 5 also goes here

    # ── fix 4: isolate real rows for quality checks ──
    today = pd.Timestamp.now().normalize()
    real_df = history_df[
        pd.to_datetime(history_df["date_str"]) <= today
    ].copy()

    # FEATURE COMPLETENESS AUDIT — on real_df
    critical_cols = ["NDVI", "precipitation", "temperature_2m", "sm_rootzone", "et"]
    feature_completeness = {}
    for col in critical_cols:
        if col in real_df.columns:
            pct = real_df[col].notna().mean() * 100
            feature_completeness[col] = round(pct, 1)
            if pct < 70:
                result.quality_issues.append(
                    f"LOW_{col.upper()}_COMPLETENESS:{round(pct,1)}%"
                )
    result.feature_completeness = feature_completeness

    # Replace the existing coverage_fraction block in fetch_all():
    if "coverage_fraction" in real_df.columns:
        result.coverage_pct = round(real_df["coverage_fraction"].mean() * 100, 1)
        result.valid_steps = int((real_df["coverage_fraction"] >= 0.5).sum())
    elif len(real_df) == 0:
        result.quality_issues.append(
            "ERROR: GEE returned zero rows. Check GEE project permissions and quota."
        )
        result.coverage_pct = 0.0
        result.valid_steps = 0
    else:
        # GEE succeeded but coverage_fraction wasn't computed (join failed)
        result.quality_issues.append(
            "WARNING: coverage_fraction column missing — GEE join may have partially failed."
        )
        ndvi_valid = real_df["NDVI"].notna().mean() if "NDVI" in real_df.columns else 0.0
        result.coverage_pct = round(ndvi_valid * 100, 1)
        result.valid_steps = int(ndvi_valid * len(real_df))

    # QUALITY FLAGS — on real_df
    if result.coverage_pct < 40:
        result.quality_issues.append("CRITICAL: Historical coverage extremely low (<40%).")
    recent_ndvi = real_df.tail(18)["NDVI"]   # last 18 real rows
    if (recent_ndvi.isna().sum() + (recent_ndvi == 0).sum()) > 9:
        result.quality_issues.append("WARNING: >50% of recent LSTM window missing.")
    if real_df["NDVI"].mean() < 0.12:
        result.quality_issues.append("NOTICE: Low biomass — may be water, urban, or fallow.")

    # FAIL-FAST — on real_df
    if result.coverage_pct < 20:
        result.quality_issues.append(
            f"ERROR: Only {result.coverage_pct}% coverage. Minimum 20% required."
        )
        return result

    # TEMPORAL CONTINUITY — on real_df
    coverage_series = (real_df["coverage_fraction"] >= 0.5).astype(int)
    history_df["coverage_valid"] = history_df["coverage_fraction"].ge(0.5).astype(int)  # still tag full df

    gap_lengths = []
    current_gap = 0
    for val in coverage_series:
        if val == 0:
            current_gap += 1
        else:
            if current_gap > 0:
                gap_lengths.append(current_gap)
            current_gap = 0
    if current_gap > 0:
        gap_lengths.append(current_gap)

    max_gap = max(gap_lengths) if gap_lengths else 0
    result.max_consecutive_gap = int(max_gap)
    if max_gap >= 8:
        result.quality_issues.append(
            f"CRITICAL: Long temporal gap detected ({max_gap} windows)"
        )

    # --- 2. Admin ---------------------------------------------------------------
    logger.info("Fetching admin details for (%.4f, %.4f)", lat, lon)
    result.admin = _fetch_admin(lat, lon, corestack_key)

    # --- 3. Forecast ------------------------------------------------------------
    logger.info("Fetching forecast: %s → %s", data_cutoff_dt.date(), prediction_dt.date())
    f_precip, f_temp_k, source, precip_steps, temp_steps = _fetch_forecast(
        lat, lon,
        prediction_dt.to_pydatetime(),
        data_cutoff_dt.to_pydatetime(),
        history_df,
        mode=result.prediction_mode,
    )
    result.forecast_precip_steps_raw = precip_steps
    result.forecast_temp_steps_raw   = temp_steps
# ... rest of existing sanity checks unchanged ...
    result.forecast_source = source

    if "CLIMATOLOGY" in source or "REASONS:" in source:
        result.forecast_fallback_reason = source
    else:
        result.forecast_fallback_reason = None

    # Stability nudge (mirrors existing LSTM logic)
# ======================================================
# TEMPERATURE SANITY CHECK
# ======================================================

    if np.isnan(f_temp_k):

        result.quality_issues.append(
            "FORECAST_TEMP_NAN_FALLBACK"
        )

        f_temp_k = 298.0

    elif not (240.0 <= f_temp_k <= 330.0):

        result.quality_issues.append(
            f"FORECAST_TEMP_IMPLAUSIBLE:{round(f_temp_k, 2)}K"
        )

        f_temp_k = 298.0
    if np.isnan(f_temp_k) or f_temp_k < 294.0:
        result.quality_issues.append("FORECAST_TEMP_FALLBACK_APPLIED")
    f_precip = 0.0 if np.isnan(f_precip) else f_precip

    result.forecast_precip_sum = f_precip
    result.forecast_temp_mean_k = f_temp_k
    result.forecast_source = source

    logger.info("Forecast: precip=%.1f mm, temp=%.1f K, source=%s", f_precip, f_temp_k, source)
    return result



