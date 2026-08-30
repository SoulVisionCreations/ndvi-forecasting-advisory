"""
advisory.engine — the IN-PROCESS AdvisoryEngine (GWL pattern).

Holds the forecaster (frozen Prithvi backbone + LoRA + TFT) loaded ONCE and drives
the whole advisory in-process — no HTTP to /forecast. It reuses the forecaster
verbatim (nothing here changes the model or serve_api) and only:

  * forecasts the plot + ~12 sampled points as ONE BATCHED forward (13 stacked
    windows + a [13] emb_idx through a single model() call). Each point keeps its
    OWN window / tile / emb_idx, so batch != share (no cross-point contamination);
    the only difference vs the per-point path is fp32 GPU reduction-order noise
    (~1e-6 on NDVI), which vanishes under vegetation_indicators' 2-dp z rounding.
  * reads the REAL per-point clear-view fraction off the fetch result (1b) instead
    of the season-prior estimate.

Everything downstream (aggregate / data_quality / rule_engine / phraser) is the
SAME code the HTTP advisory uses via build_advisory_from_facts — so the structured
advisory is identical; only the forecast source and the data-quality source change.

Runs on a GPU host in the repo-canonical env (pyproject: py3.11, torch==2.10.0+cu128).
"""
import os
import sys
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor
from uuid import uuid4

# --- make the forecaster importable exactly like serve_api does ---------------
_BASE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_BASE)
for _p in (os.path.join(_REPO, "inference"), _REPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from . import advisory as _adv          # build_advisory_from_facts (verbatim downstream)
from . import config as C
from . import sampling


# ---------------------------------------------------------------------------
# config — same env-overridable paths/defaults as serve_api (kept in sync)
# ---------------------------------------------------------------------------
RUN_DIR      = os.environ.get("RUN_DIR", "weights")
MODEL_PATH   = os.path.join(RUN_DIR, os.environ.get("MODEL_PATH", "tft_temporal_production_ft.pt"))
SCALER_PATH  = os.path.join(RUN_DIR, os.environ.get("SCALER_PATH", "standard_scaler_temporal_tft_ft.pkl"))
ENCODER_PATH = os.path.join(RUN_DIR, os.environ.get("ENCODER_PATH", "label_encoders_temporal_tft_ft.pkl"))
LOOKUP_CSV   = os.environ.get("LOOKUP_CSV", "weights/mws_static_lookup_UNSCALED.tsv")
MODEL_DIR    = os.environ.get("MODEL_DIR", "weights")
CACHE_DIR    = os.environ.get("CACHE_DIR", "inference_cache_advisory")
YEARS_LOOKBACK = int(os.environ.get("YEARS_LOOKBACK", 10))
DEVICE       = os.environ.get("DEVICE", "cuda")
KEEP_TILES   = os.environ.get("KEEP_TILES", "0") == "1"
# how many points to fetch concurrently (I/O-bound GEE/CoreStack). Cap to be nice
# to the high-volume endpoint; each fetch_all itself already fans out ~3 batches.
FETCH_WORKERS = int(os.environ.get("ADVISORY_FETCH_WORKERS", "13"))
# fetch mode: "batched" = ONE _fetch_gee_raw_batch for all points (M2, default);
# "parallel" = per-point fetch_all in threads (M1, trivially-exact fallback).
FETCH_MODE = os.environ.get("ADVISORY_FETCH_MODE", "batched")


def preflight_paths():
    """Check every model-artifact path the in-process engine needs EXISTS + is
    readable, BEFORE we try to load. Returns a list of human-readable problems
    (empty list = all good). Lets the server FAIL FAST with a clear message instead
    of starting in a degraded / HTTP-fallback state because of a bad path.

    These are exactly the files the load reads: the fused bundle, scaler, encoders,
    the static lookup, and (in MODEL_DIR) the Prithvi base config.json + prithvi_mae.py
    that load_prithvi_lora imports."""
    required = [
        ("RUN_DIR (model folder)",             RUN_DIR,                                   "dir"),
        ("MODEL_DIR (Prithvi arch + config)",  MODEL_DIR,                                 "dir"),
        ("model checkpoint (fused .pt)",       MODEL_PATH,                                "file"),
        ("feature scaler (.pkl)",              SCALER_PATH,                               "file"),
        ("label encoders (.pkl)",              ENCODER_PATH,                              "file"),
        ("static lookup (.tsv)",               LOOKUP_CSV,                                "file"),
        ("Prithvi base config (config.json)",  os.path.join(MODEL_DIR, "config.json"),    "file"),
        ("Prithvi arch code (prithvi_mae.py)", os.path.join(MODEL_DIR, "prithvi_mae.py"), "file"),
    ]
    problems = []
    for label, path, kind in required:
        if not os.path.exists(path):
            problems.append(f"{label}: MISSING -> {path}")
        elif kind == "dir" and not os.path.isdir(path):
            problems.append(f"{label}: not a directory -> {path}")
        elif kind == "file" and not os.path.isfile(path):
            problems.append(f"{label}: not a file -> {path}")
        elif not os.access(path, os.R_OK):
            problems.append(f"{label}: unreadable (permissions) -> {path}")
    return problems


class AdvisoryEngine:
    """Load once; call advise() per farmer. Mirrors serve_api's per-request pieces
    (fetch_all -> fetch_walkback_tile -> refresh_tile_store -> forecast) but batches
    the 13-point forward. serve_api / the forecaster are NOT modified."""

    def __init__(self, corestack_key=None, device=None):
        import ee
        import joblib
        from ndvi_core import download as ncd
        from ndvi_core import model_io as ncm_io
        from infer_cli import build_static_lookup

        ee.Initialize(opt_url=ncd.HIGH_VOLUME)
        os.makedirs(CACHE_DIR, exist_ok=True)
        self.device = device or DEVICE
        self.corestack_key = corestack_key if corestack_key is not None else \
            os.environ.get("CORESTACK_KEY", "")

        self.scaler = joblib.load(SCALER_PATH)
        self.encoders = joblib.load(ENCODER_PATH)
        self.lk, self.tree = build_static_lookup(LOOKUP_CSV)
        tcfg = ncm_io.load_train_config(RUN_DIR)
        self.model, _, _ = ncm_io.build_model(
            MODEL_DIR, MODEL_PATH, self.encoders, CACHE_DIR,
            latlon={}, device=self.device, tcfg=tcfg)
        self.window = int(tcfg["window"])
        self.lead = int(tcfg["lead"])
        # refresh_tile_store MUTATES the projector store; serialize the tile-refresh +
        # forward critical section (as serve_api does with _MODEL_LOCK).
        self._lock = threading.Lock()
        print(f"[advisory.engine] ready — model loaded (window={self.window} lead={self.lead}).")

    # ---- per-point data gather (I/O; parallelizable, reuses serve code verbatim) ----
    def _gather_point(self, lat, lon, target, cache_dir, prefetched_history_df=None):
        """fetch_all + walk-back tile + static profile for ONE point. Identical to
        serve_api's steps 1-2; only the forward is batched later. prefetched_history_df
        (M2) skips this point's own GEE fetch and reuses the batched rows — every
        downstream step in fetch_all is unchanged. Returns a dict or None if this point
        has no usable data (it is then dropped from the batch)."""
        from ndvi_core import download as ncd
        from infer_cli import nearest_static_profile
        from shared_data_layer import fetch_all

        try:
            shared = fetch_all(lat=lat, lon=lon, target_date=target,
                               corestack_key=self.corestack_key,
                               years_lookback=YEARS_LOOKBACK, lead_fortnights=self.lead,
                               prefetched_history_df=prefetched_history_df)
        except Exception as e:
            print(f"[advisory.engine] fetch_all failed ({lat},{lon}): {e}")
            return None
        if any("ERROR:" in q for q in shared.quality_issues):
            print(f"[advisory.engine] data-fetch ERROR ({lat},{lon}): {shared.quality_issues}")
            return None

        lid = ncd.loc_id(lat, lon)
        try:
            _, _, tyear, tq = ncd.fetch_walkback_tile(lid, lat, lon, target, cache_dir, lag=1)
        except Exception as e:
            print(f"[advisory.engine] tile fetch failed ({lat},{lon}): {e}")
            return None
        prof = nearest_static_profile(lat, lon, lk=self.lk, tree=self.tree)
        return {"lat": lat, "lon": lon, "lid": lid, "shared": shared,
                "prof": prof, "tyear": int(tyear), "tq": tq}

    # ---- the batched forward (the one genuinely new piece) ----
    def _forecast_batch(self, gathered, target):
        """gathered: list of _gather_point dicts (order preserved). Runs ONE batched
        model() over all points and returns per-point (ndvi, z, indicators). Each
        point keeps its own window + emb_idx -> batch != share."""
        import numpy as np
        import torch
        from ndvi_core import model_io as ncm_io
        from ndvi_core import inference as ncf_inf
        from ndvi_core import tensors as nct
        from ndvi_core import scaling as ncs
        from ndvi_core import indicators as ncf_ind

        cache_dir = gathered[0]["_cache"]
        latlon = {g["lid"]: (g["lat"], g["lon"]) for g in gathered}

        x_dyn, x_sn, x_sc, emb_idx = [], [], [], []
        # tile-refresh mutates the projector store, then we forward against it: one
        # critical section (all 13 tiles encoded once; each point keeps its own emb_idx).
        with self._lock:
            key_to_row, zero_idx = ncm_io.refresh_tile_store(self.model, cache_dir, latlon)
            for g in gathered:
                win, static_num, cat_ids = ncf_inf.build_window(
                    g["shared"], g["lid"], g["lat"], g["lon"], g["prof"],
                    self.scaler, self.encoders, self.window)
                xd, xsn, xsc = nct.single_window_tensors(win, static_num, cat_ids)
                x_dyn.append(xd); x_sn.append(xsn); x_sc.append(xsc)
                emb_idx.append(key_to_row.get((str(g["lid"]), int(g["tyear"]), g["tq"]), zero_idx))

            x_dyn = np.concatenate(x_dyn, axis=0)   # [N, window, n_dyn]
            x_sn  = np.concatenate(x_sn,  axis=0)   # [N, n_static_num]
            x_sc  = np.concatenate(x_sc,  axis=0)   # [N, 3]
            with torch.no_grad():
                pred, _ = self.model(
                    torch.tensor(x_dyn, device=self.device),
                    torch.tensor(x_sn,  device=self.device),
                    torch.tensor(x_sc,  device=self.device),
                    torch.tensor(emb_idx, device=self.device))
            pred = pred.detach().float().cpu().reshape(-1).tolist()

        out = []
        for g, p in zip(gathered, pred):
            ndvi = ncs.inverse_ndvi(self.scaler, float(p))
            ind = ncf_ind.vegetation_indicators(
                ndvi, g["shared"].history_df, target, prof=g["prof"],
                month_window=0, years=None, bias=0.0)
            out.append({"ndvi": ndvi, "z": ind.get("standardized_anomaly"), "ind": ind})
        return out

    # ---- M2: one batched GEE fetch for all points (mirrors fetch_all's GEE window) ----
    def _batched_raw(self, all_points, target, cache_dir=None):
        """ONE _fetch_gee_raw_batch for every point. Returns a list aligned to
        all_points (each = that point's history_df, or None on failure -> per-point
        fetch_all fallback for that point)."""
        from shared_data_layer import _fetch_gee_raw_batch
        from ndvi_core import download as ncd
        anchor_str, gee_steps = self._gee_window(target)
        try:
            by_id = _fetch_gee_raw_batch(all_points, anchor_str, gee_steps)
        except Exception as e:
            if ncd.is_rate_limit(e):                     # throttle -> record for the warning
                ncd._note_throttle(cache_dir, f"batched numerical fetch: {str(e)[:80]}")
            print(f"[advisory.engine] batched fetch failed ({e}); per-point fallback")
            return [None] * len(all_points)
        return [by_id.get(k) for k in range(len(all_points))]

    def _gee_window(self, target):
        """(anchor_str, gee_steps) for the batched fetch — mirrors fetch_all's GEE
        window math so the batched rows align with the per-point path."""
        import pandas as pd
        from shared_data_layer import _LEAD_FORTNIGHTS
        today = pd.Timestamp.now().normalize()
        prediction_dt = pd.to_datetime(target)
        days_ahead = (prediction_dt - today).days
        steps = int(YEARS_LOOKBACK * 26) + _LEAD_FORTNIGHTS
        if days_ahead <= 0:
            return prediction_dt.strftime("%Y-%m-%d"), steps
        return today.strftime("%Y-%m-%d"), max(steps - (days_ahead // 14), 1)

    # ---- Level 2: REAL clear-view fraction over the recent window (off the fetch result) ----
    def _clear_view(self, shared):
        """Fraction of CLEAR satellite views in the RECENT lookback window, read off
        the fetch result — replaces the season-prior estimate. Level 2 = MOD13Q1
        pixel reliability (SummaryQA <= 1 == good/marginal == clear); this actually
        reflects cloud, unlike is_real_ndvi (availability), which is the Level-1
        fallback when SummaryQA is absent (e.g. the M1 parallel path)."""
        try:
            import pandas as pd
            h = shared.history_df.copy()
            h["date"] = pd.to_datetime(h["date_str"])
            h = h[h["date"] <= pd.to_datetime(shared.data_cutoff_dt)].sort_values("date")
            recent = h.tail(self.window)
            if len(recent):
                if "SummaryQA" in recent.columns and recent["SummaryQA"].notna().any():
                    return round(float((recent["SummaryQA"] <= 1).mean()), 3)   # Level 2
                if "is_real_ndvi" in recent.columns:
                    return round(float(recent["is_real_ndvi"].mean()), 3)        # Level-1 fallback
        except Exception:
            pass
        cp = getattr(shared, "coverage_pct", None)
        return None if cp is None else round(cp / 100.0, 3)

    # ---- recent-trend CONTEXT (data-derived; never changes the rule row) ----
    def _recent_trend(self, shared):
        """'recovering' | 'holding' | 'declining' | 'unknown' from the plot's OWN recent
        observed NDVI vs its per-month normal (anomaly = NDVI - seasonal_baseline). Uses
        the last real NDVI readings before the anchor; CONTEXT ONLY (the rule engine
        never lets trend change the row). Any failure -> 'unknown' (line simply omitted)."""
        try:
            import pandas as pd
            from ndvi_core import indicators as ncf_ind
            from . import trend as trend_mod
            h = shared.history_df.copy()
            h["date"] = pd.to_datetime(h["date_str"])
            h = h[h["date"] <= pd.to_datetime(shared.data_cutoff_dt)]
            h = h[h["NDVI"].notna() & (h["NDVI"] != -9999)].sort_values("date")
            anomalies = []
            for _, row in h.tail(6).iterrows():          # last ~6 real readings, oldest->newest
                d = row["date"].strftime("%Y-%m-%d")
                mean, _, _ = ncf_ind.seasonal_baseline(h, d, month_window=0, years=None)
                if pd.notna(mean):
                    anomalies.append(float(row["NDVI"]) - float(mean))
            return trend_mod.classify(anomalies)
        except Exception:
            return "unknown"

    # ---- the public call ----
    def advise(self, lat, lon, regime, date, *, lang="English", use_slm=None,
               min_points=None):
        """In-process, batched end-to-end advisory for one plot. Structured output is
        identical to the HTTP advisory (same downstream), with a REAL clear-view read."""
        from ndvi_core import dates as ncdates

        _, _, target_dt = ncdates.resolve_forecast_dates(date, lead=self.lead)
        target = target_dt.strftime("%Y-%m-%d")

        try:
            meta, points = sampling.locate_and_sample(lat, lon, key=self.corestack_key,
                                                      min_points=min_points)
        except sampling.OutOfCoverage as e:
            return _adv.build_out_of_coverage(lat, lon, e.state, e.district, e.tehsil, e.nearest_km)
        all_points = [(lat, lon)] + list(points)      # plot first, then sampled points

        cache_dir = os.path.join(CACHE_DIR, f"adv_{uuid4().hex[:8]}")
        os.makedirs(cache_dir, exist_ok=True)
        try:
            # M2: one batched GEE fetch for all points (default). Per point we still
            # fetch its own tile + profile in parallel (I/O). prefetched=None per point
            # falls back to that point's own fetch_all (M1 path).
            prefetched = (self._batched_raw(all_points, target, cache_dir)
                          if FETCH_MODE == "batched" else [None] * len(all_points))
            args = list(zip(all_points, prefetched))
            with ThreadPoolExecutor(max_workers=min(FETCH_WORKERS, len(all_points))) as pool:
                gathered = list(pool.map(
                    lambda a: self._gather_point(a[0][0], a[0][1], target, cache_dir, a[1]), args))
            # plot must be present; drop failed sampled points
            if gathered[0] is None:
                raise RuntimeError("plot forecast unavailable (no usable data at the plot)")
            for g in gathered:
                if g is not None:
                    g["_cache"] = cache_dir
            live = [g for g in gathered if g is not None]

            fc = self._forecast_batch(live, target)
        finally:
            if not KEEP_TILES:
                shutil.rmtree(cache_dir, ignore_errors=True)

        from ndvi_core import download as ncd
        throttles = ncd.pop_throttle_events(cache_dir)   # rate-limit events during the concurrent tile pull
        dropped = len(all_points) - len(live)

        by_id = {id(g): r for g, r in zip(live, fc)}
        plot_res = by_id[id(gathered[0])]
        point_zs = [by_id[id(g)]["z"] for g in gathered[1:]
                    if g is not None and by_id[id(g)]["z"] is not None]
        if not point_zs:
            raise RuntimeError("no point forecasts returned")

        clear = self._clear_view(gathered[0]["shared"])     # 1b: real, off the plot fetch
        trend = self._recent_trend(gathered[0]["shared"])   # recent-trend CONTEXT (data-derived)

        adv = _adv.build_advisory_from_facts(
            plot_z=plot_res["z"], point_zs=point_zs, regime=regime,
            target_month=int(target[5:7]),
            lat=lat, lon=lon, village=meta.get("village"), district=meta.get("district"),
            tehsil=meta.get("tehsil"), target_date=target,
            clear_view_fraction=clear, trend=trend, lang=lang, use_slm=use_slm,
            conf_cap=(C.VEG_FALLBACK_CONF_CAP if meta.get("veg_fallback") else None))
        adv["derived"]["small_sample"] = meta.get("small_sample", adv["derived"]["small_sample"])
        adv["sampling"] = meta
        adv["derived"]["clear_view_source"] = "real-recent-window"   # was: season-prior estimate
        if meta.get("proxy"):
            adv["message"] += _adv.proxy_note(meta)
        elif meta.get("veg_fallback"):
            adv["message"] += _adv.veg_note(meta)
        if throttles:                                    # operator-facing rate-limit notice (NOT in the farmer message)
            suggested = max(2, FETCH_WORKERS // 2)
            w = (f"Earth Engine rate-limited {len(throttles)} of {len(all_points)} concurrent "
                 f"satellite-tile download(s) this request (retried with backoff).")
            if dropped:
                w += f" {dropped} sampled point(s) were dropped, reducing spatial coverage."
            w += (f" If this recurs, lower ADVISORY_FETCH_WORKERS (currently {FETCH_WORKERS}) — "
                  f"e.g. to {suggested} — and restart the advisory.")
            adv["warnings"] = [w]
        return adv
