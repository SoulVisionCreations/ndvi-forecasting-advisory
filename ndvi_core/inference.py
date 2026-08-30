"""
ndvi_core.inference — single-request prediction glue (shared by the CLI and the
batch sanity script, so there is ONE copy of the assemble->feature-eng->scale->
window->forward path). Every transform here calls the SAME ndvi_core functions
training uses; the only deliberate train/serve deltas are documented at the
bottom (INFERENCE vs TRAINING).
"""

import numpy as np
import pandas as pd

from . import config
from . import features as ncf
from . import scaling as ncs
from . import tensors as nct

# Static-profile columns copied from the per-MWS lookup row onto the request frame.
STATIC_PROFILE_COLS = [
    "baseline_ndvi_mean", "baseline_ndvi_std", "sensitivity_spei3",
    "historical_max_ndvi", "peak_month",
    "q1_mean_ndvi", "q2_mean_ndvi", "q3_mean_ndvi", "q4_mean_ndvi",
    "q1_std_ndvi", "q2_std_ndvi", "q3_std_ndvi", "q4_std_ndvi",
]


def scale_forecast_aggregates(scaler, precip_steps_raw, temp_steps_raw):
    """forecast_precip_sum = sum(scaled per-step precip); forecast_temp_mean =
    mean(scaled per-step temp) — mirrors training's forecast aggregates computed
    on the scaled frame (and the deployed _scale_forecast_features)."""
    p = [ncs.scale_value(scaler, "precipitation", v) for v in precip_steps_raw] or [0.0]
    t = [ncs.scale_value(scaler, "temperature_2m", v) for v in temp_steps_raw] or [0.0]
    return float(np.sum(p)), float(np.mean(t))


def build_window(shared, mws_id, lat, lon, prof, scaler, encoders, window=config.WINDOW):
    """Assemble one location's GEE history into the last-`window` scaled rows +
    the static-numeric vector + categorical ids, running the SAME ndvi_core
    feature-eng/scaling as training. Returns (window_df, static_num, cat_ids)."""
    hist = shared.history_df.copy()
    hist["date"] = pd.to_datetime(hist["date_str"])
    hist = hist[hist["date"] <= pd.to_datetime(shared.data_cutoff_dt)].sort_values("date")
    hist["mws_id"] = str(mws_id)
    hist["latitude"] = lat
    hist["longitude"] = lon
    for c in STATIC_PROFILE_COLS:
        hist[c] = prof[c]
    hist["state"] = shared.admin.get("state", "Unknown")
    hist["district"] = shared.admin.get("district", "Unknown")
    hist["tehsil"] = shared.admin.get("tehsil", "Unknown")

    # Serve-side cleaning to MIRROR training (rename_and_clean's -9999->NaN, then
    # apply_hierarchical_imputation): treat the GEE -9999 sentinel as missing and
    # impute the raw dynamic inputs via the SAME shared function training uses
    # (single-point cohort -> per-column median). Without this, serve left dewpoint /
    # radiation / wind gaps as NaN (-> fillna(0) in raw) and passed -9999 sentinels
    # straight into scaling. NDVI is left as-is (handled by upstream ffill; training
    # drops NaN-NDVI rows rather than imputing them).
    _impute_cols = [c for c in (config.WEATHER_COLS + [
        "u_component_of_wind_10m", "v_component_of_wind_10m"])
        if c in hist.columns and c != "NDVI"]
    hist[_impute_cols] = hist[_impute_cols].replace(-9999, np.nan)
    hist = ncf.apply_hierarchical_imputation(hist, weather_cols=_impute_cols)

    # C1 in-place ffill (NDVI). NDVI is NOT imputed upstream (shared_data_layer's
    # ffill excludes it) and training simply DROPS NaN-NDVI windows; serve cannot
    # drop. With the anchor snapped to ~now, the most recent composite is often
    # cloud/latency-empty, so fill the anchor (and any gap) from the previous
    # available reading rather than sliding the anchor back. Serve-only. See (5) below.
    hist["NDVI"] = hist["NDVI"].replace(-9999, np.nan).ffill().bfill()

    hist = ncf.add_time_features(hist)
    hist = ncf.add_engineered_features(hist, rad_clip_limits={})   # see divergence (2)
    hist["months_from_peak"] = hist["month"] - hist["peak_month"]
    hist = ncf.add_ndvi_q_deviation(hist)
    hist = ncf.apply_label_encoders(hist, encoders)

    # C1 final safety: no NaN may reach the tensor (serve can't drop rows the way
    # training does). Fill any residual gaps in the dynamic inputs from neighbours
    # so the last-`window` rows are always complete.
    hist[config.DYNAMIC_FEATURES] = hist[config.DYNAMIC_FEATURES].ffill().bfill()

    scaled = ncs.apply_scaler(hist, scaler)
    win = scaled.tail(window)
    if len(win) < window:
        raise ValueError(f"only {len(win)} history rows (< window {window})")

    base_static = win[config.STATIC_BASE_FEATURES].iloc[-1].values.astype(np.float32)
    f_p, f_t = scale_forecast_aggregates(
        scaler, shared.forecast_precip_steps_raw, shared.forecast_temp_steps_raw)
    static_num = np.concatenate([base_static, [f_p, f_t]]).astype(np.float32)
    cat_ids = win[config.STATIC_CATEGORICAL_FEATURES].iloc[-1].values.astype(np.int64)
    return win, static_num, cat_ids


def forecast_point(model, shared, mws_id, lat, lon, prof, tile_year, tile_quarter,
                   key_to_row, zero_idx, scaler, encoders,
                   window=config.WINDOW, device="cuda"):
    """Full single-request forecast: assemble window -> Prithvi-encode the
    walk-back tile (emb_idx) -> model forward -> inverse-scale NDVI.
    Returns (ndvi, emb_idx)."""
    import torch
    win, static_num, cat_ids = build_window(
        shared, mws_id, lat, lon, prof, scaler, encoders, window)
    x_dyn, x_sn, x_sc = nct.single_window_tensors(win, static_num, cat_ids)
    emb_idx = key_to_row.get((str(mws_id), int(tile_year), tile_quarter), zero_idx)
    with torch.no_grad():
        pred, _ = model(
            torch.tensor(x_dyn, device=device),
            torch.tensor(x_sn, device=device),
            torch.tensor(x_sc, device=device),
            torch.tensor([emb_idx], device=device))
    return ncs.inverse_ndvi(scaler, float(pred.item())), emb_idx


def anchor_staleness(shared, anchor):
    """(stale_fortnights, last_obs_date) for the anchor's NDVI.

    stale_fortnights = fortnights between the anchor and the most recent REAL NDVI
    reading at/before it: 0 = the anchor reading is present (fresh); >0 = the anchor
    NDVI was ffilled from that many fortnights back (see build_window / divergence 5).
    A satellite/latency limit, not a bug -- surfaced so the caller can note it.
    Returns (None, None) if there is no real NDVI at all. `anchor` = date str or datetime.
    """
    h = shared.history_df.copy()
    h["date"] = pd.to_datetime(h["date_str"])
    a = pd.to_datetime(anchor)
    h = h[h["date"] <= a]
    real = h[h["NDVI"].notna() & (h["NDVI"] != -9999)]
    if real.empty:
        return None, None
    last_real = real["date"].max()
    gap = max(int(round((a - last_real).days / 14)), 0)
    return gap, last_real.strftime("%Y-%m-%d")


# ===========================================================================
# INFERENCE vs TRAINING — the deliberate deltas (everything else is identical
# ndvi_core code/config). Keep this list authoritative for the model card.
#   (1) IMPUTATION: training fills weather NaNs by MWS->district->state->global
#       median; inference uses the per-request GEE history (shared_data_layer
#       ffill) — no cohort exists for a single point at serve time. Inherent.
#   (2) RAD-CLIP: training clips 2 radiation cols at the train 99th pct; those
#       limits are not persisted in the bundle, so inference skips the clip
#       (rad_clip={}). Tails of 2/20 dynamic features only. (Fixable: persist limits.)
#   (3) FORECAST STATICS: training uses the ACTUAL lead-window precip/temp;
#       inference uses the Open-Meteo/climatology forecast (future is unknown at
#       serve time) -> the main driver of the observed serve-time low bias.
#   (4) STATIC PROFILE: from mws_static_lookup_UNSCALED.csv, which training
#       EXPORTED from the same computed values -> CONSISTENT by construction.
#   (5) NDVI/DYNAMIC FFILL (C1): serve ffills NDVI (and any residual dynamic-input
#       gap) from the previous reading so the anchor -- snapped to ~now via
#       ndvi_core.dates -- survives a cloud/latency-empty recent composite.
#       Training instead DROPS NaN-NDVI windows; serve must answer, so it imputes
#       in place. Only bites the trailing row(s) when recent data is missing.
# ===========================================================================
