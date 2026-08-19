"""
ndvi_core.indicators — vegetation output indicators from an NDVI forecast.

Turns a raw NDVI forecast (the model's only numeric output) into the
interpretable drought/vegetation-condition product the workflow ships:

    set B  ->  predicted NDVI
               standardized seasonal anomaly (z)  +  percent anomaly
               3-class vegetation-condition label

The anomaly is measured against the point's OWN observed NDVI history for the
forecast month, pooled across prior years. The baseline EXCLUDES the target
year (obs with date < cutoff, default = start of the target year), so a hindcast
fetch — which pulls history up to the target to grab the actual — cannot leak the
value being predicted. Baseline center/spread are ROBUST (median / 1.4826*MAD) so
cloud-contaminated outliers don't wash out the signal, with a small spread floor
(min_std) so anomalies inside NDVI measurement noise aren't flagged extreme. Using
the point's history (rather than the nearest-MWS quarter lookup) makes the
baseline location-exact and month-specific; pooling many years keeps the
mean/std stable and keeps a multi-year drought detectable (a short trailing
window would let "normal" drift down with the drought and hide it). When a
point's history is too sparse for the month, we fall back to the shipped
per-quarter climatology.

Pure module (no torch / no I/O): callable from the CLI, the batch script, or
any serving layer.
"""

import numpy as np
import pandas as pd

# 3-class RELATIVE vegetation-condition thresholds on the standardized anomaly z.
# Labels are anchored on the location's OWN seasonal normal (z vs its month-history
# baseline), NOT absolute greenness -- "Above normal" means greener than typical for
# this point & season, not lush in absolute terms.
# Simplified from 5 -> 3 classes (boundaries only at +/-1) for robustness: the z can be
# sensitive at the pixel x month scale (tight baselines), so fewer boundaries -> far
# fewer adjacent-class flips across nearby points / dates.
CONDITION_BINS = [
    (-np.inf, -1.0, "Below normal"),
    (-1.0,     1.0, "Near normal"),
    (1.0,   np.inf, "Above normal"),
]


def classify_condition(z):
    """Map a standardized anomaly z to the 3-class relative-condition label."""
    if z is None or (isinstance(z, float) and (np.isnan(z) or np.isinf(z))):
        return "Unknown"
    for lo, hi, label in CONDITION_BINS:
        if lo <= z < hi:
            return label
    # z at/above the top edge -> "Well above normal"; below bottom -> "Severe deficit".
    return CONDITION_BINS[-1][2] if z >= CONDITION_BINS[-1][0] else CONDITION_BINS[0][2]


def _history_frame(history_df):
    """Return a copy with a parsed `date` column (accepts `date` or `date_str`)."""
    df = history_df.copy()
    if "date" not in df.columns and "date_str" in df.columns:
        df["date"] = df["date_str"]
    df["date"] = pd.to_datetime(df["date"])
    return df


def seasonal_baseline(history_df, target_date, month_window=0, years=None,
                      min_samples=6, cutoff=None, robust=True):
    """Center/spread of the point's OBSERVED NDVI around the target's time-of-year,
    pooled across prior years from the fetched history.

    month_window : include NDVI from months within +/- this many months of the
                   target month (0 = the forecast month only; 1 = +/-1 month).
    years        : use only the most recent N calendar years present (None=all).
    min_samples  : below this many valid obs, return NaNs (caller falls back).
    cutoff       : baseline uses only obs with date < cutoff. Default (None) =
                   start of the target year, which drops the target-year same-month
                   observation a hindcast fetch would include (no leakage).
    robust       : True (default) -> center = median, spread = 1.4826*MAD (robust to
                   cloud outliers); False -> mean/std.

    Returns (center, spread, n_obs).
    """
    df = _history_frame(history_df)
    if "NDVI" not in df.columns:
        return (np.nan, np.nan, 0)
    t = pd.to_datetime(target_date)

    # no-leakage cut: exclude the target period (and, by default, the whole
    # target year) from the baseline.
    cut = pd.to_datetime(cutoff) if cutoff is not None else pd.Timestamp(t.year, 1, 1)
    df = df[df["date"] < cut]

    tm = t.month
    months = {((tm - 1 + d) % 12) + 1 for d in range(-month_window, month_window + 1)}
    sub = df[df["date"].dt.month.isin(months)]

    if years:
        keep = sorted(sub["date"].dt.year.unique())[-int(years):]
        sub = sub[sub["date"].dt.year.isin(keep)]

    vals = pd.to_numeric(sub["NDVI"], errors="coerce").dropna().values
    if len(vals) < min_samples:
        return (np.nan, np.nan, int(len(vals)))

    if robust:
        center = float(np.median(vals))
        spread = float(1.4826 * np.median(np.abs(vals - center)))  # MAD -> std-equiv
    else:
        center = float(np.mean(vals))
        spread = float(np.std(vals))
    return (center, spread, int(len(vals)))


def _quarter_baseline_from_profile(prof, target_date):
    """Fallback: per-quarter climatological mean/std from the shipped static
    lookup row (q1..q4_{mean,std}_ndvi). Returns (mean, std) or (nan, nan)."""
    if prof is None:
        return (np.nan, np.nan)
    q = ((pd.to_datetime(target_date).month - 1) // 3) + 1
    mean = prof.get(f"q{q}_mean_ndvi", np.nan)
    std = prof.get(f"q{q}_std_ndvi", np.nan)
    try:
        return (float(mean), float(std))
    except (TypeError, ValueError):
        return (np.nan, np.nan)


def vegetation_indicators(forecast_ndvi, history_df, target_date, prof=None,
                          month_window=0, years=None, bias=0.0, cutoff=None,
                          robust=True, min_std=0.05):
    """Set B indicators for one forecast.

    forecast_ndvi : the model's NDVI forecast for the target period.
    history_df    : the point's fetched NDVI history (needs `NDVI` + `date`/`date_str`).
    target_date   : the forecast target date (YYYY-MM-DD) — selects the season.
    prof          : optional static-lookup row, used only as a baseline fallback.
    month_window  : +/- months around the target month for the baseline (0=month).
    years         : most-recent-N-years window for the baseline (None=all).
    bias          : optional additive serve-time bias correction; the corrected
                    forecast (forecast - bias) is what the anomaly is measured on.
                    Default 0.0 (off).
    cutoff        : baseline excludes dates >= cutoff (default None -> start of the
                    target year -> no hindcast leakage).
    robust        : robust (median/MAD) baseline stats. Default True.
    min_std       : floor on the baseline spread used for z, so anomalies within
                    NDVI measurement noise are not flagged extreme. Default 0.05
                    (a realistic NDVI variability floor; was 0.02, which let tight
                    pixel/month baselines make z explode).

    Returns a dict: forecast_ndvi, seasonal_mean_ndvi, seasonal_std_ndvi,
    baseline_n, baseline_source, ndvi_anomaly, ndvi_anomaly_pct,
    standardized_anomaly, relative_vegetation_condition.
    """
    adj = float(forecast_ndvi) - float(bias)

    mean, std, n = seasonal_baseline(history_df, target_date, month_window, years,
                                     cutoff=cutoff, robust=robust)
    source = "point-history"
    if not np.isfinite(mean) or not np.isfinite(std):
        qmean, qstd = _quarter_baseline_from_profile(prof, target_date)
        if np.isfinite(qmean) and np.isfinite(qstd):
            mean, std, source = qmean, qstd, "quarter-climatology"

    if np.isfinite(mean):
        anomaly = adj - mean
        # effective spread used for z (floored at min_std). We report THIS as
        # baseline_std so that z == anomaly / baseline_std reconciles for the caller;
        # the raw MAD spread can be below the floor for very stable pixels/months.
        std_eff = max(float(std), min_std) if np.isfinite(std) else min_std
        z = anomaly / std_eff
        pct = (anomaly / mean * 100.0) if abs(mean) > 1e-6 else np.nan
    else:
        anomaly, z, pct, std_eff, source = np.nan, np.nan, np.nan, np.nan, "unavailable"

    return {
        "forecast_ndvi": round(float(forecast_ndvi), 3),
        "seasonal_mean_ndvi": None if not np.isfinite(mean) else round(mean, 3),
        "seasonal_std_ndvi": None if not np.isfinite(std_eff) else round(std_eff, 3),
        "baseline_n": n,
        "baseline_source": source,
        "ndvi_anomaly": None if not np.isfinite(anomaly) else round(anomaly, 3),
        "ndvi_anomaly_pct": None if not np.isfinite(pct) else round(pct, 1),
        "standardized_anomaly": None if not np.isfinite(z) else round(z, 2),
        "relative_vegetation_condition": classify_condition(z),
    }
