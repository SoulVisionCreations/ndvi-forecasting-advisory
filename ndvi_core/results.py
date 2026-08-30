"""
ndvi_core.results — the canonical forecast-result dict (C2).

ONE builder shared by infer_cli (which pretty-prints it) and serve_api (which
returns it as JSON), so the CLI and the API can never present different fields.

DEFAULT payload (minimal):
    location{lat,lon} · admin{state,district,tehsil} · target_date · forecast_ndvi ·
    relative_vegetation_condition · anomaly{z, baseline_mean, baseline_std}

VERBOSE payload (--verbose CLI / ?debug=true API) additionally exposes the as-of
and anchor dates, the Prithvi tile, the baseline source, the baseline sample size
(baseline_n), the raw anomaly + percent (ndvi, pct), and the staleness `note` --
all otherwise internal.
"""


def build_result(lat, lon, admin, forecast_from, anchor, target,
                 forecast_ndvi, ind, tile, verbose=False,
                 stale_fortnights=0, last_obs=None):
    """Assemble the canonical result dict.

    ind  : the dict returned by ndvi_core.indicators.vegetation_indicators.
    tile : {"quarter": str, "year": int, "used": bool} (Prithvi walk-back tile).
    stale_fortnights / last_obs : from ndvi_core.inference.anchor_staleness -- if the
      anchor's NDVI had to be ffilled (recent imagery unavailable), a human `note` plus
      the structured values surface in verbose (?debug) only.
    """
    out = {
        "location": {"lat": lat, "lon": lon},
        "admin": admin,
        "target_date": target,
        "forecast_ndvi": round(float(forecast_ndvi), 3),
        "relative_vegetation_condition": ind["relative_vegetation_condition"],
        "anomaly": {
            "z":    ind["standardized_anomaly"],
            "baseline_mean": ind["seasonal_mean_ndvi"],
            "baseline_std":  ind["seasonal_std_ndvi"],
            # ndvi (raw anomaly), pct, and baseline_n are diagnostics -> verbose-only (added below).
        },
    }
    if verbose:
        out["as_of"] = forecast_from
        out["anchor_date"] = anchor
        out["prithvi_tile"] = tile
        out["anomaly"]["baseline_source"] = ind.get("baseline_source")
        out["anomaly"]["baseline_n"] = ind["baseline_n"]   # sample size behind mean/std
        out["anomaly"]["ndvi"] = ind["ndvi_anomaly"]        # raw anomaly (forecast - baseline_mean)
        out["anomaly"]["pct"] = ind.get("ndvi_anomaly_pct")
        out["anchor_stale_fortnights"] = stale_fortnights
        out["last_observation"] = last_obs
        # Staleness message (verbose-only): flags that the forecast is anchored on a
        # carried-forward NDVI observation (recent imagery unavailable). Nothing can be
        # done about stale data; kept out of the default payload to keep it clean.
        if stale_fortnights and stale_fortnights >= 1 and last_obs:
            out["note"] = (f"Anchored on NDVI from {last_obs}, ~{stale_fortnights} "
                           f"fortnight(s) before the {anchor} anchor -- more recent "
                           f"imagery was unavailable (cloud/latency), so the last "
                           f"observation was carried forward.")
    return out
