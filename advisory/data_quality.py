"""
advisory.data_quality — the SECOND, separate confidence axis: could the satellite
actually SEE the ground, or was it cloudy?

"Do the points agree?" (aggregate.spatial_confidence) and "could the camera see?"
(here) are different questions. Monsoon cloud (Jun-Sep, and the NE-monsoon Oct-Dec in
the SE peninsula) can drive the result more than the model. So we track a clear-view
fraction separately and take the WORSE of the two as the overall confidence.
"""
from . import config as C


def label(clear_view_fraction):
    """clear_view_fraction in [0,1] -> good | fair | poor."""
    if clear_view_fraction is None:
        return "unknown"
    if clear_view_fraction >= C.DATA_GOOD: return "good"
    if clear_view_fraction >= C.DATA_FAIR: return "fair"
    return "poor"


def overall_confidence(spatial_conf, data_quality):
    """Overall confidence = the WORSE of spatial agreement and data quality
    (a chain is as strong as its weakest link)."""
    dq_conf = C.DATA_TO_CONF.get(data_quality)      # None for "unknown" -> ignore
    if dq_conf is None:
        return spatial_conf
    worse = min(C.CONF_ORDER.index(spatial_conf), C.CONF_ORDER.index(dq_conf))
    return C.CONF_ORDER[worse]


def is_poor(data_quality):
    """POOR gates the whole advisory down to watch/low (see rule_engine)."""
    return data_quality == "poor"


# ---------------------------------------------------------------------------
# Estimating the clear-view fraction.
# The precise source is the count of valid (non-cloud, non-gap-filled) NDVI
# observations in the forecaster's lookback window. The deployed /forecast does
# not yet expose that, so until it does we use a documented approximation from
# whatever signals we have (anchor staleness + a season prior). Wiring the exact
# valid-obs count is a follow-up (it needs the forecast service to return it).
# ---------------------------------------------------------------------------

# very rough monsoon-cloudiness prior by month (1 = clear, 0 = cloudy), used only
# as a fallback when we have no per-observation validity from the forecaster.
_SEASON_CLEAR_PRIOR = {
    1: 0.90, 2: 0.90, 3: 0.88, 4: 0.85, 5: 0.80,     # dry / pre-monsoon: clear
    6: 0.55, 7: 0.45, 8: 0.45, 9: 0.55,              # SW monsoon: cloudy
    10: 0.65, 11: 0.65, 12: 0.85,                    # post-/NE-monsoon: mixed
}


def estimate_clear_view_fraction(target_month, stale_fortnights=None):
    """Approximate clear-view fraction. If the forecaster told us how stale the
    anchor NDVI is (ffilled from N fortnights back), fold that in; otherwise fall
    back to the season prior. Clearly heuristic — replace with a true valid-obs
    count when the forecast service exposes one."""
    prior = _SEASON_CLEAR_PRIOR.get(int(target_month), 0.7)
    if stale_fortnights is None:
        return prior
    # each fortnight of staleness knocks the clear-view estimate down a bit
    return round(max(0.0, prior - 0.12 * float(stale_fortnights)), 3)
