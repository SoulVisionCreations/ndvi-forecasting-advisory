"""
advisory.aggregate — turn the per-point z-scores into the area summary.

Pure functions, no I/O. Every threshold comes from advisory.config, so this module
is the single, testable home of "how the numbers become the labels".
"""
import statistics as st
from . import config as C


def severity(median_z):
    """AREA severity from its median z (distance from its OWN seasonal normal)."""
    if median_z <= C.SEVERITY_WELL_BELOW: return "well_below"
    if median_z <= C.SEVERITY_BELOW:      return "below"
    if median_z <  C.SEVERITY_ABOVE:      return "normal"
    return "above"


def spatial_confidence(sd):
    """How much the sampled points AGREE = population SD of their z-scores.
    Small SD -> tightly agree -> high confidence."""
    if sd < C.SPATIAL_SD_HIGH: return "high"
    if sd < C.SPATIAL_SD_LOW:  return "medium"
    return "low"


def attribution(plot_z, pct_below, median_z):
    """Is the below-normal read AREA-WIDE (regional) or just the farmer's plot (local)?"""
    if plot_z <= C.ATTR_Z and pct_below >= C.ATTR_PCT_BELOW and median_z <= C.ATTR_Z:
        return "regional"
    if plot_z <= C.ATTR_Z:
        return "local"
    return "none"


def summarise(point_zs, plot_z):
    """point_zs: list of per-point z-scores; plot_z: the farmer's own plot z.
    Returns the area summary dict used by the rule engine."""
    n = len(point_zs)
    if n == 0:
        raise ValueError("no sampled points to aggregate")
    below = sum(1 for z in point_zs if z <= C.POINT_BELOW_Z)
    median_z = st.median(point_zs)
    sd = st.pstdev(point_zs) if n > 1 else 0.0          # POPULATION SD (pinned)
    pct_below = round(100 * below / n)
    return {
        "n": n,
        "median_z": round(median_z, 3),
        "sd": round(sd, 3),
        "pct_below": pct_below,
        "severity": severity(median_z),
        "spatial_confidence": spatial_confidence(sd),
        "attribution": attribution(plot_z, pct_below, median_z),
        "small_sample": n < C.MIN_POINTS,
    }
