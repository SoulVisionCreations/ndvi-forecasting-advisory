"""
advisory.config — all pinned thresholds and endpoints in ONE place.

Every bin edge here is the one written into use_case_ndvi.txt (the team-reviewed spec).
Change a number here, and the whole advisory layer moves with it.
"""
import os

# --- severity: from the AREA'S median z (how far below/above its own normal) ---
#   <= -2 well-below | -2..-1 below | -1..1 normal | >= 1 above
SEVERITY_WELL_BELOW = -2.0
SEVERITY_BELOW      = -1.0
SEVERITY_ABOVE      =  1.0

# --- spatial confidence: POPULATION STANDARD DEVIATION of the per-point z-scores ---
#   (the pinned statistic — NOT IQR, NOT range).  <0.5 high | 0.5-1.0 medium | >=1.0 low
SPATIAL_SD_HIGH = 0.5
SPATIAL_SD_LOW  = 1.0

# --- data quality: fraction of CLEAR/valid satellite views feeding the forecast ---
#   (a SEPARATE axis from spatial agreement).  >=0.75 good | 0.5-0.75 fair | <0.5 poor
DATA_GOOD = 0.75
DATA_FAIR = 0.50

# --- attribution: is a below-normal read AREA-WIDE or just YOUR-PLOT? ---
ATTR_Z          = -1.0   # plot_z / median_z threshold
ATTR_PCT_BELOW  = 50     # % of area points below normal

# a point counts as "below" when its z <= this (matches the prototype)
POINT_BELOW_Z = -1.0

# --- sampling: pinned in the spec ---
MIN_POINTS       = 12    # was 6; 6 is noisy
SAMPLE_FRACTION  = 0.5   # keep cropland grid cells with cropland fraction >= this
BUFFER_KM_MAX    = 5.0   # if the village has < MIN_POINTS, buffer outward up to this

# --- WorldCover vegetated FALLBACK (sampling Tier 3) ---
#   When NO cropland is within reach (the plot's own tehsil tile AND the nearest tile <=50 km both
#   fail), sample nearby VEGETATED land from ESA WorldCover so the SELF-RELATIVE z-score still yields
#   an area read.  It is NON-cropland (and the forecaster is cropland-trained), so the caller FLAGS it
#   as a "general vegetation area read (not crop-specific)" and CAPS its confidence.
WORLDCOVER_ASSET       = "ESA/WorldCover/v200"
WORLDCOVER_VEG_CLASSES = (10, 20, 30, 40, 90, 95)  # tree, shrub, grass, cropland, herb-wetland, mangrove
VEG_SEARCH_KM          = 10.0    # sample vegetated pixels within this radius of the plot
VEG_SAMPLE_PIXELS      = 400     # random pixels to probe (only vegetated ones survive the mask)
VEG_FALLBACK_CONF_CAP  = "medium"  # a non-cropland (OOD) read may never exceed this confidence
VEG_DIST_PENALTY_KM    = 20.0    # merged fallback: add this to each veg distance so NEARBY cropland is
                                 #   preferred (near-ish cropland <=~20km stays a proxy), and local
                                 #   vegetation only beats cropland that is MUCH farther (e.g. 45km+)

# confidence is ordered low < medium < high; "overall = worse of the two"
CONF_ORDER = ["low", "medium", "high"]
# map the data-quality label onto the same ladder so we can take the worse one
DATA_TO_CONF = {"good": "high", "fair": "medium", "poor": "low"}

# --- endpoints ---
FORECAST_URL   = os.environ.get("FORECAST_URL", "http://localhost:8001/forecast")
CORESTACK_WFS  = "https://geoserver.core-stack.org:8443/geoserver/ows"
CORESTACK_ADMIN = "https://geoserver.core-stack.org/api/v1/get_admin_details_by_latlon/"
CORESTACK_KEY  = os.environ.get("CORESTACK_KEY", "")

# --- SLM (the phraser only; the deterministic template is ALWAYS the guarded fallback) ---
#   The rule engine owns every fact; the SLM only rewords the matched row and is leashed by
#   phraser.validate() — any drift -> fall back to the (contradiction-free) template.
#   Model: gemma3:4b (Gemma 3 4B, instruction-tuned, ~3.3 GB) — better instruction-following,
#   aligns with the GWL forecasting use case. (gemma3n:e2b is the previous model; set
#   ADVISORY_SLM_MODEL to override.)
SLM_MODEL   = os.environ.get("ADVISORY_SLM_MODEL", "gemma3:4b")
SLM_URL     = os.environ.get("ADVISORY_SLM_URL", "http://localhost:11434/api/generate")
SLM_ENABLED = os.environ.get("ADVISORY_SLM", "1") == "1"          # default ON -> SLM rewords (template = fallback)
SLM_TEMP    = float(os.environ.get("ADVISORY_SLM_TEMP", "0"))     # 0 = greedy/deterministic (reproducible)

# never allowed in farmer-facing text (positioning: indicator, not a drought alarm)
FORBIDDEN_WORDS = ("drought", "famine", "disaster", "crop failure")
