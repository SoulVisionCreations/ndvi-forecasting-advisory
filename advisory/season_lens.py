"""
advisory.season_lens — turn the TARGET date + location into the LOCAL wet/dry state,
then pick the stress-mitigation levers that fit that state.

Why not kharif/rabi? Those are calendar labels that are uniform across India but
climatically inconsistent: the SW-monsoon belt is wet in kharif / dry in rabi, but the
NE-monsoon belt (Tamil Nadu, coastal AP, Rayalaseema, S-interior Karnataka) is the
OPPOSITE. So we key on LOCAL wet/dry, derived from the location's rainfall climatology
at the target fortnight — which handles SW-, NE- and bimodal regions automatically.

The model output and the rule-engine DECISION never change with the season. Only the
LEVERS / DRIVER / ESCALATION swap here. All levers are stress mitigation on the STANDING
vegetation — never crop-type or variety choices.
"""

# coarse NE-monsoon-belt bounding box (SE peninsula). A stand-in until we wire real
# per-location rainfall climatology; documented as approximate.
_NE_BELT = dict(lat_min=8.0, lat_max=16.5, lon_min=76.0, lon_max=81.5)
_SW_WET_MONTHS = {6, 7, 8, 9}        # SW monsoon
_NE_WET_MONTHS = {10, 11, 12}        # NE / retreating monsoon (SE peninsula)


def local_wet_dry(lat, lon, target_month, precip_normals=None):
    """Return 'wet' or 'dry' for the target fortnight at this location.

    precip_normals: optional list of 12 monthly rainfall normals for the point. If
    given, the target month is WET when its normal is above the year's mean (i.e. in
    the upper part of the annual cycle) — the correct, general definition. If absent,
    fall back to a coarse SW/NE-belt month rule.
    """
    m = int(target_month)
    if precip_normals and len(precip_normals) == 12:
        mean = sum(precip_normals) / 12.0
        return "wet" if precip_normals[m - 1] > mean else "dry"
    in_ne_belt = (_NE_BELT["lat_min"] <= lat <= _NE_BELT["lat_max"]
                  and _NE_BELT["lon_min"] <= lon <= _NE_BELT["lon_max"])
    wet_months = _NE_WET_MONTHS if in_ne_belt else _SW_WET_MONTHS
    return "wet" if m in wet_months else "dry"


# the lens table (== the appendix in use_case_ndvi.txt). Stress-mitigation levers only.
_LENS = {
    "wet": {
        "driver": "D1 rainfall deficit (+ D3 heat)",
        "escalate_if": "the rains underperform (the monsoon that feeds your area)",
        "levers": [
            "conserve soil moisture (mulch, keep weeds down)",
            "a life-saving irrigation at the critical stage if a dry spell hits",
            "a foliar potassium (anti-stress) spray to ride out short dry spells",
        ],
        "resource": "Krishi Vigyan Kendra (KVK) / Gramin Krishi Mausam Seva (GKMS)",
    },
    "dry": {
        "driver": "D2 water-source depletion (+ D3 heat)",
        "escalate_if": "your water source (canal / borewell / release) draws down",
        "levers": [
            "stretch limited water with micro-irrigation to the critical stage",
            "mulch to cut evaporation and water in the cooler hours",
            "an anti-transpirant / kaolin spray to cut heat stress",
        ],
        "resource": "Krishi Vigyan Kendra (KVK) + Water Users Association (WUA) / canal department",
    },
    # deep dry-season fallow window: the relevance gate (little vegetation to lose)
    "dry_fallow": {
        "driver": "(relevance gate — typically bare/fallow now)",
        "escalate_if": "not applicable — re-check closer to sowing",
        "levers": ["nothing to act on now; re-check nearer your sowing window"],
        "resource": "Krishi Vigyan Kendra (KVK)",
    },
}


def apply(season, fallow=False, regime=None):
    """Return the lens fields (driver / escalate_if / levers / resource).

    regime='irrigated' -> the WATER-SOURCE (D2) levers regardless of season: an irrigated
    farmer's stress mitigation is always managing the SOURCE (micro-irrigate, ration, mulch)
    — never a rain-fed "life-saving irrigation", which would contradict a "source is low"
    read. rain-fed / unspecified -> the season-appropriate levers (wet vs dry). The reported
    lens['season'] is always the true local season."""
    if fallow:
        key = "dry_fallow"
    elif regime == "irrigated":
        key = "dry"                       # water-source management, any season
    else:
        key = season if season in ("wet", "dry") else "dry"
    lens = dict(_LENS[key])
    lens["season"] = "dry_fallow" if fallow else season
    return lens
