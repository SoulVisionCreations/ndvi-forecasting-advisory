"""
advisory.advisory — the orchestrator. Wires the 7 steps into one call:

  locate village + sample points  (sampling)
        -> forecast plot + points  (forecast_client -> the black-box /forecast)
        -> aggregate               (aggregate)
        -> data quality            (data_quality)
        -> recent trend, context   (trend)
        -> rule engine             (rule_engine)
        -> phrase                  (phraser: template, or leashed Gemma-4)

Two entry points:
  build_advisory(...)            live: hits CoreStack + /forecast.
  build_advisory_from_facts(...) offline: you supply plot_z + point z's (tests/demo).
"""
from . import aggregate, config as C, data_quality, forecast_client, phraser, rule_engine, sampling


def build_advisory_from_facts(plot_z, point_zs, regime, target_month, *,
                              lat=0.0, lon=0.0, village=None, district=None, tehsil=None,
                              target_date=None, clear_view_fraction=None,
                              recent_anomalies=None, trend=None, precip_normals=None,
                              fallow=False, lang="English", use_slm=None, rules=None, conf_cap=None):
    """Build an advisory from already-computed forecast z's. No network. This is the
    testable core — build_advisory() just fills these in from live services.
    conf_cap (e.g. "medium") caps the headline confidence before the row match — used by the
    non-cropland WorldCover veg-fallback so an out-of-domain read never claims high confidence."""
    from . import trend as trend_mod
    area = aggregate.summarise(point_zs, plot_z)
    facts = {
        "profile": {"regime": regime},
        "location": {"lat": lat, "lon": lon, "village": village, "district": district,
                     "tehsil": tehsil, "target_date": target_date,
                     "target_month": target_month, "precip_normals": precip_normals},
        "plot": {"z": plot_z},
        "area": area,
        "data": {"clear_view_fraction": clear_view_fraction},
        "trend": trend if trend is not None else trend_mod.classify(recent_anomalies),
        "fallow": fallow,
        "conf_cap": conf_cap,
    }
    adv = rule_engine.evaluate(facts, rules)
    adv["message"] = phraser.phrase(adv, lang=lang, use_slm=use_slm)
    return adv


def build_out_of_coverage(lat, lon, state, district, tehsil, nearest_km=None):
    """Graceful response when no cropland tile is within range — never a hard failure.
    Same shape as a normal advisory so callers handle it uniformly (risk_level/confidence/message)."""
    where = ", ".join(x.title() for x in (tehsil, district) if x) or "this area"
    km_txt = f" (nearest mapped cropland is ~{round(nearest_km)} km away)" if nearest_km else ""
    return {
        "input": {"regime": None, "lat": lat, "lon": lon, "village": None,
                  "district": district, "tehsil": tehsil, "target_date": None},
        "derived": {"effective_confidence": "n/a", "out_of_coverage": True},
        "matched": {"risk_level": "out-of-coverage"},
        "trend_context": "unknown",
        "message": (f"{where} isn't mapped for the vegetation outlook yet{km_txt} — there's no "
                    "reliable local read here. Please check with your Krishi Vigyan Kendra (KVK)."),
    }


def proxy_note(meta):
    """The bullet appended when a nearby tile stood in for an unmapped one."""
    tile = str(meta.get("proxy_tile", "")).replace("_", " ").title()
    return (f"\n• Note: your exact area isn't mapped yet — this uses the nearest mapped cropland "
            f"({tile}, ~{meta.get('proxy_km')} km) as a nearby proxy; treat it as an area read.")


def veg_note(meta):
    """The bullet appended when NO cropland was found and nearby VEGETATED land stood in (WorldCover)."""
    km = meta.get("veg_km")
    km_txt = f" (within ~{km} km)" if km else ""
    return (f"\n• Note: no mapped cropland near here — this reads NEARBY VEGETATION{km_txt} as a general "
            "area signal (not crop-specific), so treat it as a lower-confidence area read.")


def build_advisory(lat, lon, regime, date, *, forecast_url=None, corestack_key=None,
                   lang="English", use_slm=None, min_points=None):
    """Live end-to-end advisory for one plot. Requires the /forecast service and
    CoreStack reachable."""
    try:
        meta, points = sampling.locate_and_sample(lat, lon, key=corestack_key,
                                                   min_points=min_points)
    except sampling.OutOfCoverage as e:
        return build_out_of_coverage(lat, lon, e.state, e.district, e.tehsil, e.nearest_km)

    plot = forecast_client.extract(
        forecast_client.forecast(lat, lon, date, base_url=forecast_url))
    point_zs = []
    for (plat, plon) in points:
        r = forecast_client.extract(
            forecast_client.forecast(plat, plon, date, base_url=forecast_url))
        if r["z"] is not None:
            point_zs.append(r["z"])
    if not point_zs:
        raise RuntimeError("no point forecasts returned")

    target_month = plot["target_month"]
    clear = data_quality.estimate_clear_view_fraction(
        target_month, stale_fortnights=plot.get("stale_fortnights"))

    adv = build_advisory_from_facts(
        plot_z=plot["z"], point_zs=point_zs, regime=regime, target_month=target_month,
        lat=lat, lon=lon, village=meta.get("village"), district=meta.get("district"),
        tehsil=meta.get("tehsil"), target_date=plot["target_date"],
        clear_view_fraction=clear, lang=lang, use_slm=use_slm,
        conf_cap=(C.VEG_FALLBACK_CONF_CAP if meta.get("veg_fallback") else None))
    adv["derived"]["small_sample"] = meta.get("small_sample", adv["derived"]["small_sample"])
    adv["sampling"] = meta
    if meta.get("proxy"):
        adv["message"] += proxy_note(meta)
    elif meta.get("veg_fallback"):
        adv["message"] += veg_note(meta)
    return adv
