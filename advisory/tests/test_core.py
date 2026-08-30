"""
Pure-logic tests for the advisory layer — no network, no model. Run either with
pytest, or directly:  python -m advisory.tests.test_core
"""
from advisory import aggregate, data_quality, phraser, rule_engine, season_lens


# MURLI-like sample (median ~ -2.5, population SD ~ 0.57 -> medium)
# 12 points (= MIN_POINTS, matching the live MURLI read; small_sample is False). Same
# distribution as the original 6-point fixture, so median/spread/pct_below are unchanged.
MURLI_PTS = [-1.4, -2.1, -2.4, -2.6, -3.0, -3.1, -1.4, -2.1, -2.4, -2.6, -3.0, -3.1]


def _murli_facts(clear=0.6, month=9, lat=20.17, lon=78.32):
    return {
        "profile": {"regime": "rain-fed"},
        "location": {"lat": lat, "lon": lon, "village": "MURLI",
                     "district": "Yavatmal", "tehsil": "Ralegaon",
                     "target_date": "2018-09-13", "target_month": month},
        "plot": {"z": -2.87},
        "area": aggregate.summarise(MURLI_PTS, -2.87),
        "data": {"clear_view_fraction": clear},
        "trend": "recovering",
    }


def test_severity_bins():
    assert aggregate.severity(-2.5) == "well_below"
    assert aggregate.severity(-1.5) == "below"
    assert aggregate.severity(0.0) == "normal"
    assert aggregate.severity(1.2) == "above"


def test_spatial_confidence_is_population_sd():
    assert aggregate.spatial_confidence(0.3) == "high"
    assert aggregate.spatial_confidence(0.7) == "medium"
    assert aggregate.spatial_confidence(1.2) == "low"


def test_attribution():
    assert aggregate.attribution(-2.0, 100, -2.0) == "regional"
    assert aggregate.attribution(-2.0, 10, -0.1) == "local"     # plot bad, area fine
    assert aggregate.attribution(0.3, 0, 0.2) == "none"


def _tight_facts(conf_cap=None, clear=0.9):
    """A read that WOULD be high confidence (points identical -> agreement high, data good)."""
    pts = [-2.5] * 12
    return {
        "profile": {"regime": "rain-fed"},
        "location": {"lat": 20.17, "lon": 78.32, "village": None, "district": "X", "tehsil": "Y",
                     "target_date": "2018-09-13", "target_month": 9},
        "plot": {"z": -2.5}, "area": aggregate.summarise(pts, -2.5),
        "data": {"clear_view_fraction": clear}, "trend": "unknown", "conf_cap": conf_cap,
    }


def test_conf_cap_caps_before_row_match_and_flags():
    """The WorldCover veg-fallback caps confidence: it applies BEFORE the row match (so the matched
    row is the softer one) and is flagged in derived.conf_capped."""
    hi = rule_engine.evaluate(_tight_facts(conf_cap=None))
    assert hi["derived"]["effective_confidence"] == "high"
    assert hi["derived"]["conf_capped"] is False
    capped = rule_engine.evaluate(_tight_facts(conf_cap="medium"))
    assert capped["derived"]["effective_confidence"] == "medium"
    assert capped["derived"]["conf_capped"] is True
    # the cap changed the DECISION: a different (softer) rule row than the uncapped high read.
    assert capped["matched"]["active_advisory"] != hi["matched"]["active_advisory"]


def test_aggregate_murli():
    a = aggregate.summarise(MURLI_PTS, -2.87)
    assert a["severity"] == "well_below"
    assert a["spatial_confidence"] == "medium"
    assert a["attribution"] == "regional"
    assert a["pct_below"] == 100


def test_murli_matches_well_below_prepare():
    """The bug the team caught: median -2.5 must be reported as WELL BELOW, and the
    matched row must be an act/prepare row — not softened to 'a little below'."""
    adv = rule_engine.evaluate(_murli_facts())
    d = adv["derived"]
    assert d["severity"] == "well_below"
    assert d["effective_confidence"] == "medium"
    assert adv["matched"]["risk_level"] == "below-usual"
    assert adv["matched"]["action_tier"] == "low_regret+prepare"
    # trend is context only — it did NOT downgrade the row
    assert adv["trend_context"] == "recovering"
    msg = phraser.render_template(adv)
    assert "well below" in msg.lower()
    assert "drought" not in msg.lower()


def test_data_poor_gates_to_watch():
    adv = rule_engine.evaluate(_murli_facts(clear=0.3))   # poor data quality
    d = adv["derived"]
    assert d["data_quality"] == "poor"
    assert d["data_poor_gate"] is True
    assert d["effective_confidence"] == "low"
    assert adv["matched"]["risk_level"] == "watch"
    assert "limited clear satellite views" in phraser.render_template(adv).lower()


def test_wet_dry_lens_flips_by_location_same_month():
    # November: Maharashtra (SW belt) = DRY ; Tamil Nadu delta (NE belt) = WET
    mh = season_lens.local_wet_dry(20.17, 78.32, 11)
    tn = season_lens.local_wet_dry(10.85, 79.10, 11)
    assert mh == "dry"
    assert tn == "wet"
    # and the driver flips accordingly
    assert "D2" in season_lens.apply("dry")["driver"]
    assert "D1" in season_lens.apply("wet")["driver"]


def test_overall_confidence_is_worse_of_two():
    assert data_quality.overall_confidence("high", "fair") == "medium"
    assert data_quality.overall_confidence("medium", "good") == "medium"
    assert data_quality.overall_confidence("high", "poor") == "low"


def test_phraser_rejects_invented_numbers_and_words():
    adv = rule_engine.evaluate(_murli_facts())
    assert phraser.validate(phraser.render_template(adv), adv) is True
    assert phraser.validate("expect a 45% yield loss", adv) is False   # invented number
    assert phraser.validate("a drought is coming", adv) is False       # forbidden word


def test_phraser_rejects_second_opinion_label():
    """Positioning: the message must READ as an opinion ('one opinion') but never use the
    literal label 'second opinion'. Same faithful base — only the label differs -> rejected."""
    adv = rule_engine.evaluate(_murli_facts())              # well_below, NOT gated
    assert phraser.validate("may run well below usual; one opinion", adv) is True
    assert phraser.validate("may run well below usual; a second opinion", adv) is False
    assert phraser.validate("may run well below usual; a 2nd opinion", adv) is False


def test_phraser_enforces_magnitude_fidelity():
    """The team's #1 concern, applied to the SLM: a confident well_below read must not
    be softened to just 'below' / 'a little below'."""
    adv = rule_engine.evaluate(_murli_facts())              # well_below, NOT gated
    assert phraser.validate("may run well below usual; one opinion", adv) is True
    assert phraser.validate("may run a little below usual; one opinion", adv) is False
    # gated (data poor) -> soft framing is intended, so the magnitude rule is skipped
    gated = rule_engine.evaluate(_murli_facts(clear=0.3))
    assert gated["derived"]["data_poor_gate"] is True
    assert phraser.validate("please watch closely; a low-confidence read", gated) is True


def test_classify_error_maps_upstream_and_input():
    from advisory.serve_advisory import _classify_error
    assert _classify_error(Exception("HTTPSConnectionPool: NameResolutionError('Failed to resolve geoserver.core-stack.org')"))[0] == 503
    assert _classify_error(ConnectionRefusedError("Connection refused"))[0] == 503
    assert _classify_error(ValueError("date '2035-06-15' is in the future; it must be today or earlier."))[0] == 422
    assert _classify_error(ValueError("Invalid date 'x' -- expected format YYYY-MM-DD."))[0] == 422
    assert _classify_error(Exception("HTTP Error 404: Not Found"))[0] == 422
    assert _classify_error(RuntimeError("plot forecast unavailable (no usable data at the plot)"))[0] == 422
    assert "corestack_key" in _classify_error(Exception("HTTP Error 401: Unauthorized"))[1].lower()
    assert _classify_error(RuntimeError("boom"))[0] == 502


def test_rate_limit_detection_and_registry():
    from ndvi_core import download as ncd
    # detection
    for m in ("HTTP 429 Too Many Requests", "User rate limit exceeded",
              "RESOURCE_EXHAUSTED: quota exceeded"):
        assert ncd.is_rate_limit(m), m
    for m in ("HTTP 404 not found", "connection reset", "some other error"):
        assert not ncd.is_rate_limit(m), m
    # request-scoped registry: per-dir accumulation, pop clears, isolation, missing-key
    ncd._note_throttle("/t/adv_a", "one"); ncd._note_throttle("/t/adv_a", "two")
    ncd._note_throttle("/t/adv_b", "other")
    ncd._note_throttle(None, "ignored")                       # falsy dir = no-op
    assert len(ncd.pop_throttle_events("/t/adv_a")) == 2
    assert ncd.pop_throttle_events("/t/adv_a") == []          # cleared after pop
    assert len(ncd.pop_throttle_events("/t/adv_b")) == 1
    assert ncd.pop_throttle_events("/t/never") == []


def test_preflight_paths_flags_missing_artifacts():
    import os, tempfile
    from advisory import engine
    saved = {k: getattr(engine, k) for k in
             ("RUN_DIR", "MODEL_DIR", "MODEL_PATH", "SCALER_PATH", "ENCODER_PATH", "LOOKUP_CSV")}
    try:
        d = tempfile.mkdtemp()
        files = {"MODEL_PATH": "tft_temporal_production_ft.pt",
                 "SCALER_PATH": "standard_scaler_temporal_tft_ft.pkl",
                 "ENCODER_PATH": "label_encoders_temporal_tft_ft.pkl",
                 "LOOKUP_CSV": "mws_static_lookup_UNSCALED.tsv"}
        for name in list(files.values()) + ["config.json", "prithvi_mae.py"]:
            open(os.path.join(d, name), "w").close()
        engine.RUN_DIR = engine.MODEL_DIR = d
        for attr, name in files.items():
            setattr(engine, attr, os.path.join(d, name))
        assert engine.preflight_paths() == []                     # all present -> clean
        os.remove(engine.MODEL_PATH)                              # drop the fused bundle
        probs = engine.preflight_paths()
        assert len(probs) == 1 and "MISSING" in probs[0] and "fused .pt" in probs[0], probs
    finally:
        for k, v in saved.items():
            setattr(engine, k, v)


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"\n{len(fns)} tests passed.")


if __name__ == "__main__":
    _run_all()
