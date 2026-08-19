"""
advisory.rule_engine — the deterministic brain. Given the aggregated facts it:
  1. reads the pinned derivations (severity / spatial confidence / attribution),
  2. folds in DATA QUALITY as a separate axis (overall confidence = worse of the two;
     POOR data gates the whole thing down to watch/low),
  3. matches exactly ONE row of uc1_advisory_rules.csv,
  4. attaches the season lens (local wet/dry -> driver/levers/escalation),
  5. carries the recent trend as CONTEXT only (never changes the row).
Owns every number and decision. The phraser downstream only rewords the result.
"""
import csv
import os

from . import config as C
from . import data_quality as dq
from . import season_lens

_RULES_PATH = os.path.join(os.path.dirname(__file__), "rules", "uc1_advisory_rules.csv")


def load_rules(path=_RULES_PATH):
    with open(path) as f:
        rows = [ln for ln in f if not ln.lstrip().startswith("#")]
    return list(csv.DictReader(rows))


def _match(rules, attribution, regime, severity, confidence):
    def ok(rule_val, in_val):
        return rule_val == in_val or rule_val == "any"
    for r in rules:
        if (ok(r["attribution"], attribution) and ok(r["regime"], regime)
                and ok(r["severity"], severity) and ok(r["confidence"], confidence)):
            return r
    return None


def _synth_watch(regime):
    """Safety-net row for a state with NO CSV match. After the none+below / none+well_below
    rows were added this is unreachable for real aggregator output, but it stays so we never
    leave the farmer with 'NO RULE MATCHED'; degrade to a neutral watch."""
    return {
        "attribution": "none", "regime": regime, "severity": "any", "confidence": "low",
        "risk_level": "watch", "action_tier": "monitor",
        "active_advisory": "A weak / mixed signal for your area — nothing firm. Keep "
                           "watching and confirm with your Krishi Vigyan Kendra (KVK). "
                           "One opinion only.",
        "escalate_if": "re-check next update", "driver_check": "",
    }


def _merge_regimes(rf, ir):
    """Combine the rain-fed + irrigated advisories (IDENTICAL numbers/lens/trend — only the
    matched row differs) into ONE. If the two rows coincide (regime-agnostic scenario:
    normal / above / check-your-field) -> a single branch. If they differ (area-wide
    below-usual) -> keep both so the phraser presents 'if rain-fed / if irrigated'."""
    adv = dict(rf)                              # shared derived / lens / trend / plot_z
    adv["input"] = {**rf["input"], "regime": None}
    if rf["matched"]["active_advisory"] == ir["matched"]["active_advisory"]:
        adv["regime_split"] = False            # regimes coincide -> single advice
    else:
        adv["regime_split"] = True             # rf["matched"] = rain-fed row; add irrigated
        adv["matched_irrigated"] = ir["matched"]
        adv["lens_irrigated"] = ir["lens"]     # irrigated water-source levers (rf["lens"] is rain-fed)
    return adv


def evaluate(facts, rules=None):
    """facts = {
         profile:{regime}, location:{lat,lon,target_month,precip_normals?,...},
         plot:{z}, area:{...aggregate.summarise output...},
         data:{clear_view_fraction}, trend:'recovering'|..., fallow:bool
       }
    Returns the full, structured advisory (still machine-readable; the phraser turns
    it into text)."""
    rules = rules if rules is not None else load_rules()
    regime = facts["profile"]["regime"]
    if regime is None:                          # unspecified -> evaluate BOTH regimes and merge
        prof = facts["profile"]
        rf = evaluate({**facts, "profile": {**prof, "regime": "rain-fed"}}, rules)
        ir = evaluate({**facts, "profile": {**prof, "regime": "irrigated"}}, rules)
        return _merge_regimes(rf, ir)
    area = facts["area"]
    loc = facts["location"]

    # --- confidence: two separate axes, take the worse; POOR data gates to low ---
    data_label = dq.label(facts.get("data", {}).get("clear_view_fraction"))
    overall_conf = dq.overall_confidence(area["spatial_confidence"], data_label)
    gated = dq.is_poor(data_label)
    effective_conf = "low" if gated else overall_conf
    # a caller may CAP the confidence (e.g. a non-cropland WorldCover veg-fallback read may never
    # exceed "medium"). The cap is applied BEFORE the row match, so the matched row reflects it.
    cap = facts.get("conf_cap")
    conf_capped = bool(cap and C.CONF_ORDER.index(effective_conf) > C.CONF_ORDER.index(cap))
    if conf_capped:
        effective_conf = cap

    # --- the season lens (local wet/dry at the target date) ---
    season = season_lens.local_wet_dry(
        loc["lat"], loc["lon"], loc["target_month"], loc.get("precip_normals"))
    lens = season_lens.apply(season, fallow=bool(facts.get("fallow")), regime=regime)

    # --- match exactly one row ---
    row = _match(rules, area["attribution"], regime, area["severity"], effective_conf)
    if row is None:
        row = _synth_watch(regime)

    return {
        "input": {"regime": regime, **{k: loc.get(k) for k in
                  ("lat", "lon", "village", "district", "tehsil", "target_date")}},
        "derived": {
            "median_z": area["median_z"], "pct_below": area["pct_below"],
            "sd": area["sd"], "n_points": area["n"], "small_sample": area["small_sample"],
            # "point_agreement" (= how tightly the sampled points agree, the population SD
            # band) is a SEPARATE axis from the headline "confidence" (which also folds in
            # data quality). Named distinctly so the two can differ without contradicting.
            "severity": area["severity"], "point_agreement": area["spatial_confidence"],
            "data_quality": data_label, "overall_confidence": overall_conf,
            "effective_confidence": effective_conf, "attribution": area["attribution"],
            "data_poor_gate": gated, "conf_capped": conf_capped,
        },
        "matched": row,
        "lens": lens,
        "trend_context": facts.get("trend", "unknown"),
        "plot_z": facts["plot"]["z"],
    }
