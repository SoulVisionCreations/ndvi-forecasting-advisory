"""
Offline demo of the advisory layer — no /forecast, no CoreStack. We hand in the plot &
point z-scores directly (as if the forecaster returned them) so you can see the rule
engine + season lens + phraser end to end.

Run from the repo root:  python -m advisory.examples.advisory_demo

For a LIVE run (needs /forecast on :8001 + CoreStack key), use instead:
    from advisory.advisory import build_advisory
    build_advisory(20.17, 78.32, "rain-fed", "2018-06-15")
"""
from advisory.advisory import build_advisory_from_facts

# one fixed sampled distribution (median ~ -2.5, SD ~ 0.57 -> medium) so only the
# LOCATION + TARGET MONTH change between scenarios — isolating the wet/dry lens.
PTS = [-1.4, -2.1, -2.4, -2.6, -3.0, -3.1]

SCENARIOS = [
    dict(name="A1  MURLI, Maharashtra — target SEPTEMBER (SW monsoon = WET)",
         lat=20.17, lon=78.32, target_month=9, clear_view_fraction=0.6,
         trend="recovering", village="MURLI"),
    dict(name="A2  MURLI, Maharashtra — target NOVEMBER (post-monsoon = DRY)",
         lat=20.17, lon=78.32, target_month=11, clear_view_fraction=0.85,
         trend="holding", village="MURLI"),
    dict(name="B1  Delta, Tamil Nadu — target NOVEMBER (NE monsoon = WET)",
         lat=10.85, lon=79.10, target_month=11, clear_view_fraction=0.6,
         trend="declining", village="Delta village"),
]


def main():
    for s in SCENARIOS:
        adv = build_advisory_from_facts(
            plot_z=-2.87, point_zs=PTS, regime="rain-fed",
            target_month=s["target_month"], lat=s["lat"], lon=s["lon"],
            village=s["village"], target_date=f"2018-{s['target_month']:02d}-13",
            clear_view_fraction=s["clear_view_fraction"], trend=s["trend"])
        d = adv["derived"]
        print("=" * 78)
        print(s["name"])
        print("-" * 78)
        print(f"  severity={d['severity']} point_agreement={d['point_agreement']} "
              f"data={d['data_quality']} overall={d['overall_confidence']} "
              f"attr={d['attribution']}")
        print(f"  season={adv['lens']['season']}  driver={adv['lens']['driver']}")
        print(f"  risk={adv['matched']['risk_level']} tier={adv['matched']['action_tier']}")
        print(f"\n  {adv['message']}\n")
    print("=" * 78)
    print("Note A2 vs B1: same month, same z, same risk decision — opposite driver &")
    print("levers, purely from LOCAL wet/dry. That is the season lens.")


if __name__ == "__main__":
    main()
