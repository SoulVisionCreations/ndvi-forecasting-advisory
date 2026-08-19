"""
advisory.exactness_harness — prove the BATCHED forward matches the per-point path.

Gate for the in-process engine. It gathers the plot + 12 sampled points ONCE
(fetch_all + walk-back tile), then runs:

  * BASELINE  — per-point ncf_inf.forecast_point (the exact serve_api forward), and
  * BATCHED   — engine._forecast_batch (13 stacked windows, one model()),

over the IDENTICAL combined tile store, so BATCHING is the only variable. It then
builds the full advisory both ways.

ACCEPT (per the agreed gate):
  HARD     — derived labels (severity / confidence / attribution), matched row, and
             the final message are IDENTICAL for every point/advisory.
  TRIPWIRE — |z_batched - z_baseline| <= 1e-3 for every point (catches batch!=share /
             tile-emb_idx contamination, which jumps 0.1+; ignores fp32 ~1e-6 noise).

Run on a GPU host in the repo .venv with CORESTACK_KEY + GEE creds in env:
    python -m advisory.exactness_harness
Exit 0 = PASS, 1 = FAIL.
"""
import os
import sys

os.environ.setdefault("no_proxy", "localhost,127.0.0.1")

from advisory.engine import AdvisoryEngine, CACHE_DIR
from advisory import advisory as adv_mod

# 13 points around MURLI (plot + spread), matching the fetch-batch prototype.
BASE = (20.17, 78.32)
OFFS = [(0, 0), (0.011, 0.004), (-0.009, 0.006), (0.003, -0.012), (-0.007, -0.008),
        (0.016, 0.010), (-0.014, -0.011), (0.020, -0.003), (-0.019, 0.002), (0.006, 0.018),
        (-0.005, -0.017), (0.013, -0.014), (-0.012, 0.015)]
POINTS = [(BASE[0] + dy, BASE[1] + dx) for (dy, dx) in OFFS]
DATE = os.environ.get("HARNESS_DATE", "2018-06-15")     # as-of; target ~ early Sep
TRIP = float(os.environ.get("HARNESS_TRIPWIRE", "1e-3"))


def _labels(a):
    d = a["derived"]
    return (d["severity"], d["effective_confidence"], d["attribution"],
            a["matched"]["risk_level"], a["matched"]["active_advisory"], a.get("message"))


def main():
    import uuid
    from ndvi_core import model_io as ncm_io
    from ndvi_core import inference as ncf_inf
    from ndvi_core import indicators as ncf_ind

    eng = AdvisoryEngine()
    _, _, target_dt = __import__("ndvi_core.dates", fromlist=["dates"]).resolve_forecast_dates(
        DATE, lead=eng.lead)
    target = target_dt.strftime("%Y-%m-%d")
    print(f"[harness] {len(POINTS)} points  as-of={DATE}  target={target}  tripwire={TRIP}")

    cache = os.path.join(CACHE_DIR, f"harness_{uuid.uuid4().hex[:8]}")
    os.makedirs(cache, exist_ok=True)

    # ---- gather all points once (fetch_all + tile) ----
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=len(POINTS)) as pool:
        gathered = list(pool.map(lambda pt: eng._gather_point(pt[0], pt[1], target, cache), POINTS))
    live = [g for g in gathered if g is not None]
    for g in live:
        g["_cache"] = cache
    print(f"[harness] gathered {len(live)}/{len(POINTS)} points with usable data")
    if len(live) < 3:
        print("[harness] FAIL — too few usable points to test"); return 1

    latlon = {g["lid"]: (g["lat"], g["lon"]) for g in live}

    # ---- BASELINE: per-point forecast_point over the combined store ----
    with eng._lock:
        key_to_row, zero_idx = ncm_io.refresh_tile_store(eng.model, cache, latlon)
        base = []
        for g in live:
            ndvi_b, _ = ncf_inf.forecast_point(
                eng.model, g["shared"], g["lid"], g["lat"], g["lon"], g["prof"],
                g["tyear"], g["tq"], key_to_row, zero_idx, eng.scaler, eng.encoders,
                window=eng.window, device=eng.device)
            zb = ncf_ind.vegetation_indicators(
                ndvi_b, g["shared"].history_df, target, prof=g["prof"],
                month_window=0, years=None, bias=0.0)["standardized_anomaly"]
            base.append({"ndvi": ndvi_b, "z": zb})

    # ---- BATCHED: one forward over the same gathered/store ----
    batch = eng._forecast_batch(live, target)

    # ---- compare per point ----
    ok = True
    max_ndvi = 0.0
    max_z = 0.0
    print("\n  idx   (lat,lon)          ndvi_base   ndvi_batch   |dNDVI|     z_base  z_batch")
    for i, g in enumerate(live):
        b, q = base[i], batch[i]
        dn = abs(b["ndvi"] - q["ndvi"])
        dz = abs((b["z"] or 0.0) - (q["z"] or 0.0))
        max_ndvi = max(max_ndvi, dn)
        max_z = max(max_z, dz)
        flag = "" if (dz <= TRIP) else "  <-- TRIPWIRE"
        if dz > TRIP:
            ok = False
        print(f"  [{i:2d}] ({g['lat']:.3f},{g['lon']:.3f})  {b['ndvi']:.6f}   "
              f"{q['ndvi']:.6f}   {dn:.2e}   {str(b['z']):>6}  {str(q['z']):>6}{flag}")
    print(f"\n  max |dNDVI| = {max_ndvi:.2e}   max |dz(2dp)| = {max_z:.2e}   tripwire = {TRIP}")

    # ---- M2: batched-fetch column parity vs per-point _fetch_gee_raw (first 3 pts) ----
    # If the batched fetch reproduces the per-point fetch column-for-column, then
    # fetch_all(prefetched=df) runs identical downstream by construction, so combined
    # with the forward parity above this proves the whole M2 path.
    import pandas as pd
    from shared_data_layer import _fetch_gee_raw, _fetch_gee_raw_batch
    anchor_str, gee_steps = eng._gee_window(target)
    bdfs = _fetch_gee_raw_batch([(g["lat"], g["lon"]) for g in live], anchor_str, gee_steps)
    PARCOLS = ["NDVI", "precipitation", "temperature_2m", "et", "runoff", "sm_rootzone",
               "sm_surface", "dewpoint_temperature_2m", "u_component_of_wind_10m",
               "v_component_of_wind_10m", "surface_solar_radiation_downwards_sum",
               "surface_net_thermal_radiation_sum", "coverage_fraction", "is_real_ndvi"]
    ncheck = min(3, len(live))
    fetch_worst = 0.0
    for k in range(ncheck):
        g = live[k]
        pp = _fetch_gee_raw(g["lat"], g["lon"], anchor_str, gee_steps).sort_values("step").reset_index(drop=True)
        bb = bdfs[k].sort_values("step").reset_index(drop=True)
        m = pp.merge(bb, on="step", suffixes=("_pp", "_b"))
        for c in PARCOLS:
            if f"{c}_pp" in m.columns and f"{c}_b" in m.columns:
                d = (pd.to_numeric(m[f"{c}_pp"], errors="coerce")
                     - pd.to_numeric(m[f"{c}_b"], errors="coerce")).abs().max()
                if pd.notna(d):
                    fetch_worst = max(fetch_worst, float(d))
    has_summ = all("SummaryQA" in bdfs[k].columns for k in range(ncheck))
    fetch_ok = fetch_worst < 1e-6 and has_summ
    print(f"  M2 batched-fetch column parity (first {ncheck} pts): max|diff|={fetch_worst:.2e}  "
          f"SummaryQA present={has_summ}  -> {'OK' if fetch_ok else 'DIVERGES'}")
    ok = ok and fetch_ok

    # ---- full advisory both ways: labels + message must be identical ----
    plot_b = base[0]; plot_q = batch[0]
    zb = [x["z"] for x in base[1:] if x["z"] is not None]
    zq = [x["z"] for x in batch[1:] if x["z"] is not None]
    a_base = adv_mod.build_advisory_from_facts(
        plot_z=plot_b["z"], point_zs=zb, regime="rain-fed", target_month=int(target[5:7]),
        lat=BASE[0], lon=BASE[1], village="MURLI", target_date=target,
        clear_view_fraction=eng._clear_view(live[0]["shared"]))
    a_batch = adv_mod.build_advisory_from_facts(
        plot_z=plot_q["z"], point_zs=zq, regime="rain-fed", target_month=int(target[5:7]),
        lat=BASE[0], lon=BASE[1], village="MURLI", target_date=target,
        clear_view_fraction=eng._clear_view(live[0]["shared"]))
    labels_match = _labels(a_base) == _labels(a_batch)
    print(f"\n  advisory labels+row+message identical : {labels_match}")
    print(f"    severity={a_batch['derived']['severity']}  "
          f"conf={a_batch['derived']['effective_confidence']}  "
          f"attribution={a_batch['derived']['attribution']}  "
          f"risk={a_batch['matched']['risk_level']}")
    print(f"    clear_view(real)={a_batch['derived']['data_quality']} "
          f"({eng._clear_view(live[0]['shared'])})")

    ok = ok and labels_match
    import shutil
    shutil.rmtree(cache, ignore_errors=True)
    print(f"\n[harness] {'PASS' if ok else 'FAIL'} — "
          f"batched forward {'matches' if ok else 'DIVERGES from'} per-point "
          f"(labels {'identical' if labels_match else 'DIFFER'}, "
          f"max|dz|={max_z:.2e} vs tripwire {TRIP})")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
