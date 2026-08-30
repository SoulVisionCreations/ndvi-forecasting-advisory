"""
infer_cli.py — CLI NDVI forecast with the TFT + Prithvi champion bundle.

    python infer_cli.py --lat 25.44 --lon 91.71 --date 2024-08-15

Flow (max reuse, no duplication):
  1. fetch_all()  -> fortnightly numerical history (NDVI + ERA5) + admin + forecast
                     aggregates  [reused from Unified_Inference_Pipeline; inference-
                     only, NOT shared with training which reads the offline CSV]
  2. ndvi_core.download.fetch_walkback_tile() -> fresh quarterly HLS composite for
                     (lat,lon), cached locally (same recipe as training; never the
                     training tiles)
  3. ndvi_core.features -> the SAME feature-eng training ran (using saved
                     scaler/encoders + nearest-MWS static profile)
  4. ndvi_core.model_io -> rebuild ProTFT_Elite + LivePrithviProjector, load bundle
  5. forward -> inverse-scale -> NDVI forecast

Numerical data + composite are downloaded fresh per request and cached; training
composites are never read.
"""

import os
import sys
import argparse

import joblib
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)                       # inference/: shared_data_layer, infer_cli
sys.path.insert(0, os.path.dirname(BASE))      # repo root: ndvi_core (works run-in-place)

from ndvi_core import config as ncfg
from ndvi_core import download as ncd
from ndvi_core import dates as ncdates       # shared anchor/target-date geometry (C1)
from ndvi_core import model_io as ncm_io
from ndvi_core import inference as ncf_inf   # shared assemble->predict path
from ndvi_core import indicators as ncf_ind  # NDVI -> vegetation output indicators
from ndvi_core import results as ncres       # shared canonical result dict (C2)


# ---------------------------------------------------------------------------
# Nearest-MWS static profile (q-stats / baseline / peak_month / sensitivity)
# from the saved UNSCALED lookup. Small BallTree lookup (canonical home here).
# ---------------------------------------------------------------------------
def build_static_lookup(lookup_csv):
    """Load the UNSCALED static lookup + a haversine BallTree over its coords.
    Build once and reuse for repeated queries (serving)."""
    from sklearn.neighbors import BallTree
    lk = pd.read_csv(lookup_csv, sep=None, engine="python")   # sniff comma or tab (.csv or .tsv)
    tree = BallTree(np.radians(lk[["latitude", "longitude"]].values),
                    metric="haversine")
    return lk, tree


def nearest_static_profile(lat, lon, lookup_csv=None, lk=None, tree=None):
    """Nearest-MWS static profile row. Pass `lookup_csv` for a one-shot query
    (CLI), or a prebuilt `(lk, tree)` for repeated queries (serving)."""
    if lk is None or tree is None:
        lk, tree = build_static_lookup(lookup_csv)
    _, idx = tree.query(np.radians([[lat, lon]]), k=1)
    return lk.iloc[int(idx[0][0])].to_dict()


def _print_result(res, verbose=False):
    """Pretty-print the canonical result dict (same dict the API returns)."""
    loc, adm, an = res["location"], res["admin"], res["anomaly"]
    admin_str = " / ".join(str(adm.get(k, "?")) for k in ("state", "district", "tehsil"))
    pct = an.get("pct")
    pct_str = f"{pct:+.1f}%" if pct is not None else "n/a"
    print("\n================= NDVI / DROUGHT FORECAST =================")
    print(f"  location   : ({loc['lat']}, {loc['lon']})   {admin_str}")
    if verbose:
        print(f"  as-of      : {res['as_of']}   (data through {res['anchor_date']})")
    print(f"  target     : {res['target_date']}")
    print(f"  forecast   : NDVI {res['forecast_ndvi']}   ->   {res['relative_vegetation_condition']}")
    ndvi_str = f"   |   {an['ndvi']} NDVI   |   {pct_str}" if an.get("ndvi") is not None else ""  # verbose-only
    print(f"  anomaly    : z {an['z']}{ndvi_str}")
    n_str = f"  (n={an['baseline_n']})" if an.get("baseline_n") is not None else ""   # verbose-only
    print(f"               baseline {an['baseline_mean']} +/- {an['baseline_std']}{n_str}")
    if res.get("note"):
        print(f"  note       : {res['note']}")
    if verbose:
        t = res["prithvi_tile"]
        print(f"  prithvi    : {t['quarter']} {t['year']}  ({'used' if t['used'] else 'ZERO/missing'})")
    print("==========================================================")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lat", type=float, required=True)
    ap.add_argument("--lon", type=float, required=True)
    ap.add_argument("--date", default=None,
                    help="anchor / forecast-from (current) date YYYY-MM-DD; "
                         "default = today. The forecast target = date + lead. "
                         "Must be today or earlier (future dates are rejected).")
    ap.add_argument("--run_dir", default="weights",
                    help="dir holding the model asset (bundle + scaler + encoders + config); default: weights/")
    ap.add_argument("--model_path", default="tft_temporal_production_ft.pt")
    ap.add_argument("--scaler_path", default="standard_scaler_temporal_tft_ft.pkl")
    ap.add_argument("--encoder_path", default="label_encoders_temporal_tft_ft.pkl")
    ap.add_argument("--lookup_csv", default="weights/mws_static_lookup_UNSCALED.tsv")  # reader sniffs .csv/.tsv
    ap.add_argument("--model_dir", default="weights",
                    help="dir with prithvi_mae.py + config.json (vendored into the model asset); default: weights/")
    ap.add_argument("--cache_dir", default="inference_cache",
                    help="local cache for freshly downloaded composites")
    ap.add_argument("--project", default=os.environ.get("GEE_PROJECT", ""),
                    help="GEE project; empty (default) uses the credentials' own project")
    ap.add_argument("--corestack_key", default=os.environ.get("CORESTACK_KEY",
                    ""))
    ap.add_argument("--window", type=int, default=None,
                    help="lookback steps; default: the model's train_config.json (else config.WINDOW)")
    ap.add_argument("--lead", type=int, default=None,
                    help="forecast lead in fortnights; default: the model's train_config.json (else config.LEAD)")
    ap.add_argument("--years_lookback", type=int, default=10)
    ap.add_argument("--baseline_years", type=int, default=None,
                    help="most-recent-N-years window for the seasonal baseline "
                         "(default: all available years)")
    ap.add_argument("--baseline_month_window", type=int, default=0,
                    help="+/- months around the target month for the baseline "
                         "(0 = target month only)")
    ap.add_argument("--bias", type=float, default=0.0,
                    help="additive serve-time bias correction applied to the "
                         "forecast before computing anomalies (default 0 = off)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--verbose", action="store_true",
                    help="also show the as-of/anchor dates and the Prithvi tile")
    args = ap.parse_args()

    rd = args.run_dir
    ckpt = os.path.join(rd, args.model_path)
    scaler_path = os.path.join(rd, args.scaler_path)
    encoder_path = os.path.join(rd, args.encoder_path)

    # --- model's OWN train config (window/lead travel with the model) ------
    tcfg = ncm_io.load_train_config(rd)
    window = args.window if args.window is not None else tcfg["window"]
    lead = args.lead if args.lead is not None else tcfg["lead"]
    _src = "train_config.json" if os.path.exists(os.path.join(rd, "train_config.json")) else "config.py defaults"
    print(f"[infer] model config: window={window} lead={lead}  (from {_src})")

    # --- date semantics (C1): `date` = the as-of / forecast-from date (default
    # today). anchor s0 = nearest 14-day grid fortnight <= date; forecast TARGET =
    # anchor + lead (mirrors training's target = anchor_slot + lead). A missing
    # anchor reading is ffilled in build_window, never slid back. ----------------
    try:
        ff_dt, anchor_dt, target_dt = ncdates.resolve_forecast_dates(args.date, lead=lead)
    except ValueError as e:
        raise SystemExit(f"[infer] {e}")
    forecast_from = ff_dt.strftime("%Y-%m-%d")
    anchor = anchor_dt.strftime("%Y-%m-%d")
    target = target_dt.strftime("%Y-%m-%d")
    print(f"[infer] forecast-from: {forecast_from}{' = today' if not args.date else ''} "
          f"(anchor {anchor} = nearest fortnight <= date)  ->  target: {target}")

    # --- EE init (HLS high-volume) -----------------------------------------
    import ee
    if args.project:
        ee.Initialize(project=args.project, opt_url=ncd.HIGH_VOLUME)
    else:
        ee.Initialize(opt_url=ncd.HIGH_VOLUME)   # use credentials' own project

    # --- 1) numerical history + admin + forecast (reused fetch_all) ---------
    from shared_data_layer import fetch_all
    shared = fetch_all(lat=args.lat, lon=args.lon, target_date=target,
                       corestack_key=args.corestack_key,
                       years_lookback=args.years_lookback, lead_fortnights=lead)
    if any("ERROR:" in q for q in shared.quality_issues):
        raise SystemExit(f"data fetch failed: {shared.quality_issues}")

    # --- 2) fresh composite for the walk-back quarter (cached) --------------
    # lid keys the tile file by location (not a fixed 'req') so a shared cache dir
    # can never hand one point another's tile (see ncd.loc_id).
    lid = ncd.loc_id(args.lat, args.lon)
    tile_path, _, tyear, tq = ncd.fetch_walkback_tile(
        lid, args.lat, args.lon, target, args.cache_dir, lag=1)
    print(f"[infer] composite: {os.path.basename(tile_path)} (quarter {tq} {tyear})")

    # --- 3) artifacts + static profile + model -----------------------------
    scaler = joblib.load(scaler_path)
    encoders = joblib.load(encoder_path)
    prof = nearest_static_profile(args.lat, args.lon, args.lookup_csv)
    model, key_to_row, zero_idx = ncm_io.build_model(
        args.model_dir, ckpt, encoders, args.cache_dir,
        latlon={lid: (args.lat, args.lon)}, device=args.device, tcfg=tcfg)

    # --- 4) forecast via the shared ndvi_core.inference path ---------------
    ndvi, emb_idx = ncf_inf.forecast_point(
        model, shared, lid, args.lat, args.lon, prof, tyear, tq,
        key_to_row, zero_idx, scaler, encoders,
        window=window, device=args.device)

    # --- 5) NDVI -> vegetation output indicators (the workflow's product) --
    ind = ncf_ind.vegetation_indicators(
        ndvi, shared.history_df, target, prof=prof,
        month_window=args.baseline_month_window, years=args.baseline_years,
        bias=args.bias)

    # --- 6) canonical result dict (shared with the API) -> pretty-print -----
    stale_fn, last_obs = ncf_inf.anchor_staleness(shared, anchor)
    tile = {"quarter": tq, "year": int(tyear), "used": emb_idx != zero_idx}
    res = ncres.build_result(
        args.lat, args.lon, shared.admin, forecast_from, anchor, target,
        ndvi, ind, tile, verbose=args.verbose,
        stale_fortnights=stale_fn, last_obs=last_obs)
    _print_result(res, verbose=args.verbose)
    return res


if __name__ == "__main__":
    main()
