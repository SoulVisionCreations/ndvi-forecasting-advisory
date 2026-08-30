"""
batch_sanity.py — forecast vs actual NDVI over N sampled MWS points -> CSV.

Samples N MWS from the static lookup, runs the TFT+Prithvi inference for each
(reusing ndvi_core + infer_cli helpers), fetches the actual MODIS NDVI at the
same (point, date), and writes a comparison CSV. The model/encoders/scaler load
ONCE; composites are pre-fetched so the graph is built a single time.

    python batch_sanity.py --n 20 --date 2023-08-15 --out sanity_20.csv
"""

import os
import sys
import argparse
import random

import joblib
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
sys.path.insert(0, os.path.join(BASE, "..", "Unified_Inference_Pipeline"))

from ndvi_core import config as ncfg
from ndvi_core import download as ncd
from ndvi_core import model_io as ncm_io
from ndvi_core import inference as ncf_inf   # shared assemble->predict path (no dup)


# actual_ndvi moved to ndvi_core.download (shared with batch_metrics) -> ncd.actual_ndvi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--date", default="2023-08-15")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--run_dir", default="weights")
    ap.add_argument("--lookup_csv", default="weights/mws_static_lookup_UNSCALED.tsv")
    ap.add_argument("--model_dir", default="weights")
    ap.add_argument("--cache_dir", default="inference_cache_batch")
    ap.add_argument("--corestack_key", default=os.environ.get("CORESTACK_KEY",
                    ""))
    ap.add_argument("--out", default="sanity_forecast_vs_actual.csv")
    ap.add_argument("--window", type=int, default=None,
                    help="lookback steps; default: the model's train_config.json (else config.WINDOW)")
    ap.add_argument("--years_lookback", type=int, default=10)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    rd = args.run_dir
    ckpt = os.path.join(rd, "tft_temporal_production_ft.pt")
    scaler = joblib.load(os.path.join(rd, "standard_scaler_temporal_tft_ft.pkl"))
    encoders = joblib.load(os.path.join(rd, "label_encoders_temporal_tft_ft.pkl"))

    tcfg = ncm_io.load_train_config(rd)                 # model's own window/lead
    window = args.window if args.window is not None else tcfg["window"]
    lead = tcfg["lead"]
    print(f"[batch] model config: window={window} lead={lead}", flush=True)

    import ee
    ee.Initialize(opt_url=ncd.HIGH_VOLUME)
    from shared_data_layer import fetch_all

    lk = pd.read_csv(args.lookup_csv)
    random.seed(args.seed)
    sample = lk.sample(n=args.n, random_state=args.seed).reset_index(drop=True)
    print(f"sampled {len(sample)} MWS | date {args.date}", flush=True)

    # --- phase 1: fetch composite + numerical for each point -----------------
    pts = []
    for i, row in sample.iterrows():
        mws = str(row.mws_id); lat = float(row.latitude); lon = float(row.longitude)
        try:
            _, _, yr, q = ncd.fetch_walkback_tile(mws, lat, lon, args.date, args.cache_dir, lag=1)
            shared = fetch_all(lat=lat, lon=lon, target_date=args.date,
                               corestack_key=args.corestack_key,
                               years_lookback=args.years_lookback, lead_fortnights=lead)
            if any("ERROR:" in s for s in shared.quality_issues):
                print(f"  [{i+1}/{len(sample)}] {mws} SKIP: {shared.quality_issues[:1]}", flush=True)
                continue
            pts.append({"row": row, "shared": shared, "yr": yr, "q": q})
            print(f"  [{i+1}/{len(sample)}] {mws} fetched (tile {q} {yr})", flush=True)
        except Exception as e:
            print(f"  [{i+1}/{len(sample)}] {mws} ERROR: {str(e)[:120]}", flush=True)

    if not pts:
        raise SystemExit("no points fetched")

    # --- build model ONCE over the cache (all tiles present) -----------------
    latlon = {str(p["row"].mws_id): (float(p["row"].latitude), float(p["row"].longitude))
              for p in pts}
    model, key_to_row, zero_idx = ncm_io.build_model(
        args.model_dir, ckpt, encoders, args.cache_dir, latlon, device=args.device, tcfg=tcfg)

    # --- phase 2: predict (shared ncf_inf.forecast_point) + actual per point --
    out = []
    for p in pts:
        row, shared, yr, q = p["row"], p["shared"], p["yr"], p["q"]
        mws = str(row.mws_id); lat = float(row.latitude); lon = float(row.longitude)
        try:
            fc, emb_idx = ncf_inf.forecast_point(
                model, shared, mws, lat, lon, row, yr, q,
                key_to_row, zero_idx, scaler, encoders,
                window=window, device=args.device)
            act = ncd.actual_ndvi(lat, lon, args.date)
            out.append({
                "mws_id": mws, "lat": lat, "lon": lon, "date": args.date,
                "tile_quarter": q, "tile_year": int(yr),
                "tile_used": "yes" if emb_idx != zero_idx else "ZERO/missing",
                "state": shared.admin.get("state"),
                "forecast_ndvi": round(fc, 4),
                "actual_ndvi": round(act, 4) if act is not None else None,
                "abs_error": round(abs(fc - act), 4) if act is not None else None,
            })
            print(f"  {mws}: forecast {fc:.4f} | actual {act if act is None else round(act,4)}", flush=True)
        except Exception as e:
            print(f"  {mws} PREDICT ERROR: {str(e)[:140]}", flush=True)

    df = pd.DataFrame(out)
    df.to_csv(args.out, index=False)
    print(f"\nwrote {args.out} ({len(df)} rows)")
    paired = df.dropna(subset=["actual_ndvi"])
    if len(paired) >= 2:
        from sklearn.metrics import r2_score
        a = paired["actual_ndvi"].values
        f = paired["forecast_ndvi"].values
        r2 = r2_score(a, f)                                   # forecast vs actual
        mae = paired["abs_error"].mean()
        rmse = float(np.sqrt(np.mean((f - a) ** 2)))
        bias = float(np.mean(f - a))
        corr = float(paired["forecast_ndvi"].corr(paired["actual_ndvi"]))
        summary = (f"paired n={len(paired)} | R2 {r2:.4f} | MAE {mae:.4f} | "
                   f"RMSE {rmse:.4f} | bias {bias:+.4f} | corr {corr:.3f}")
        print(summary)
        with open(os.path.splitext(args.out)[0] + "_summary.txt", "w") as fh:
            fh.write(summary + "\n")


if __name__ == "__main__":
    main()
