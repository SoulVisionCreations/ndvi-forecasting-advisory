"""
batch_metrics.py — forecast-vs-actual evaluation of the TFT+Prithvi serving path.

Samples N_MWS x DATES_PER_MWS target dates (default 50 x 10 = 500), runs the SAME
ndvi_core engine that infer_cli / serve_api use (no duplicated inference logic),
compares each forecast to the actual MODIS NDVI at the target, and reports
training-style metrics + per-query latency. Writes one CSV row per sample,
including failures (status column).

Design:
  * TARGET-DATE semantics: cutoff = target - LEAD fortnights (the model's true
    3.5-month forecast) — NO 2-week serving lag — so results are comparable to
    the training test.
  * Model is built ONCE; GEE fetches are parallelised (fetch_composite skips
    existing + writes atomically -> parallel-safe).
  * Actual NDVI = MODIS/061/MOD13Q1 16-day median x1e-4 at the target.

    python batch_metrics.py --n_mws 50 --dates_per_mws 10 \
        --min_date 2024-01-01 --max_date 2026-06-15 --out batch_metrics_500.csv
"""

import os
import sys
import time
import random
import argparse
import threading
from datetime import date, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

import joblib
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
sys.path.insert(0, os.path.join(BASE, "..", "Unified_Inference_Pipeline"))

from ndvi_core import config as ncfg
from ndvi_core import download as ncd
from ndvi_core import model_io as ncm_io
from ndvi_core import inference as ncf_inf          # shared forecast engine (same as infer_cli)
from ndvi_core import indicators as ncf_ind
from infer_cli import build_static_lookup, nearest_static_profile   # reuse, no dup


# ---------------------------------------------------------------------------
# Transient-failure retry. Under load the GEE high-volume endpoint can drop
# bands (e.g. coverage_fraction), return 429/5xx, or throw permission/quota
# errors — these are retried with exponential backoff + jitter. Genuine
# terminal conditions (no MODIS actual, no >=0.75-coverage tile after
# walk-back) are NOT retried.
# ---------------------------------------------------------------------------
class _Retryable(Exception):
    pass


_TRANSIENT_HINTS = (
    "coverage_fraction", "missing columns", "permission", "quota",
    "rate limit", "ratelimit", "429", "500", "502", "503", "504",
    "timed out", "timeout", "deadline", "temporarily", "unavailable",
    "connection", "reset by peer", "eof occurred",
    # GEE degradation-window phrases (Guard 1/3 in shared_data_layer.fetch_all)
    "all-nan", "returned no data", "returned zero rows", "partial fetch",
    "access was denied", "denied mid-batch", "access denied", "no satellite coverage",
)


def _is_transient(msg):
    m = str(msg).lower()
    return any(h in m for h in _TRANSIENT_HINTS)


# actual_ndvi moved to ndvi_core.download (shared with batch_sanity) -> ncd.actual_ndvi


def last_observed_ndvi(shared):
    """Persistence baseline = last valid NDVI in the window (<= cutoff)."""
    h = shared.history_df.copy()
    h["_d"] = pd.to_datetime(h["date"] if "date" in h.columns else h["date_str"])
    h = h[h["_d"] <= pd.to_datetime(shared.data_cutoff_dt)]
    v = pd.to_numeric(h["NDVI"], errors="coerce").dropna()
    return float(v.iloc[-1]) if len(v) else np.nan


def quarter_clim(prof, target_date):
    """Per-MWS per-quarter NDVI climatology mean from the lookup (anomaly ref)."""
    q = ((pd.to_datetime(target_date).month - 1) // 3) + 1
    try:
        return float(prof.get(f"q{q}_mean_ndvi", np.nan))
    except (TypeError, ValueError):
        return np.nan


def sample_pairs(lk, n_mws, dates_per_mws, min_date, max_date, seed):
    """n_mws random MWS x dates_per_mws random target dates in [min,max]."""
    rng = random.Random(seed)
    sub = lk.sample(n=min(n_mws, len(lk)), random_state=seed).reset_index(drop=True)
    d0 = date.fromisoformat(min_date); span = (date.fromisoformat(max_date) - d0).days
    rows = []
    for _, r in sub.iterrows():
        for _ in range(dates_per_mws):
            t = (d0 + timedelta(days=rng.randint(0, span))).isoformat()
            rows.append({"mws_id": str(r.mws_id), "lat": float(r.latitude),
                         "lon": float(r.longitude), "target": t})
    rng.shuffle(rows)   # de-cluster MWS so a partial/interrupted run stays unbiased
    return rows


def fetch_one(smp, fetch_all, cache_dir, corestack_key, years_lookback, lead,
              retries=3, base_delay=8.0):
    """Parallel worker: numerical + composite + actual for one sample (timed).
    Transient GEE failures are retried with exponential backoff + jitter;
    terminal conditions (no actual, no covered tile) return immediately.
    `attempts` records how many tries the sample took."""
    t0 = time.time()
    out = dict(smp)
    last_err = None
    attempt = 0
    for attempt in range(retries + 1):
        try:
            shared = fetch_all(lat=smp["lat"], lon=smp["lon"], target_date=smp["target"],
                               corestack_key=corestack_key, years_lookback=years_lookback,
                               lead_fortnights=lead)
            _errs = [q for q in shared.quality_issues if "ERROR:" in q]
            if _errs:
                q0 = _errs[0]                      # the REAL blocker, not a masking WARNING
                if _is_transient(q0) and attempt < retries:
                    last_err = f"fetch {q0}"
                    raise _Retryable(last_err)
                out["status"] = f"ERROR: fetch {q0[:120]}"
                break
            _, _, tyear, tq = ncd.fetch_walkback_tile(
                smp["mws_id"], smp["lat"], smp["lon"], smp["target"], cache_dir, lag=1)
            act = ncd.actual_ndvi(smp["lat"], smp["lon"], smp["target"])
            if act is None:
                out["status"] = "ERROR: no actual NDVI at target"
                break
            out.update(shared=shared, tyear=int(tyear), tq=tq, actual=act,
                       admin_state=shared.admin.get("state"),
                       data_cutoff=str(pd.to_datetime(shared.data_cutoff_dt).date()),
                       status="OK")
            break
        except _Retryable as ex:
            last_err = str(ex)
        except Exception as ex:
            last_err = str(ex)[:140]
            if not (_is_transient(last_err) and attempt < retries):
                out["status"] = f"ERROR: {last_err}"
                break
        # transient -> exponential backoff + jitter before the next attempt
        delay = base_delay * (2 ** attempt)
        time.sleep(delay + random.uniform(0, 0.3 * delay))
    out["fetch_time"] = time.time() - t0
    out["attempts"] = attempt + 1
    return out


def compute_metrics(df):
    """Global + median-basin + anomaly + persistence + delta metrics on OK rows."""
    from sklearn.metrics import r2_score
    a = df["actual_ndvi"].values; f = df["forecast_ndvi"].values
    c = df["current_ndvi"].values; cl = df["seasonal_clim"].values
    bias = float(np.mean(f - a))
    m = {
        "n_samples": int(len(df)),
        "global_r2": float(r2_score(a, f)),
        "rmse": float(np.sqrt(np.mean((f - a) ** 2))),
        "mae": float(np.mean(np.abs(f - a))),
        "bias": bias,
        "corr": float(np.corrcoef(a, f)[0, 1]),
        "debiased_r2": float(r2_score(a, f - bias)),
        "global_anomaly_r2": float(r2_score(a - cl, f - cl)),
        "delta_r2": float(r2_score(a - c, f - c)),
        "persistence_global_r2": float(r2_score(a, c)),
    }

    def _basin(fn):
        vals = []
        for _, g in df.groupby("mws_id"):
            if len(g) >= 2:
                vals.append(fn(g))
        return float(np.median(vals)) if vals else float("nan"), len(vals)

    m["median_basin_r2"], m["n_basins"] = _basin(
        lambda g: r2_score(g["actual_ndvi"], g["forecast_ndvi"]))
    m["median_basin_anomaly_r2"], _ = _basin(
        lambda g: r2_score(g["actual_ndvi"] - g["seasonal_clim"],
                           g["forecast_ndvi"] - g["seasonal_clim"]))
    m["median_basin_persistence_r2"], _ = _basin(
        lambda g: r2_score(g["actual_ndvi"], g["current_ndvi"]))
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_mws", type=int, default=50)
    ap.add_argument("--dates_per_mws", type=int, default=10)
    ap.add_argument("--min_date", default="2024-01-01")
    ap.add_argument("--max_date", default="2026-06-15")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--run_dir", default="weights")
    ap.add_argument("--lookup_csv", default="weights/mws_static_lookup_UNSCALED.tsv")
    ap.add_argument("--model_dir", default="weights")
    ap.add_argument("--cache_dir", default="batch_metrics_cache")
    ap.add_argument("--corestack_key", default=os.environ.get("CORESTACK_KEY",
                    ""))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--years_lookback", type=int, default=10)
    ap.add_argument("--retries", type=int, default=2,
                    help="inline retries for transient GEE failures (exp backoff + jitter)")
    ap.add_argument("--retry_base_delay", type=float, default=8.0,
                    help="base seconds for the first inline retry backoff")
    ap.add_argument("--sweeps", type=int, default=3,
                    help="max fetch sweeps; transient (GEE) failures are re-queued to the "
                         "next sweep so they retry in a later healthy window")
    ap.add_argument("--sweep_wait", type=float, default=120.0,
                    help="seconds to wait between sweeps (GEE degradation windows recover)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="batch_metrics_500.csv")
    args = ap.parse_args()

    rd = args.run_dir
    os.makedirs(args.cache_dir, exist_ok=True)
    scaler = joblib.load(os.path.join(rd, "standard_scaler_temporal_tft_ft.pkl"))
    encoders = joblib.load(os.path.join(rd, "label_encoders_temporal_tft_ft.pkl"))
    ckpt = os.path.join(rd, "tft_temporal_production_ft.pt")

    tcfg = ncm_io.load_train_config(rd)
    lead, window = int(tcfg["lead"]), int(tcfg["window"])
    print(f"[metrics] model config: window={window} lead={lead} | run_dir={rd}", flush=True)

    import ee
    ee.Initialize(opt_url=ncd.HIGH_VOLUME)
    from shared_data_layer import fetch_all

    lk = pd.read_csv(args.lookup_csv)
    lk_tree_df, tree = build_static_lookup(args.lookup_csv)
    samples = sample_pairs(lk, args.n_mws, args.dates_per_mws,
                           args.min_date, args.max_date, args.seed)
    print(f"[metrics] {len(samples)} samples ({args.n_mws} MWS x {args.dates_per_mws} dates) "
          f"in [{args.min_date}, {args.max_date}]", flush=True)

    # --- phase 1: parallel fetch with RE-SWEEP of transient failures ----------
    # GEE degrades for minutes at a time then recovers, so a sample that fails
    # transiently in one sweep is re-queued and retried in a later (healthy)
    # sweep. Terminal conditions (no MODIS actual / no covered tile) are never
    # re-swept. This is what lets the run complete despite intermittent GEE.
    from collections import Counter
    t_phase1 = time.time()

    def _terminal(status):
        return ("no actual NDVI at target" in status) or ("no quarterly tile" in status)

    def _do_sweep(pending, sweep_no):
        results, done = [], 0
        err_ctr = Counter()
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(fetch_one, s, fetch_all, args.cache_dir,
                              args.corestack_key, args.years_lookback, lead,
                              args.retries, args.retry_base_delay) for s in pending]
            for fu in as_completed(futs):
                r = fu.result(); results.append(r); done += 1
                if r["status"] != "OK":
                    key = (r["status"].split(":", 1)[1].strip()[:45]
                           if ":" in r["status"] else r["status"])
                    err_ctr[key] += 1
                if done % 25 == 0 or done == len(pending):
                    ok = sum(1 for x in results if x["status"] == "OK")
                    print(f"  [sweep {sweep_no}] {done}/{len(pending)} (ok={ok}) | "
                          f"fetch-error-modes={dict(err_ctr)}", flush=True)
        return results

    best = {}                                   # (mws_id, target) -> latest result
    pending = list(samples)
    sweeps_run = 0
    for sweep_no in range(1, args.sweeps + 1):
        if not pending:
            break
        sweeps_run = sweep_no
        if sweep_no > 1:
            print(f"[metrics] sweep {sweep_no}: re-fetching {len(pending)} transient "
                  f"failures after {args.sweep_wait:.0f}s wait", flush=True)
            time.sleep(args.sweep_wait)
        for r in _do_sweep(pending, sweep_no):
            best[(r["mws_id"], r["target"])] = r
        pending = [{"mws_id": r["mws_id"], "lat": r["lat"], "lon": r["lon"],
                    "target": r["target"]}
                   for r in best.values()
                   if r["status"] != "OK" and not _terminal(r["status"])]
        ok_now = sum(1 for v in best.values() if v["status"] == "OK")
        print(f"[metrics] after sweep {sweep_no}: ok={ok_now}/{len(best)} | "
              f"{len(pending)} transient failures queued", flush=True)

    fetched = list(best.values())
    phase1_wall = time.time() - t_phase1
    ok_fetch = [r for r in fetched if r["status"] == "OK"]
    final_errs = Counter()
    for r in fetched:
        if r["status"] != "OK":
            key = (r["status"].split(":", 1)[1].strip()[:45]
                   if ":" in r["status"] else r["status"])
            final_errs[key] += 1
    print(f"[metrics] phase-1 fetch ({sweeps_run} sweep(s)): {len(ok_fetch)}/{len(fetched)} ok "
          f"in {phase1_wall/60:.1f} min | final-error-modes={dict(final_errs)}", flush=True)

    # --- build model ONCE over the cached tiles -------------------------------
    latlon = {r["mws_id"]: (r["lat"], r["lon"]) for r in ok_fetch}
    model, key_to_row, zero_idx = ncm_io.build_model(
        args.model_dir, ckpt, encoders, args.cache_dir, latlon, device=args.device, tcfg=tcfg)

    # --- phase 2: forecast each OK sample (timed) -----------------------------
    rows = []
    for r in fetched:
        if r["status"] != "OK":
            rows.append({k: r.get(k) for k in ("mws_id", "lat", "lon", "target")}
                        | {"status": r["status"], "attempts": r.get("attempts"),
                           "query_time_s": round(r.get("fetch_time", 0), 2)})
            continue
        t0 = time.time()
        try:
            prof = nearest_static_profile(r["lat"], r["lon"], lk=lk_tree_df, tree=tree)
            ndvi, emb_idx = ncf_inf.forecast_point(
                model, r["shared"], r["mws_id"], r["lat"], r["lon"], prof,
                r["tyear"], r["tq"], key_to_row, zero_idx, scaler, encoders,
                window=window, device=args.device)
            ind = ncf_ind.vegetation_indicators(ndvi, r["shared"].history_df, r["target"], prof=prof)
            cur = last_observed_ndvi(r["shared"])
            clim = quarter_clim(prof, r["target"])
            ftime = time.time() - t0
            rows.append({
                "mws_id": r["mws_id"], "lat": r["lat"], "lon": r["lon"],
                "target_date": r["target"], "data_cutoff": r["data_cutoff"],
                "tile_quarter": r["tq"], "tile_year": r["tyear"],
                "tile_used": "yes" if emb_idx != zero_idx else "ZERO/missing",
                "admin_state": r["admin_state"],
                "forecast_ndvi": round(float(ndvi), 4),
                "current_ndvi": round(cur, 4) if np.isfinite(cur) else None,
                "actual_ndvi": round(float(r["actual"]), 4),
                "seasonal_clim": round(clim, 4) if np.isfinite(clim) else None,
                "std_anomaly": ind["standardized_anomaly"],
                "relative_vegetation_condition": ind["relative_vegetation_condition"],
                "abs_error": round(abs(float(ndvi) - float(r["actual"])), 4),
                "query_time_s": round(r["fetch_time"] + ftime, 2),
                "attempts": r.get("attempts"),
                "status": "OK",
            })
        except Exception as ex:
            rows.append({"mws_id": r["mws_id"], "lat": r["lat"], "lon": r["lon"],
                         "target_date": r["target"],
                         "status": f"ERROR: forecast {str(ex)[:120]}",
                         "attempts": r.get("attempts"),
                         "query_time_s": round(r.get("fetch_time", 0), 2)})

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    total_wall = time.time() - t_phase1
    print(f"\n[metrics] wrote {args.out} ({len(df)} rows)", flush=True)

    # --- metrics + timing on OK rows ------------------------------------------
    ok = df[(df["status"] == "OK") & df["actual_ndvi"].notna()].copy()
    ok = ok[ok[["forecast_ndvi", "current_ndvi", "seasonal_clim"]].notna().all(axis=1)]
    lat_all = df["query_time_s"].dropna().values
    lat_ok = ok["query_time_s"].values
    lines = [
        f"samples: total={len(df)} | OK={len(ok)} | errors={len(df)-len(ok)}",
        f"per-query latency (s): avg={np.mean(lat_ok):.1f} median={np.median(lat_ok):.1f} "
        f"p90={np.percentile(lat_ok,90):.1f}   (all incl errors: avg={np.mean(lat_all):.1f})",
        f"wall-clock: total={total_wall/60:.1f} min | throughput={len(df)/(total_wall/60):.1f} samples/min "
        f"({args.workers} workers)",
        f"retries: {int((pd.to_numeric(df.get('attempts'), errors='coerce').fillna(1) > 1).sum())}"
        f"/{len(df)} samples needed >1 attempt (transient GEE, backoff+jitter)",
    ]
    if len(ok) >= 5:
        m = compute_metrics(ok)
        lines += [
            "",
            f"GLOBAL:   R2={m['global_r2']:.4f}  RMSE={m['rmse']:.4f}  MAE={m['mae']:.4f}  "
            f"bias={m['bias']:+.4f}  corr={m['corr']:.3f}  debiased_R2={m['debiased_r2']:.4f}",
            f"MEDIAN-BASIN ({m['n_basins']} basins):  R2={m['median_basin_r2']:.4f}  "
            f"anomaly_R2={m['median_basin_anomaly_r2']:.4f}  persistence_R2={m['median_basin_persistence_r2']:.4f}",
            f"ANOMALY (global drought skill):  R2={m['global_anomaly_r2']:.4f}",
            f"BASELINES:  persistence_global_R2={m['persistence_global_r2']:.4f}  delta_R2={m['delta_r2']:.4f}",
        ]
    summary = "\n".join(lines)
    print("\n" + "=" * 70 + "\n" + summary + "\n" + "=" * 70, flush=True)
    with open(os.path.splitext(args.out)[0] + "_summary.txt", "w") as fh:
        fh.write(summary + "\n")


if __name__ == "__main__":
    main()
