"""
training/download_composites.py — bulk HLS quarterly-composite downloader.

A thin loop over ndvi_core.download.fetch_composite — the SAME quarterly recipe
the inference path uses and the training tiles were built with — so downloaded
tiles are byte-for-byte consumable by the trainer's folder TileStore. Reads an
MWS manifest (columns: mws_id,lat,lon) and writes
    composite_<mws_id>_<year>_<Q>.tif   (6-band HLS, 224x224)
into --out_dir.

The GEE cloud project id comes from the GEE_PROJECT env var (or --project).
Sequential by default (GEE's token bucket throttles at ~4+ concurrent workers);
raise --workers cautiously.

Example (a few tiles, to verify the pipeline):
    GEE_PROJECT=my-gee-project \\
    python training/download_composites.py \\
        --mws_csv all_mws_locations_dedup.csv --out_dir qtiles \\
        --years 2023 --quarters Q3 --limit 5

Full quarterly set (like the training tiles; drops gappy tiles <0.75 coverage):
    python training/download_composites.py --mws_csv ... --out_dir quarterly_composites \\
        --years 2016 2017 2018 2019 2020 2021 2022 2023 --min_coverage 0.75
"""
import os
import sys
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

# repo root on sys.path so `from ndvi_core import ...` resolves when run directly.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ndvi_core import download as ncd


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mws_csv", required=True, help="manifest with columns mws_id,lat,lon")
    ap.add_argument("--out_dir", required=True, help="output dir for the .tif tiles")
    ap.add_argument("--years", nargs="+", type=int, required=True, help="e.g. 2021 2022 2023")
    ap.add_argument("--quarters", nargs="+", default=["Q1", "Q2", "Q3", "Q4"],
                    choices=["Q1", "Q2", "Q3", "Q4"])
    ap.add_argument("--limit", type=int, default=None, help="only the first N MWS rows (for testing)")
    ap.add_argument("--min_coverage", type=float, default=0.0,
                    help="report tiles below this NIR-valid fraction as low-coverage (0 = keep all)")
    ap.add_argument("--project", default=os.environ.get("GEE_PROJECT", ""),
                    help="GEE cloud project id (default: $GEE_PROJECT)")
    ap.add_argument("--workers", type=int, default=1,
                    help="parallel download threads (default 1; GEE throttles at ~4+)")
    args = ap.parse_args()

    if not args.project:
        raise SystemExit("no GEE project — set GEE_PROJECT or pass --project")

    import ee
    ee.Initialize(project=args.project, opt_url=ncd.HIGH_VOLUME)

    df = pd.read_csv(args.mws_csv)
    for col in ("mws_id", "lat", "lon"):
        if col not in df.columns:
            raise SystemExit(f"--mws_csv missing column '{col}' (has {list(df.columns)})")
    if args.limit:
        df = df.head(args.limit)

    jobs = [(str(r.mws_id), float(r.lat), float(r.lon), y, q)
            for r in df.itertuples(index=False)
            for y in args.years for q in args.quarters]
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"[dl] {len(df)} MWS x {len(args.years)} yr x {len(args.quarters)} q = {len(jobs)} tiles "
          f"-> {args.out_dir} (project {args.project}, workers {args.workers})")

    def _one(job):
        mws_id, lat, lon, year, quarter = job
        try:
            path, valid = ncd.fetch_composite(mws_id, lat, lon, year, quarter, args.out_dir)
            return (job, path, valid, None)
        except Exception as e:                    # noqa: BLE001 — report + continue
            return (job, None, None, str(e)[:160])

    ok = low = fail = 0

    def _report(res):
        nonlocal ok, low, fail
        job, path, valid, err = res
        mws_id, lat, lon, year, quarter = job
        tag = f"{mws_id} {year} {quarter}"
        if err:
            fail += 1
            print(f"[dl][FAIL] {tag}: {err}")
        elif valid is not None and valid < args.min_coverage:
            low += 1
            print(f"[dl][low ] {tag}: coverage {valid:.2f} < {args.min_coverage}")
        else:
            ok += 1
            cov = "cached" if valid is None else f"cov {valid:.2f}"
            print(f"[dl][ok  ] {tag}: {os.path.basename(path)} ({cov})")

    if args.workers <= 1:
        for job in jobs:
            _report(_one(job))
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            for res in as_completed([ex.submit(_one, j) for j in jobs]):
                _report(res.result())

    print(f"[dl] done: ok={ok} low_coverage={low} fail={fail} / {len(jobs)}")


if __name__ == "__main__":
    main()
