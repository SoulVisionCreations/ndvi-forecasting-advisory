# Inference

How the package turns a `(lat, lon, date)` request into an NDVI forecast + vegetation-condition
signal. The **CLI** (`inference/infer_cli.py`) and the **API** (`inference/serve_api.py`) share the
exact same code path (`ndvi_core`), so they always produce identical output for the same input.

---

## 1. What you get

A forecast of **NDVI ~3 months ahead** of the given date, plus a *relative* vegetation-condition
class and a standardized anomaly. Default response:

```jsonc
{
  "location": { "lat": 25.44, "lon": 91.71 },
  "admin":    { "state": "...", "district": "...", "tehsil": "..." },
  "target_date": "2024-11-09",                 // = anchor + ~3 months
  "forecast_ndvi": 0.608,                       // 0..1 (rounded to 3 decimals)
  "relative_vegetation_condition": "Above normal",
  "anomaly": {
    "z": 1.58,                                  // standardized anomaly (drives the class)
    "baseline_mean": 0.529,                      // the location's seasonal normal
    "baseline_std": 0.05                         // effective spread used for z (floored at min_std=0.05)
  }
}
```

`z` is computed from full-precision internals and the other values are rounded to 3 dp, so recomputing
`z` from the rounded `forecast_ndvi` / `baseline_mean` is very close but `z` remains the authoritative
value.

Add `?debug=true` (API) or `--verbose` (CLI) to also get: `as_of`, `anchor_date`,
`prithvi_tile{quarter,year,used}`, `anomaly.baseline_source`, `anomaly.baseline_n`,
`anomaly.ndvi` (raw anomaly) + `anomaly.pct`, `anchor_stale_fortnights`, `last_observation`, and a
staleness `note` when the anchor imagery had to be carried forward.

---

## 2. Dates: anchor and target

`date` is the **as-of / forecast-from** date (default = today). Two dates are derived from it:

- **anchor** = the most recent 14-day grid fortnight **on or before** `date` (snapped back, ≤ 1
  fortnight). This is the last input step.
- **target** = anchor **+ 7 fortnights (98 days ≈ 3 months)**. This is what the model predicts.

So `target` is always ~3 months **after** `date`. If the anchor's NDVI is missing (cloud / latency),
its value is forward-filled from the previous reading — the anchor position never slides.

**`date` must be today or earlier.** A future date is rejected (HTTP 422 / CLI error):

```
date '2028-01-15' is in the future; it must be today (2026-07-08) or earlier.
```

Why: the model can only forecast ~3 months ahead of the latest *real* data (≈ today). A future date
would pull today-anchored data but label it with a future target season — a temporal mismatch the
model was never trained on. Use **today** for a live forecast, or a **past** date for a backtest.

---

## 3. The relative vegetation-condition class

The class is **relative to the location's own history for that time of year**, not an absolute
greenness threshold. It is the standardized anomaly `z = (forecast − baseline_mean) / baseline_std`,
where the baseline is the point's own NDVI observations for the target month across prior years.

| z | class |
|---|-------|
| < −1      | Below normal |
| −1 … 1    | Near normal |
| > 1       | Above normal |

The baseline spread is floored at `min_std = 0.05` (a realistic NDVI variability floor), so very
stable pixels/months don't produce an exploding z; `baseline_std` in the response is this effective
spread, so `z = anomaly / baseline_std` reconciles.

`baseline_n` (verbose) is the **number of observations** behind the baseline — with a fortnightly
cadence that is roughly 2 per month × ~10 years ≈ 20, not a count of years.

Because the label is relative, a lush evergreen point can read "Below normal" (below *its* normal)
while a semi-arid point at moderate NDVI reads "Above normal" (above *its* normal). A fixed
"NDVI < 0.5 = bad" rule would mislabel both. Note the class is deliberately coarse (3 bins) and
best read at MWS scale — see the sensitivity note in the release docs.

---

## 4. Pipeline (per request)

1. Resolve `anchor` / `target` from `date` (`ndvi_core.dates`).
2. Fetch input data **live** (`shared_data_layer.fetch_all` + `ndvi_core.download`):
   numerical history (GEE), admin (CoreStack), forecast weather (Open-Meteo → climatology fallback),
   and the HLS composite tile (GEE). See `docs/DATA_SOURCES.md`.
3. Build the 24-fortnight scaled input window + static features (same feature engineering as training).
4. Encode the tile with Prithvi (+ LoRA) → project to the 32-d image context.
5. TFT forward pass → inverse-scale → `forecast_ndvi`.
6. Post-process into anomaly + relative class (`ndvi_core.indicators`), assemble the canonical result.

The model loads **once** (at CLI start, or API startup); each request only fetches data and runs the
forward pass. Latency is dominated by the live GEE history fetch (~30–60 s; highly variable).

---

## 5. Running it

**CLI**
```bash
python inference/infer_cli.py --lat 25.44 --lon 91.71 --date 2024-08-15 \
  --run_dir weights --model_dir weights --lookup_csv weights/mws_static_lookup_UNSCALED.tsv
# --verbose adds the as-of / anchor / tile detail.
```

**API**
```bash
RUN_DIR=weights MODEL_DIR=weights LOOKUP_CSV=weights/mws_static_lookup_UNSCALED.tsv \
  CORESTACK_KEY=<key> PORT=8001 python inference/serve_api.py

curl -s localhost:8001/health
curl -s -X POST localhost:8001/forecast -H 'Content-Type: application/json' \
     -d '{"lat":25.44,"lon":91.71,"date":"2024-08-15"}'
```

Runnable end-to-end checks are in [`examples/`](../examples/).

**Request fields:** `lat` (−90..90, required), `lon` (−180..180, required), `date` (YYYY-MM-DD,
optional, default today, must be ≤ today). The seasonal-baseline and bias knobs are CLI-only
(`--baseline_years`, `--baseline_month_window`, `--bias`) — they are not exposed on the API request.

**Environment overrides (API):** `RUN_DIR`, `MODEL_DIR`, `LOOKUP_CSV`, `CACHE_DIR`, `DEVICE`, `PORT`,
`HOST`, `CORESTACK_KEY`.

---

## 6. Errors (API)

Clean, user-facing messages; the technical cause is logged server-side.

| status | meaning |
|--------|---------|
| 422 | invalid or future `date`; missing/invalid lat/lon; too little NDVI history |
| 502 | no coverage (ocean / no vegetation); no cloud-free tile; upstream fetch failed |
| 503 | model still loading |

---

## 7. Requirements

- Google Earth Engine credentials (`~/.config/earthengine/credentials`) and network access to GEE.
- Optional `CORESTACK_KEY` for admin lookups (absent → admin `"Unknown"`, forecast still runs).
- The model asset unzipped into `weights/` (see the repo README).
