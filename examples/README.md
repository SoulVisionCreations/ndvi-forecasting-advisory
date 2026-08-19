# examples/

Quick, runnable checks to confirm your **NDVI_Forecasting** setup works — both the CLI and the API.

These are **inputs and scripts only**. They print whatever your install returns; there are no
baked-in "expected" numbers to match, because the forecast depends on live satellite/weather data.
If a run comes back with a populated `forecast_ndvi` and a `relative_vegetation_condition`, the
setup is working.

## Prerequisites

- Package installed: `uv pip install -e .` (see the repo `README` / `env/SETUP.md`).
- The model asset unzipped into `weights/` — 7 files:
  `tft_temporal_production_ft.pt`, `standard_scaler_temporal_tft_ft.pkl`,
  `label_encoders_temporal_tft_ft.pkl`, `train_config.json`,
  `mws_static_lookup_UNSCALED.tsv`, `prithvi_mae.py`, `config.json`.
- Google Earth Engine credentials (`~/.config/earthengine/credentials`) and network access to GEE.
- Optional: `export CORESTACK_KEY=<key>` for admin (state / district / tehsil) lookups. Without it,
  admin comes back `"Unknown"` but the forecast still runs.

Notes:
- `date` must be **today or earlier** (a future date is rejected).
- The first request is slow (~30–60s) — it fetches ~10 years of history from GEE live.

## Files

| file | what it is |
|------|------------|
| `sample_points.csv`     | a few `lat,lon,date` rows across India to run |
| `forecast_request.json` | one sample POST body for the API |
| `run_cli_examples.sh`   | runs the CLI over `sample_points.csv` |
| `call_api.sh`           | curl the API: `/health` + `/forecast` (default and `?debug=true`) |
| `call_api.py`           | the same via Python `requests`, looping `sample_points.csv` |

## 1) CLI

```bash
# from the repo root, with weights/ populated
bash examples/run_cli_examples.sh

# ...or a single point:
python inference/infer_cli.py --lat 25.44 --lon 91.71 --date 2024-08-15 \
  --run_dir weights --model_dir weights --lookup_csv weights/mws_static_lookup_UNSCALED.tsv
```

## 2) API

```bash
# start the server first (see the repo README), e.g.:
#   RUN_DIR=weights MODEL_DIR=weights LOOKUP_CSV=weights/mws_static_lookup_UNSCALED.tsv \
#   PORT=8001 python inference/serve_api.py

bash   examples/call_api.sh                 # or: bash examples/call_api.sh http://HOST:PORT
python examples/call_api.py                 # or: python examples/call_api.py http://HOST:PORT
```

## What a healthy response looks like (shape, not values)

A successful `POST /forecast` returns:

```jsonc
{
  "location": { "lat": <num>, "lon": <num> },
  "admin": { "state": <str>, "district": <str>, "tehsil": <str> },
  "target_date": "YYYY-MM-DD",               // = date + ~3 months
  "forecast_ndvi": <num>,                    // 0..1
  "relative_vegetation_condition": <str>,    // Severe/Moderate deficit · Near normal · Above/Well above normal
  "anomaly": { "z": <num>, "ndvi": <num>, "pct": <num>,
               "baseline_mean": <num>, "baseline_std": <num> }
}
```

Add `?debug=true` to also see `as_of`, `anchor_date`, `prithvi_tile`, `anomaly.baseline_source`,
`anomaly.baseline_n`, and a staleness `note` when the anchor imagery was carried forward.
