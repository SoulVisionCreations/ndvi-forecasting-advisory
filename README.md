# NDVI_Forecasting

Fortnightly **NDVI drought forecasting for India** — a from-scratch Temporal Fusion Transformer (TFT)
forecaster on top of a **frozen Prithvi-EO-2.0-300M** geospatial foundation model (LoRA-adapted). For
any `(lat, lon, date)` in India it forecasts vegetation condition **~3 months ahead** from satellite +
weather history.

- **Model:** quarterly-window TFT + frozen Prithvi-300M (LoRA r16/α32, qkv) + 1024→32 projector.
  `WINDOW=24` fortnights (≈11 months), `LEAD=7` (98 days ≈ 3 months).
- **Output:** predicted NDVI + standardized seasonal anomaly (z) + a 3-class *relative
  vegetation-condition* label (Below / Near / Above normal).
- **Result:** median-basin R² ≈ 0.586.

📄 See **[MODEL_CARD.md](MODEL_CARD.md)** for the model overview, and **[docs/](docs/)** for details.

## Layout

| Folder | What |
|---|---|
| `ndvi_core/` | shared engine (config, features, tensors, TFT model, Prithvi loader, indicators) — imported by **both** paths |
| `inference/` | **run a forecast** — `infer_cli.py` (CLI) + `serve_api.py` (FastAPI); reused in-process by the advisory |
| `advisory/`  | **Vegetation Outlook advisory** (the use case) — `serve_advisory.py` (FastAPI, :8011) |
| `training/`  | **fine-tune / retrain** — the TFT + Prithvi training driver + tile tools |
| `weights/`   | the model artifacts (downloaded — see [`weights/README.md`](weights/README.md)) |
| `data/`      | dataset download / regeneration (not the raw data — see [`data/README.md`](data/README.md)) |
| `examples/`  | runnable CLI + API setup checks ([`examples/README.md`](examples/README.md)) |
| `docs/`      | [ADVISORY](docs/ADVISORY.md) · [INFERENCE](docs/INFERENCE.md) · [DATA_SOURCES](docs/DATA_SOURCES.md) · [TRAINING](docs/TRAINING.md) |

## Quickstart — Vegetation Outlook advisory (the use case)

The flagship use case: it turns the ~3-month NDVI forecast into a soft, area-relative
**vegetation-outlook opinion** + stress-mitigation levers. It loads the **same model in-process**
(no separate forecast service), and a leashed **Gemma-3-4B** (via [Ollama](https://ollama.com)) only
rewords the message (deterministic template fallback — `ADVISORY_SLM=0` skips it).

**In plain terms:** it forecasts greenness (NDVI) ~3 months out and compares it to *that spot's own
~10-year seasonal normal* — a **standardized seasonal anomaly** (z-score) = how far above/below usual,
in units of local variability (a **relative** read, not absolute NDVI). Across ~12 nearby cropland
points: the **median** z → severity (Below / Near / Above normal), the **spread** (SD) → confidence,
and your **plot vs the area** → whether it's area-wide (regional) or just your plot (local). Full
walkthrough: [docs/ADVISORY.md](docs/ADVISORY.md).

```bash
# install + populate weights/ (the Model asset) + creds — see "Configuration" below
uv venv --python 3.11 && source .venv/bin/activate && uv pip install -e .

# language model (Linux): install Ollama + pull Gemma
curl -fsSL https://ollama.com/install.sh | sh          # Ollama; serves :11434
ollama pull gemma3:4b                                  # ~3.3 GB — the phraser

# serve the advisory (from the repo root; loads the model on a GPU)
export RUN_DIR=weights MODEL_DIR=weights LOOKUP_CSV=weights/mws_static_lookup_UNSCALED.tsv \
       ADVISORY_ENGINE=inprocess CORESTACK_KEY=<key> CUDA_VISIBLE_DEVICES=0 \
       ADVISORY_SLM=1 ADVISORY_SLM_MODEL=gemma3:4b
python -m uvicorn advisory.serve_advisory:app --host 0.0.0.0 --port 8011  # run inside the activated venv
# POST /advisory {"lat":20.17,"lon":78.32,"regime":"rain-fed","date":"2018-06-15"}
#   -> { location, risk_level, confidence, message }
```

**Prefer a browser?** Open **`http://127.0.0.1:8011/docs`** (FastAPI's interactive Swagger UI) →
expand **`POST /advisory`** → **Try it out** → edit the JSON → **Execute**. `/health` is a plain GET
you can open directly; `/redoc` is a read-only reference. The service binds `127.0.0.1`, so to reach
a remote host's `/docs` from your machine, tunnel first: `ssh -L 8011:127.0.0.1:8011 <host>`.

**Full step-by-step local setup** (unzip · uv env · all env vars · start · browse): [`docs/LOCAL_SETUP.md`](docs/LOCAL_SETUP.md).
Advisory doc: [`docs/ADVISORY.md`](docs/ADVISORY.md). Architecture + rules: [`advisory/README.md`](advisory/README.md) · [`advisory/PIPELINE.md`](advisory/PIPELINE.md).
**Customizing the advisory** (which files to edit — thresholds, the rules table, levers, wording — and in what format, without retraining): [`advisory/CUSTOMIZE.md`](advisory/CUSTOMIZE.md).

## Run the model directly — forecast (User A)

The advisory above reuses this same forecast in-process; you can also run the model standalone:

```bash
# 1. install
uv venv --python 3.11 && source .venv/bin/activate && uv pip install -e .

# 2. populate weights/  (AIKosh Model asset: aikosh.indiaai.gov.in/web/models/details/ndvi_forecasting_model.html
#    -> python -m zipfile -e /path/to/ndvi_aikosh_model.zip weights/ ; see weights/README.md.
#    The AIKosh Dataset asset is NOT needed for inference/advisory — only User B's retraining below.)

# 3. forecast  (date must be today or earlier)
python inference/infer_cli.py --lat 25.44 --lon 91.71 --date 2024-08-15 \
  --run_dir weights --model_dir weights --lookup_csv weights/mws_static_lookup_UNSCALED.tsv
```

Or serve it:

```bash
RUN_DIR=weights MODEL_DIR=weights LOOKUP_CSV=weights/mws_static_lookup_UNSCALED.tsv \
  PORT=8001 python inference/serve_api.py
# GET /health ; POST /forecast {"lat":25.44,"lon":91.71,"date":"2024-08-15"}  (add ?debug=true)
```

More runnable checks: [`examples/`](examples/).  Full request/response contract: [docs/INFERENCE.md](docs/INFERENCE.md).

## Quickstart — fine-tune / retrain (User B)

```bash
uv pip install -e ".[train]"
# download the dataset (see data/README.md + DATA.md), regenerate tiles, then train:
python training/Run_With_MWS_Split_Temporal_TFT_FT.py --help
```

Full recipe (champion args, data prep, outputs, serving your model): [docs/TRAINING.md](docs/TRAINING.md).

## Configuration & credentials (via env — never committed)

**Credentials at a glance** — no keys are shipped in this repo:

| Service | Needed for | Key? | How to get it |
|---|---|---|---|
| **Google Earth Engine** | live satellite / weather imagery (always) | OAuth | **One-time:** register at [earthengine.google.com](https://earthengine.google.com) → *Get Started* (noncommercial = free) — creates/registers a **Cloud project** and gives its **id**. **Then:** `earthengine authenticate` **+ `earthengine set_project <id>`** (`authenticate` only signs the *user* in; the project registration + `set_project` are separate. Or use a service account for headless/server auth). |
| **CoreStack** | admin lookup (forecast) · **cropland sampling (advisory — required)** | `CORESTACK_KEY` | register at [dashboard.core-stack.org](https://dashboard.core-stack.org/) → generate an API key (guide: [core-stack.org/use-apis](https://core-stack.org/use-apis/)) |
| **Open-Meteo** | forecast weather | none | public API |
| **Ollama + `gemma3:4b`** | advisory phraser wording only (optional) | none | `ollama pull gemma3:4b`; `ADVISORY_SLM=0` skips it (template-only) |

- **Forecaster** (`inference/`): needs GEE; `CORESTACK_KEY` is **optional** (absent → admin `"Unknown"`, forecast still runs).
- **Advisory** (`advisory/`): additionally **requires `CORESTACK_KEY`** (cropland sampling) plus env `ADVISORY_ENGINE=inprocess`, `ADVISORY_SLM`, `ADVISORY_SLM_MODEL`, `ADVISORY_SLM_URL`.
- **`ADVISORY_FETCH_WORKERS`** (default `13`): satellite-tile downloads fired **concurrently** per request (the plot + ~12 sampled points). If Earth Engine throttles the burst, the response includes a top-level **`"warnings"`** array; the request still completes (throttled tiles retry with backoff). Seeing it often → **lower `ADVISORY_FETCH_WORKERS`** (e.g. `6`) and **restart**. Matters at scale too: N concurrent requests = `N × ADVISORY_FETCH_WORKERS` concurrent pulls.

**GPU is optional.** The in-process engine loads on GPU by default (`DEVICE=cuda`); it also runs on
**CPU** (`DEVICE=cpu`) — the deterministic decision layer is identical and the neural forecast matches
to within CPU-vs-GPU float rounding (just slower). See [docs/DATA_SOURCES.md](docs/DATA_SOURCES.md) for
what's fetched live.

## License

**Apache-2.0** (see [LICENSE](LICENSE)). Built on **Prithvi-EO-2.0-300M** (IBM / NASA, Apache-2.0;
arXiv:2412.02732). The frozen backbone **and** the trained LoRA adapters (r16/α32, qkv) + 1024→32
projector + TFT head are all fused into the single model checkpoint (`tft_temporal_production_ft.pt`) —
no separate base download. Full attribution — including the MODIS / ERA5-Land / SMAP / HLS / CoreStack /
Open-Meteo data sources — is in [NOTICE](NOTICE).

The advisory phraser uses **Gemma-3-4B** (Google **Gemma Terms of Use**) via **Ollama** (MIT) —
referenced, not redistributed; text-only, no role in the forecast.
