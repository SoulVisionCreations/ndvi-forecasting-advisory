# Model Card — NDVI Forecasting (Prithvi + TFT)

A satellite-driven model that forecasts vegetation greenness (**NDVI**) about **three months ahead**
for micro-watersheds (MWS) across India, and turns that forecast into a relative
drought / vegetation-condition signal.

- **Task:** given ~11 months of history at a location, predict NDVI ~98 days (7 fortnights) ahead.
- **Headline result:** median-basin R² **0.586** (published seed).
- **Backbone:** NASA–IBM **Prithvi-EO-2.0-300M** geospatial foundation model (frozen).

---

## 1. Architecture

The model has three parts: a frozen image encoder, a small time-series forecaster, and a light
adapter that connects them.

```
  HLS composite tile (6 × 224 × 224)                 fortnightly history (≈11 months)
            │                                          NDVI + weather + static profile
   ┌────────▼─────────┐                                          │
   │ Prithvi-EO-2.0    │  frozen (330M params)          ┌────────▼────────────┐
   │ 300M encoder      │  + LoRA adapters (r16/α32,      │  TFT forecaster      │
   │ (ViT, MAE-pretr.) │    qkv, 24 blocks)              │  (ProTFT_Elite)      │
   └────────┬─────────┘                                  │  window = 24         │
        1024-d embedding                                 │  lead   = 7 (98 d)   │
            │  LayerNorm → Linear 1024→32                └────────┬────────────┘
            └──────────── 32-d image context ────────────────────┘
                                       │
                                 NDVI forecast at target date
```

- **Image encoder — Prithvi-EO-2.0-300M (frozen).** A Vision-Transformer geospatial foundation model
  pretrained (masked auto-encoding) on Harmonized Landsat–Sentinel-2 (HLS) imagery. It ingests one
  6-band `224×224` composite tile per location and produces a 1024-d embedding. The 330M backbone
  weights are **frozen**; we adapt it with **LoRA** (rank 16, α 32, on the q/k/v projections of all
  24 transformer blocks).
- **Image → context projector.** A pre-projection **LayerNorm** followed by a linear **1024 → 32**
  layer compresses the embedding into a 32-d "image context" vector.
- **Time-series forecaster — TFT (`ProTFT_Elite`).** A Temporal Fusion Transformer that consumes a
  **24-fortnight** window (≈11 months) of dynamic inputs plus static features, fused with the 32-d
  image context, and predicts NDVI **7 fortnights (98 days ≈ 3 months) ahead** of the last input.

**Inputs**
- *Dynamic (per fortnight):* NDVI and weather/land variables (precipitation, temperature, dewpoint,
  wind, solar & thermal radiation, evapotranspiration, runoff, soil moisture) plus engineered
  time/lag features.
- *Static (per location):* an MWS profile (seasonal NDVI baselines, quarter statistics, SPEI-3
  sensitivity, peak month) and administrative categoricals (state / district / tehsil).
- *Forecast statics:* aggregated forecast weather over the lead window.
- *Image:* one quarterly HLS composite tile for the target season.

**Output**
- `forecast_ndvi` at the target date, plus a **relative vegetation condition** — a z-score against the
  location's own history for that time of year, mapped to three classes
  (Below normal · Near normal · Above normal).

**Size.** One self-contained bundle of ~335M parameters (757 tensors): frozen backbone 330.7M,
TFT 2.40M, LoRA adapters 1.57M, projector + LayerNorm 0.035M. Only ~4.0M parameters are trainable.

**How the LoRA adapters load & adapt the encoder.** The bundle stores the frozen backbone **and** the
trained LoRA together, loaded in one pass:

1. Build the `PrithviMAE` structure, then **`get_peft_model`** wraps each block's `attn.qkv` with a LoRA
   branch — adding two small matrices per block, **A** `(r=16, 1024)` and **B** `(3072, r=16)`,
   zero-init so the adapter starts as a no-op — and rewires the forward.
2. A single **`load_state_dict(bundle, strict=False)`** then fills every slot at once: the frozen
   backbone `W`, the **trained** LoRA `A,B`, the LayerNorm, the projector, and the TFT.

At forward time each adapted projection computes a low-rank update added to the frozen weight:

```
qkv(x) = W_frozen·x  +  (α/r)·B·(A·x)        # α/r = 2.0 ;  B·A is a low-rank (3072×1024) delta on W
```

The LoRA weights are **48 tensors = 24 encoder blocks × {A, B}**, keyed
`prithvi.encoder.base_model.model.encoder.blocks.<N>.attn.qkv.lora_{A,B}.default.weight` — only the
attention **qkv**, all 24 blocks (nothing on the MLP). Because the backbone was **frozen** during
training, its tensors in the bundle are byte-identical to the base Prithvi weights, so the bundle is
self-contained (no separate base download). Code: `ndvi_core/prithvi_finetune.py::load_prithvi_lora`
(build + `get_peft_model`) and `ndvi_core/model_io.py::build_model` (the `load_state_dict`).

---

## 2. Data sources

**Training data**
- **NDVI** — MODIS `MOD13Q1` (16-day, 250 m), resampled to a 14-day fortnightly grid.
- **Weather / land** — ECMWF **ERA5-Land** (precipitation, temperature, radiation, ET, runoff, …).
- **Soil moisture** — NASA **SMAP** (`SPL4SMGP`).
- **Drought index** — **SPEI-3**, computed and joined into the numerical training table.
- **Imagery** — **HLS** (Harmonized Landsat–Sentinel-2) quarterly composites, 6-band `224×224` tiles
  at 30 m, used as the Prithvi encoder input.
- Packaged as `final_spei_output.csv` (numerical) + a location manifest; the ~755k image tiles are
  regenerated on demand (a generator script ships with the package rather than the tiles themselves).

**Inference data (fetched live, per request)**
- **Google Earth Engine (GEE)** — the numerical history (NDVI + weather) and the HLS composite tile.
- **CoreStack** — administrative lookup (state / district / tehsil) for a lat/lon.
- **Open-Meteo** — forward forecast weather over the lead window (with a climatology fallback where
  Open-Meteo is unreachable).

See `docs/DATA_SOURCES.md` for the full breakdown and reachability requirements.

---

## 3. Usage

The package ships a **CLI** and a **FastAPI** server that share one inference path. Point both at a
`weights/` folder containing the downloaded model asset. `date` must be **today or earlier**
(today = a live ~3-month forecast; a past date = a backtest).

**CLI**
```bash
python inference/infer_cli.py --lat 25.44 --lon 91.71 --date 2024-08-15 \
  --run_dir weights --model_dir weights --lookup_csv weights/mws_static_lookup_UNSCALED.tsv
```

**API**
```bash
RUN_DIR=weights MODEL_DIR=weights LOOKUP_CSV=weights/mws_static_lookup_UNSCALED.tsv \
  PORT=8001 python inference/serve_api.py
# GET  /health
# POST /forecast  {"lat":25.44,"lon":91.71,"date":"2024-08-15"}   (add ?debug=true for detail)
```

A successful response contains `location`, `admin`, `target_date`, `forecast_ndvi`,
`relative_vegetation_condition`, and an `anomaly` block. Runnable checks are in `examples/`; the full
request/response contract and the anchor/target date rules are in `docs/INFERENCE.md`. Training and
fine-tuning are in `docs/TRAINING.md`.

**Requirements:** Google Earth Engine credentials and network access to GEE (and CoreStack /
Open-Meteo for admin and forecast weather). The first request is slow (~30–60 s) because it fetches
~10 years of history live.

---

## 4. License & attribution

- **This model and code** are released under the **Apache License 2.0**.
- The fine-tuned weights are a **derivative of Prithvi-EO-2.0-300M** (NASA & IBM), which is itself
  released under Apache-2.0. We add LoRA adapters, a pre-projection LayerNorm + linear projector, and
  a TFT forecasting head; the base backbone weights are unchanged (frozen) and are distributed baked
  into the bundle — we do **not** redistribute them as a separate artifact.
- The **advisory layer** additionally uses **Gemma-3-4B** (Google **Gemma Terms of Use**) via **Ollama**
  (MIT) — referenced, not redistributed; text-only phraser, no role in the forecast.

**Please cite / credit:**
- **Prithvi-EO-2.0** — NASA–IBM geospatial foundation model. Paper: *Prithvi-EO-2.0: A Versatile
  Multi-Temporal Foundation Model for Earth Observation Applications*, arXiv:2412.02732.
- **MODIS** (`MOD13Q1`), **HLS**, and **SMAP** — NASA.
- **ERA5-Land** — ECMWF / Copernicus Climate Change Service.
- **CoreStack** — administrative boundaries (core-stack.org).
- **Open-Meteo** — forecast weather.
- **Google Earth Engine** — data access platform.

Each upstream dataset / service carries its own terms; users are responsible for complying with them
(most are free for research / non-commercial use, with attribution). See `NOTICE` for the full
attribution text.

---

## 5. Intended use

**Intended**
- Research and decision-support for **vegetation / drought early-warning** in India, at the
  micro-watershed scale, on a ~3-month horizon.
- Producing a **relative** vegetation-condition signal (how a location compares to its own seasonal
  normal), and backtesting past forecasts against observed NDVI.

**Out of scope / not intended**
- **Dates in the future** as the `date` input — the model forecasts ~3 months ahead of the most
  recent real data, so `date` must be today or earlier (future dates are rejected).
- **Outside India** — the administrative categoricals and MWS profiles are trained on Indian
  locations; other regions are out of distribution.
- **Absolute greenness thresholds** — the condition label is *relative* to each location's own
  seasonal history, not an absolute "NDVI < X = drought" rule.
- **Sub-fortnight or single-pixel precision**, real-time alerting guarantees, or use as the sole basis
  for high-stakes operational decisions without human review.

Forecast quality depends on live data being reachable and reasonably complete; where forecast weather
is unavailable it falls back to climatology, which can bias high-vegetation / monsoon targets.

---

## 6. Downstream use case — Vegetation Outlook advisory

The flagship consumer of this model is the **Vegetation Outlook advisory** (`advisory/`) — a farmer-facing
layer that turns the ~3-month NDVI forecast into a soft, area-relative vegetation-condition **opinion**
+ stress-mitigation levers. It **reuses this model unchanged** (loaded in-process — no separate forecast
service), samples ~12 cropland points, and a **leashed Gemma-3-4B** (via Ollama) only *rewords* the
deterministic message (**no role in the forecast numbers**; template fallback).
See [`advisory/README.md`](advisory/README.md) and [`advisory/PIPELINE.md`](advisory/PIPELINE.md).

Advisory-specific limitations (beyond the model's): clear-view fraction is a proxy; recent trend is often
unknown at serve time; local wet/dry uses a coarse regional rule; unmapped tehsils fall back to a
nearest-cropland-tile proxy.
