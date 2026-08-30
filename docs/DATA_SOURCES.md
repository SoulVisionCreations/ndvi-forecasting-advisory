# Data sources

The model ships **weights and static lookups**; all per-request **input data is fetched live**. The
CLI and the API use the same fetch path (`shared_data_layer.fetch_all` + `ndvi_core.download`).

---

## Live, per request

| source | what it provides | notes |
|--------|------------------|-------|
| **Google Earth Engine (GEE)** | (a) numerical history — NDVI + weather/ERA5, ~10 years (267 fortnights); (b) the HLS composite tile for the target season | dominates latency (~30–60 s, variable). Needs GEE credentials + network. |
| **CoreStack** | administrative lookup — state / district / tehsil for the lat/lon | if unreachable, admin = `"Unknown"` and the forecast still runs (with a small effect on the result). |
| **Open-Meteo** | forward **forecast weather** over the lead window (the "forecast statics") | falls back to **climatology** where Open-Meteo is unreachable. |

### GEE collections used

- **NDVI** — MODIS `MOD13Q1` (061), 16-day / 250 m, sampled onto the 14-day fortnightly grid.
- **Weather / land** — ECMWF `ERA5_LAND/DAILY_AGGR`: temperature, dewpoint, wind (u/v), solar &
  thermal radiation, precipitation, runoff, evapotranspiration.
- **Soil moisture** — NASA `SMAP` `SPL4SMGP` (008): surface + root-zone.
- **Imagery** — HLS (Harmonized Landsat–Sentinel-2) → a 6-band `224×224` quarterly composite
  (~30 m; a ~6.7 km box), selected by a walk-back over prior years until a sufficiently cloud-free
  tile is found.

---

## Local (shipped with the Model asset, not fetched)

Loaded from `weights/`:

- `tft_temporal_production_ft.pt` — the model bundle (frozen Prithvi backbone + LoRA + projector + TFT).
- `standard_scaler_temporal_tft_ft.pkl`, `label_encoders_temporal_tft_ft.pkl` — feature scaler + label encoders.
- `train_config.json` — window / lead / architecture config (travels with the model).
- `mws_static_lookup_UNSCALED.tsv` — per-MWS static profile (seasonal NDVI baselines, quarter stats,
  SPEI-3 sensitivity, peak month); nearest-MWS is matched to the request lat/lon.
- `prithvi_mae.py`, `config.json` — the Prithvi architecture code + base config (mean / std / dims).

---

## The climatology fallback (and the serve-time bias)

The forecast needs weather **over the future lead window**, which the model saw as *actual* values in
training. At serve time the future is unknown, so:

- where **Open-Meteo** is reachable → forward forecast weather is used;
- where it is **not** reachable → a **climatology** estimate (day-of-year-matched weather from prior
  years) is used instead.

The climatology path is the main driver of a documented serve-time bias — it tends to over-predict
high-vegetation / monsoon targets. This is expected behavior, not a bug; treat the **relative**
condition (z-score) as more robust than the absolute NDVI when the forecast weather is climatology.

---

## Requirements summary

- **GEE credentials** (`~/.config/earthengine/credentials`) and network access to GEE — required.
- **CoreStack** reachable — recommended (else admin `"Unknown"`).
- **Open-Meteo** reachable — recommended (else climatology + a larger serve bias).

Secrets are provided by environment variable only (e.g. `CORESTACK_KEY`); none are committed.
