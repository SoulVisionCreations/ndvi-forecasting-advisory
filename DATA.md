# Dataset — NDVI / SPEI fortnightly panel (India)

The **Dataset asset** for this project: the numerical training data plus the location manifest needed
to **fine-tune or retrain** the model. **Inference does not need this** — only `training/` does
(see `docs/TRAINING.md`).

- **License:** CC BY 4.0.
- **Coverage:** micro-watersheds (MWS) across India (~24,800 in the manifest; the model was trained on
  the ~21,900 with usable history).
- **Cadence:** 14-day "fortnightly" grid; multi-year history through 2023.

---

## Files

Download these into `data/` (they are not committed to git):

| file | what it is |
|------|-----------|
| `final_spei_output.csv` | the numerical panel — one row per (MWS, fortnight) |
| `all_mws_locations_dedup.csv` | the location manifest — `mws_id, lat, lon` (~24,800 rows) |

The ~755k Prithvi image tiles are **not** part of this dataset — they are regenerated from the
manifest with `training/download_composites.py` (the package ships the recipe, not the tiles).

---

## `final_spei_output.csv` — columns

One row per micro-watershed per 14-day fortnight. Columns (grouped):

- **Identity / time** — MWS id, latitude / longitude, date (fortnight).
- **Vegetation** — `NDVI`.
- **Weather / land (ERA5-Land)** — precipitation, temperature, dewpoint, wind (u/v components),
  downward solar radiation, net thermal radiation, evapotranspiration, runoff.
- **Soil moisture (SMAP)** — surface and root-zone.
- **Drought index** — `SPEI-3` (a 3-month standardized precipitation-evapotranspiration index,
  ≈ 7 fortnights).

(See the CSV header for the exact column names; the training code reads the columns it needs by name
via `ndvi_core.config`.)

---

## Provenance

- **NDVI** — MODIS `MOD13Q1` (16-day, 250 m) resampled onto the 14-day fortnightly grid.
- **Weather / land** — ECMWF **ERA5-Land** (precipitation is ERA5, not CHIRPS).
- **Soil moisture** — NASA **SMAP**.
- **SPEI-3** — computed from the precipitation / evapotranspiration series and joined in.
- The numerical history ends in **2023**.

**Train vs. serve collection versions (for reproducibility).** The training table was built with
MODIS `MOD13Q1` collection **006** and SMAP **007**; live inference now reads the current collections
(`MOD13Q1` **061**, SMAP **008**). This is a deliberate, documented train/serve delta — the newer
collections are the maintained ones, and the differences are small relative to the model's signal.

---

## How it's used

```bash
# 1. download the two files into data/
# 2. regenerate the image tiles from the manifest (GEE creds needed):
python training/download_composites.py \
  --mws_csv data/all_mws_locations_dedup.csv --out_dir quarterly_composites \
  --years 2016 2017 2018 2019 2020 2021 2022 2023 2024 --quarters Q1 Q2 Q3 Q4 \
  --min_coverage 0.75 --workers 1
# 3. train / fine-tune (see docs/TRAINING.md)
```

---

## Attribution

Derived from MODIS (NASA), ERA5-Land (ECMWF / Copernicus), and SMAP (NASA). See `NOTICE` for full
attribution. When you use this dataset, please credit those upstream providers and this project.
