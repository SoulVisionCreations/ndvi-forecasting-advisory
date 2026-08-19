# Training & fine-tuning

How to reproduce (or re-train) the champion model: a **from-scratch, one-phase fine-tune** that
jointly trains the TFT forecaster + the Prithvi **LoRA adapters** + the projector, on top of the
**frozen** Prithvi-EO-2.0-300M backbone.

> Inference is self-contained (the frozen backbone is baked into the shipped bundle). **Training is
> not** — it starts from the *base* Prithvi checkpoint, which you obtain separately (see prerequisites).

---

## 1. What the champion is

- **quarterly-w24 TFT + Prithvi LoRA**, `WINDOW=24` fortnights (≈11 months), `LEAD=7` (98 days ≈ 3 mo).
- One-phase fine-tune (no separate embedding pre-pass): `--tile_source folder` + `--prithvi_norm layernorm`.
- LoRA r16 / α32 on q/k/v across all 24 encoder blocks; a 1024→32 projector with a pre-projection
  LayerNorm; the TFT is `ProTFT_Elite`.
- Trainable ≈ 4.0M params (TFT 2.40M + LoRA 1.57M + projector 0.035M); backbone 330M frozen.
- Result: median-basin R² ≈ **0.586** (published seed s42).

Driver: `training/Run_With_MWS_Split_Temporal_TFT_FT.py`.

---

## 2. Prerequisites

**Environment** — install the training extras:
```bash
uv pip install -e ".[train]"      # adds matplotlib / seaborn / tqdm on top of the inference deps
```

**Data & assets**
| what | example path | notes |
|------|--------------|-------|
| numerical training table | `final_spei_output.csv` | NDVI + ERA5 weather + SPEI-3, fortnightly. The **Dataset** asset. |
| location manifest        | `all_mws_locations_dedup.csv` | columns `mws_id, lat, lon`. |
| quarterly composite tiles | `quarterly_composites/` | `composite_<mws_id>_<year>_<Q>.tif`, 6-band 224×224 int16. Regenerate — see step 3. |
| **base** Prithvi checkpoint | `Prithvi-EO-2.0-300M-TL/` | `Prithvi_EO_V2_300M_TL.pt` + `prithvi_mae.py`. Download from Hugging Face / NASA (**not** shipped in the Model asset). |

A GPU is required (the champion trained on a single high-memory GPU). GEE credentials are only needed
for step 3 (regenerating tiles), not for training itself.

---

## 3. Get the composite tiles

`--tile_source folder` scans `COMPOSITE_DIR` for `composite_<mws_id>_<year>_<Q>.tif`. The full set is
~755k tiles and is **not** shipped — regenerate it with the same recipe used to build it:

```bash
export GEE_PROJECT=<your-gee-project>
python training/download_composites.py \
  --mws_csv all_mws_locations_dedup.csv --out_dir quarterly_composites \
  --years 2016 2017 2018 2019 2020 2021 2022 2023 2024 --quarters Q1 Q2 Q3 Q4 \
  --min_coverage 0.75 --workers 1
```

- Writes one `.tif` per (MWS, year, quarter). `--limit N` restricts to the first N MWS rows (handy for
  a quick test). Keep `--workers` at 1–2 — GEE throttles higher concurrency to all-NaN.
- `emb_lag=1` (a training arg, below) means the tile used for a target year is the **previous** year's
  quarter, so you need tiles back to at least `min(target_year) − 1`.

---

## 4. Train + fine-tune

```bash
export DATA_PATH=final_spei_output.csv                 # numerical (NDVI + ERA5 + SPEI)
export COMPOSITE_DIR=quarterly_composites              # the tiles from step 3
export MWS_CSV=all_mws_locations_dedup.csv             # mws_id, lat, lon
export MODEL_DIR=Prithvi-EO-2.0-300M-TL                # base Prithvi checkpoint + prithvi_mae.py
export OUT_ROOT=runs/my_run                            # artifacts land here
export WINDOW=24 LEAD=7 SEED=42 EPOCHS=20              # champion ran 18 epochs; keeps the BEST ckpt
export CUDA_VISIBLE_DEVICES=0

python -u training/Run_With_MWS_Split_Temporal_TFT_FT.py \
  --emb_mode quarterly --tile_source folder --warmstart none --prithvi_norm layernorm \
  --ft_lr 1e-4 --forecaster_lr_scale 10 --n_tiles 128 --samples_per_tile 32 \
  --lora_r 16 --lora_alpha 32 --lora_target qkv --emb_proj_dim 32 --emb_lag 1 --doy 182
```

Key arguments:
- `--tile_source folder` + `--prithvi_norm layernorm` — the champion (GWL-style) one-phase path.
- `--ft_lr 1e-4` (LoRA + projector) with `--forecaster_lr_scale 10` (so the TFT trains at ~1e-3).
- `--lora_r 16 --lora_alpha 32 --lora_target qkv` — the LoRA adapters.
- `--emb_proj_dim 32` — the image-context width.
- `--emb_lag 1` — the tile year is the target year minus 1. `--doy 182` — the composite day-of-year anchor.
- `EPOCHS=20`, but the run keeps the **best** checkpoint by validation median-basin R² (the champion's
  best was around epoch 18).

---

## 5. Outputs

Everything lands in `OUT_ROOT` (basenames get the `_ft` suffix; the `_ft.pt` **is** the served bundle):

```
tft_temporal_production_ft.pt          # frozen backbone + LoRA + projector + TFT (self-contained)
standard_scaler_temporal_tft_ft.pkl
label_encoders_temporal_tft_ft.pkl
train_config.json                      # window / lead / arch — travels with the model
training_history_temporal_tft_ft.csv   # per-epoch metrics
plots_tft_ft/
```

---

## 6. Serve your model

Copy the four artifacts into a `weights/` folder alongside the two shipped Prithvi files, then run
inference exactly as in [`INFERENCE.md`](INFERENCE.md):

```
weights/
  tft_temporal_production_ft.pt          # from your run
  standard_scaler_temporal_tft_ft.pkl    # from your run
  label_encoders_temporal_tft_ft.pkl     # from your run
  train_config.json                      # from your run
  mws_static_lookup_UNSCALED.tsv         # static profile (ships with the Model asset)
  prithvi_mae.py                         # ships with the Model asset
  config.json                            # ships with the Model asset
```

```bash
python inference/infer_cli.py --lat 25.44 --lon 91.71 --date 2024-08-15 \
  --run_dir weights --model_dir weights --lookup_csv weights/mws_static_lookup_UNSCALED.tsv
```

The self-contained loader takes the frozen backbone from your fine-tuned bundle (no base Prithvi
download needed at inference time).

---

## 7. Notes

- **Determinism.** Deep-learning training on GPU is non-deterministic, so a full retrain reproduces the
  champion **quality** (~0.586) within seed / GPU variance, **not** bit-for-bit. The release ships the
  champion `.pt` itself, so exact reproduction is not required to use the model.
- **Tiles.** `composite_<mws_id>_<year>_<Q>.tif` are 6-band 224×224 int16 (HLS reflectance ×10000).
- **Parquet is not used here.** `embeddings_quarterly.parquet` is only for the alternative two-phase
  path (`--tile_source parquet` + `--prithvi_norm musd`); the champion (`folder` + `layernorm`) does
  not read it.
- **Verified** (fresh-clone dogfood): this exact command reproduced the champion config, and a 1-epoch
  smoke run produced loadable artifacts that ran end-to-end through `infer_cli`.
