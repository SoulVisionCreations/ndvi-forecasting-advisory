# weights/

Holds the **model artifacts** — i.e. the **AIKosh Model asset**. These files are
NOT committed to git; you download them into this folder.

Expected contents (the *self-contained* model asset — nothing external needed):

```
tft_temporal_production_ft.pt        # fused bundle: frozen Prithvi backbone + LoRA + projector + TFT
standard_scaler_temporal_tft_ft.pkl  # feature scaler
label_encoders_temporal_tft_ft.pkl   # categorical encoders (state / district / tehsil)
train_config.json                    # model config (window / lead / proj / lora / …)
mws_static_lookup_UNSCALED.tsv       # per-MWS static feature lookup (TSV; AIKosh Model rejects CSV)
prithvi_mae.py                       # Prithvi architecture code (Apache-2.0, IBM/NASA)
config.json                          # base Prithvi config (mean / std / dims)
```

**How to get them**
- Download the zip from the **AIKosh Model** page —
  <https://aikosh.indiaai.gov.in/web/models/details/ndvi_forecasting_model.html> — then extract it
  into this folder: `python -m zipfile -e /path/to/ndvi_aikosh_model.zip weights/`.
- Local testing: copy the extracted model files into this folder (from wherever you staged them).

The AIKosh **Dataset** asset is a *separate* download and is **not** needed to serve the advisory or
run inference — only the train/fine-tune pipeline (`training/`, `data/`) uses it.

**Why self-contained:** the loader treats this folder as `MODEL_DIR` — it builds the
Prithvi architecture from `prithvi_mae.py` + `config.json` and fills **every** weight
from the fused bundle. The Prithvi backbone is frozen, so its weights inside the
bundle are byte-identical to the base model and **no separate base checkpoint is
needed** (no Hugging Face download).
