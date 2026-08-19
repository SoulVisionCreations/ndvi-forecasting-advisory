"""
ndvi_core — shared NDVI forecasting library.

Single source of truth for the feature contract, feature engineering, tensor
building, the TFT model, scaling, and bundle I/O. Imported by BOTH the training
drivers (Run_With_MWS_Split_Temporal_TFT_FT.py / _FT.py) and the inference
service, so train-time and serve-time run the exact same code (no drift).

Companion modules that already live alongside this package and are reused
directly: prithvi_finetune.py (load_prithvi_lora, LivePrithviProjector,
TileStore) and prithvi_embed.py (idx_for / make_key / year helpers).
"""
