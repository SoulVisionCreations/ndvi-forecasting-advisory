"""
ndvi_core.model_io — load the champion TFT+Prithvi bundle for inference.

Reconstructs the EXACT graph the training driver built (ProTFT_Elite +
LivePrithviProjector over a folder TileStore) and loads
`tft_temporal_production_ft.pt` into it. For inference the TileStore points at the
local CACHE dir (freshly fetched composites), not the training tile folder.

Mirrors the MODEL INIT block of Run_With_MWS_Split_Temporal_TFT_FT.py so the
served model is byte-identical to the trained one.
"""

import json
import os

import joblib
import torch

from . import prithvi_finetune as pf   # folded into the engine (shared inference+train dep)
from . import config
from . import models_tft as ncm


def load_train_config(run_dir):
    """Per-model training config that travels WITH the model: `train_config.json`
    written into the run dir at train time (window/lead/emb/lora/...). Falls back
    to config.py defaults when absent (e.g. the champion, trained before this was
    added — its defaults ARE the champion's values). So inference always uses the
    served model's OWN window/lead, not a hardcoded constant.

    Returns a dict; the geometry-critical keys are `window` and `lead`."""
    cfg = {
        "window": config.WINDOW, "lead": config.LEAD,
        "emb_mode": "quarterly", "emb_proj_dim": config.EMB_PROJ_DIM,
        "emb_lag": 1, "doy": config.PRITHVI_DOY,
        "lora_r": config.LORA_R, "lora_alpha": config.LORA_ALPHA,
        "lora_dropout": config.LORA_DROPOUT, "lora_target": config.LORA_TARGET,
        "prithvi_norm": config.PRITHVI_NORM,
    }
    path = os.path.join(run_dir, "train_config.json")
    if os.path.exists(path):
        with open(path) as f:
            cfg.update(json.load(f))
    return cfg


def load_artifacts(scaler_path, encoder_path):
    """Load the saved StandardScaler + label-encoders dict."""
    return joblib.load(scaler_path), joblib.load(encoder_path)


def emb_configs_from_encoders(encoders, dim=config.CAT_EMB_DIM):
    """(num_classes, emb_dim) per categorical, in the training order."""
    return [(len(encoders[c].classes_), dim) for c in ("state", "district", "tehsil")]


def build_model(model_dir, ckpt_path, encoders, cache_dir, latlon,
                device="cuda", proj_dim=config.EMB_PROJ_DIM, lora_r=config.LORA_R,
                lora_alpha=config.LORA_ALPHA, lora_dropout=config.LORA_DROPOUT,
                lora_target=config.LORA_TARGET, doy=config.PRITHVI_DOY,
                encode_chunk=128, norm=config.PRITHVI_NORM, tcfg=None):
    """Reconstruct the trained graph and load the bundle.

    cache_dir MUST already contain the fetched composite tile(s) (the TileStore
    builds its key table by scanning that folder). latlon maps each cached tile's
    mws_id -> (lat, lon). Returns (model, key_to_row, zero_idx); map a tile key
    (mws_id, year, 'Qn') -> emb_idx via key_to_row, then call
    model(x_dyn, x_stat_num, x_stat_cat, emb_idx=[idx])."""
    # Architecture from the model's OWN train_config.json (tcfg) takes precedence
    # over config.py, so a served model is built with the arch it was TRAINED with
    # (not whatever config.py currently says). Champion: identical either way.
    if tcfg:
        proj_dim = int(tcfg.get("emb_proj_dim", proj_dim))
        lora_r = int(tcfg.get("lora_r", lora_r))
        lora_alpha = int(tcfg.get("lora_alpha", lora_alpha))
        lora_dropout = float(tcfg.get("lora_dropout", lora_dropout))
        lora_target = tcfg.get("lora_target", lora_target)
        doy = int(tcfg.get("doy", doy))
        norm = tcfg.get("prithvi_norm", norm)
    # 1) frozen Prithvi backbone + LoRA adapters (zero-init; bundle overwrites them)
    lora_model, cfg = pf.load_prithvi_lora(
        model_dir, r=lora_r, alpha=lora_alpha, dropout=lora_dropout,
        num_frames=1, device=device, target=lora_target, exclude_ckpt=ckpt_path)
    emb_dim = int(cfg["embed_dim"])

    # 2) TileStore over the CACHE dir (the fetched tiles), period-aware doy
    tile_keys, key_to_row, zero_idx = pf.build_tile_table_from_folder(cache_dir)
    store = pf.TileStore(
        tile_keys, cache_dir, latlon, cfg["mean"], cfg["std"],
        doy=doy, cache_size=4096, n_threads=4)

    # 3) live projector (layernorm mode -> mu/sd unused; weights come from bundle)
    projector = pf.LivePrithviProjector(
        lora_model, store, None, None, zero_idx,
        in_dim=emb_dim, proj_dim=proj_dim, pool=config.PRITHVI_POOL,
        encode_chunk=encode_chunk, amp=True, device=device, norm=norm)

    # 4) the forecaster, identical args to training MODEL INIT (all from config)
    model = ncm.ProTFT_Elite(
        dynamic_input_size=len(config.DYNAMIC_FEATURES),
        num_numerical_static=len(config.STATIC_NUMERICAL_FEATURES),
        embedding_configs=emb_configs_from_encoders(encoders),
        hidden_size=config.HIDDEN_SIZE, num_heads=config.NUM_HEADS,
        d_model=config.D_MODEL, lstm_layers=config.LSTM_LAYERS,
        tft_dropout=config.TFT_DROPOUT, prithvi_projector=projector).to(device)
    print(f"[model_io] TFT head built (ProTFT_Elite) | hidden={config.HIDDEN_SIZE} "
          f"heads={config.NUM_HEADS} d_model={config.D_MODEL} lstm_layers={config.LSTM_LAYERS} "
          f"window={config.WINDOW} lead={config.LEAD}")

    # 5) load the bundle (TFT + projector/LayerNorm + LoRA + frozen backbone)
    sd = torch.load(ckpt_path, map_location=device, weights_only=True)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    # The only acceptable gaps are PEFT/pos_embed buffers; loud if a real weight is missing.
    real_missing = [k for k in missing if "lora" not in k.lower() and "pos_embed" not in k]
    if real_missing:
        print(f"[model_io] WARNING missing {len(real_missing)} non-LoRA keys, e.g. {real_missing[:5]}")
    if unexpected:
        print(f"[model_io] note: {len(unexpected)} unexpected keys (e.g. {unexpected[:3]})")
    model.eval()
    print(f"[model_io] loaded bundle | emb_dim {emb_dim} | {len(tile_keys)} cached tile(s)")
    return model, key_to_row, zero_idx


def refresh_tile_store(model, cache_dir, latlon, doy=config.PRITHVI_DOY,
                       cache_size=4096, n_threads=4):
    """Rebuild the projector's TileStore from the CURRENT cache_dir contents
    (after a new composite was fetched) WITHOUT reloading the backbone — for
    serving, where the model loads once and each request adds its own tile.

    Reuses the trained backbone's (mean, std) held on the existing store, so the
    normalization is unchanged. Returns the fresh (key_to_row, zero_idx) for the
    emb_idx lookup. Not thread-safe: serialize with the model forward."""
    proj = model.prithvi
    old = proj.store
    mean = old.mean[:, 0, 0]                     # recover 1-D backbone mean/std
    std = old.std[:, 0, 0]
    tile_keys, key_to_row, zero_idx = pf.build_tile_table_from_folder(cache_dir)
    proj.store = pf.TileStore(
        tile_keys, cache_dir, latlon, mean, std,
        doy=doy, cache_size=cache_size, n_threads=n_threads)
    proj.zero_idx = int(zero_idx)
    return key_to_row, zero_idx
