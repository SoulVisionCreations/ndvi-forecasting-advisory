# ============================================================
# PRODUCTION TEMPORAL NDVI FORECASTING PIPELINE
# ============================================================
# FEATURES
# ------------------------------------------------------------
# ✅ STRICT TEMPORAL SPLIT (70/20/10 per MWS)
# ✅ EXPANDING HISTORY VALIDATION
# ✅ SPLIT-SAFE FORECAST HORIZONS
# ✅ LEAKAGE-SAFE FEATURE ENGINEERING
# ✅ TRAIN-ONLY IMPUTATION
# ✅ TRAIN-ONLY QUARTERLY BASELINES
# ✅ EXPLICIT LEAD-TIME GEOMETRY
# ✅ PRODUCTION-GRADE TENSOR GENERATION
# ✅ DYNAMIC FORECAST AGGREGATION
# ✅ FULL LOGGING + SAVING PRESERVED
# ============================================================

import warnings
warnings.filterwarnings("ignore")

import gc
import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import r2_score

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ============================================================
# CONFIG
# ============================================================

import os

# All configs are overridable via environment variables (see train_job.sh).
# The values below are the defaults used when the env var is not set.

DATA_PATH = os.environ.get('DATA_PATH', 'data/final_spei_output.csv')   # numerical dataset (Dataset asset)

WINDOW = int(os.environ.get('WINDOW', 18))
LEAD = int(os.environ.get('LEAD', 7))
FORECAST_HORIZON = int(os.environ.get('FORECAST_HORIZON', 7))

TRAIN_RATIO = float(os.environ.get('TRAIN_RATIO', 0.70))
VAL_RATIO = float(os.environ.get('VAL_RATIO', 0.20))
TEST_RATIO = float(os.environ.get('TEST_RATIO', 0.10))

TRAIN_BATCH = int(os.environ.get('TRAIN_BATCH', 8192))
VAL_BATCH = int(os.environ.get('VAL_BATCH', 2048))
TEST_BATCH = int(os.environ.get('TEST_BATCH', 2048))

EPOCHS = int(os.environ.get('EPOCHS', 100))
SEED = int(os.environ.get('SEED', 42))

MODEL_PATH = os.environ.get('MODEL_PATH', 'tft_temporal_production.pt')
SCALER_PATH = os.environ.get('SCALER_PATH', 'standard_scaler_temporal_tft.pkl')
ENCODER_PATH = os.environ.get('ENCODER_PATH', 'label_encoders_temporal_tft.pkl')
HISTORY_PATH = os.environ.get('HISTORY_PATH', 'training_history_temporal_tft.csv')

# Directory for all generated plots (suffixed + created below once flag is known).
OUTPUT_DIR = os.environ.get('OUTPUT_DIR', 'plots_tft')

# ------------------------------------------------------------
# PRITHVI STATIC EMBEDDING (Phase 1) — added vs the original script.
# --use_prithvi turns the frozen-encoder static embedding on; when OFF this
# script reproduces the baseline (same sample set, no embedding) so the two
# runs are an apples-to-apples ablation.
# ------------------------------------------------------------
import argparse
import os, sys
# repo root on sys.path so `from ndvi_core import ...` resolves when this script
# runs directly (python training/Run_...py); prithvi_embed stays a training/
# sibling (the script's own dir is already on sys.path when run directly).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import prithvi_embed as pe

# Shared NDVI core library (feature contract, feature-eng, tensors, TFT model,
# scaling) — the SAME modules the inference path imports, so train/serve run
# identical logic. See ndvi_core/.
from ndvi_core import config as ncfg
from ndvi_core import features as ncf
from ndvi_core import tensors as nct
from ndvi_core import models_tft as ncm
from ndvi_core import prithvi_finetune as pf   # folded into the engine (was top-level import)

_p = argparse.ArgumentParser(add_help=False)
_p.add_argument('--use_prithvi', action='store_true',
                help='(Phase 2 always uses Prithvi; flag kept for parity.)')
_p.add_argument('--emb_path', default=os.environ.get('EMB_PATH', 'embeddings.parquet'),
                help='Frozen embeddings.parquet — used ONLY to derive the (mu,sd) '
                     'standardizer and the tile table; values are re-encoded live.')
_p.add_argument('--emb_proj_dim', type=int, default=int(os.environ.get('EMB_PROJ_DIM', 32)))
_p.add_argument('--emb_lag', type=int, default=int(os.environ.get('EMB_LAG', 1)),
                help='Composite year = target year - emb_lag (Fix A, default 1).')
_p.add_argument('--emb_mode', default=os.environ.get('EMB_MODE', 'annual'),
                choices=['annual', 'halfyear', 'quarterly'],
                help='Tile granularity / lookup. annual (default): (mws, year-lag) '
                     'tile. halfyear/quarterly: prior-year SAME-SEASON (H1/H2 or '
                     'Q1-4) walk-back, matching the _PRITHVI runner. Point '
                     '--composite_dir at the matching tiles.')
# --- Phase 2 (fine-tune) ---
_p.add_argument('--model_dir', default=os.environ.get('MODEL_DIR',
                'Prithvi-EO-2.0-300M-TL'))          # base Prithvi checkpoint (separate download)
_p.add_argument('--composite_dir', default=os.environ.get('COMPOSITE_DIR',
                'data/quarterly_composites'))       # tiles (regenerate via download_composites.py)
_p.add_argument('--mws_csv', default=os.environ.get('MWS_CSV',
                'data/all_mws_locations_dedup.csv'))
_p.add_argument('--doy', type=int, default=int(os.environ.get('DOY', 182)))
_p.add_argument('--lora_r', type=int, default=int(os.environ.get('LORA_R', 16)))
_p.add_argument('--lora_alpha', type=int, default=int(os.environ.get('LORA_ALPHA', 32)))
_p.add_argument('--lora_dropout', type=float, default=float(os.environ.get('LORA_DROPOUT', 0.05)))
_p.add_argument('--lora_target', default=os.environ.get('LORA_TARGET', 'qkv'))
_p.add_argument('--warmstart', default=os.environ.get('WARMSTART', 'none'),
                help='Phase-1 +Prithvi checkpoint to warm-start forecaster+proj, or '
                     '"none" (DEFAULT on this branch) to train the TFT from scratch '
                     '(GWL-style joint train). For the two-phase path pass the '
                     'Phase-1 .pt (and use --prithvi_norm musd --tile_source parquet).')
_p.add_argument('--ft_lr', type=float, default=float(os.environ.get('FT_LR', 1e-4)),
                help='LR for LoRA + projection head.')
_p.add_argument('--forecaster_lr_scale', type=float,
                default=float(os.environ.get('FORECASTER_LR_SCALE', 10.0)),
                help='Forecaster LR = ft_lr * this. From-scratch TFT -> 10 (fast, '
                     'GWL-style, DEFAULT); warm-started TFT -> 0.1 (gentle refine).')
_p.add_argument('--tile_grouped', dest='tile_grouped', action='store_true', default=True,
                help='Tile-grouped batches (default ON): few unique tiles/batch -> fast.')
_p.add_argument('--no_tile_grouped', dest='tile_grouped', action='store_false',
                help='Per-sample random batches (exact epoch; slower; use small TRAIN_BATCH).')
_p.add_argument('--n_tiles', type=int, default=int(os.environ.get('N_TILES', 256)))
_p.add_argument('--samples_per_tile', type=int, default=int(os.environ.get('SAMPLES_PER_TILE', 32)))
_p.add_argument('--encode_chunk', type=int, default=int(os.environ.get('ENCODE_CHUNK', 256)),
                help='Max unique tiles encoded per forward chunk (caps activation memory).')
_p.add_argument('--tile_cache', type=int, default=int(os.environ.get('TILE_CACHE', 20000)))
_p.add_argument('--prithvi_norm', choices=['layernorm', 'musd'],
                default=os.environ.get('PRITHVI_NORM', 'layernorm'),
                help='Pre-projection norm of the pooled Prithvi vector. "layernorm" '
                     '(DEFAULT, GWL-style): learned LayerNorm, no parquet (mu,sd). '
                     '"musd": fixed offline (mu,sd) from the parquet (two-phase path).')
_p.add_argument('--tile_source', choices=['folder', 'parquet'],
                default=os.environ.get('TILE_SOURCE', 'folder'),
                help='Source of the sample->tile table. "folder" (DEFAULT, GWL-style): '
                     'scan --composite_dir, no parquet. "parquet": keys from --emb_path.')
_p.add_argument('--validate_tile_table', default=os.environ.get('VALIDATE_TILE_TABLE', ''),
                help='Optional parquet path. If set, build the tile table from BOTH '
                     'the folder and this parquet and assert per-sample tile '
                     'resolution is identical, then continue. One-time check.')
_p.add_argument('--eval_only', action='store_true',
                help='Load the trained FT checkpoint (MODEL_PATH), run the test '
                     'eval (+persistence/delta/anomaly), then exit — no training.')
_args, _ = _p.parse_known_args()

USE_PRITHVI = True                              # Phase 2 requires the encoder in-graph
EMB_PATH = _args.emb_path
EMB_PROJ_DIM = _args.emb_proj_dim
EMB_LAG = _args.emb_lag
EMB_MODE = _args.emb_mode
MODEL_DIR = _args.model_dir
COMPOSITE_DIR = _args.composite_dir
MWS_CSV = _args.mws_csv
DOY = _args.doy
LORA_R, LORA_ALPHA, LORA_DROPOUT, LORA_TARGET = (
    _args.lora_r, _args.lora_alpha, _args.lora_dropout, _args.lora_target)
WARMSTART = _args.warmstart
if WARMSTART is not None and WARMSTART.lower() in ('', 'none'):
    WARMSTART = None
EVAL_ONLY = _args.eval_only or os.environ.get('EVAL_ONLY', '0') == '1'
if EVAL_ONLY:
    WARMSTART = None   # eval loads the trained FT checkpoint directly, not a warm-start
FT_LR = _args.ft_lr
FORECASTER_LR_SCALE = _args.forecaster_lr_scale
TILE_GROUPED = _args.tile_grouped
N_TILES = _args.n_tiles
SAMPLES_PER_TILE = _args.samples_per_tile
ENCODE_CHUNK = _args.encode_chunk
TILE_CACHE = _args.tile_cache
PRITHVI_NORM = _args.prithvi_norm
TILE_SOURCE = _args.tile_source
VALIDATE_TILE_TABLE = _args.validate_tile_table or None
# Fail-fast compatibility guards (before any heavy work):
if WARMSTART is not None and PRITHVI_NORM == 'layernorm':
    raise SystemExit(
        "[ft] --warmstart requires --prithvi_norm musd: the Phase-1 checkpoint was "
        "trained with the fixed (mu,sd) standardizer and no learned pre-norm, so a "
        "layernorm projector would not match it. Use musd to warm-start, or "
        "--warmstart none for the from-scratch layernorm path.")
if PRITHVI_NORM == 'musd' and TILE_SOURCE != 'parquet':
    raise SystemExit(
        "[ft] --prithvi_norm musd requires --tile_source parquet (the (mu,sd) "
        "standardizer is derived from embeddings.parquet).")
print(f"[ft] PHASE2 fine-tune (TFT) | proj_dim={EMB_PROJ_DIM} | lag={EMB_LAG} "
      f"| mode={EMB_MODE} | LoRA r={LORA_R} a={LORA_ALPHA} drop={LORA_DROPOUT} target={LORA_TARGET}")
print(f"[ft] warmstart={WARMSTART} | ft_lr={FT_LR} forecaster_scale={FORECASTER_LR_SCALE} "
      f"| norm={PRITHVI_NORM} | tiles={TILE_SOURCE} "
      f"| tile_grouped={TILE_GROUPED} (n_tiles={N_TILES} x spt={SAMPLES_PER_TILE})")

# Distinct output suffix so Phase-2 artifacts never collide with Phase-1.
_SUFFIX = '_ft'


def _with_suffix(path):
    root, ext = os.path.splitext(path)
    return f"{root}{_SUFFIX}{ext}"


if 'MODEL_PATH' not in os.environ:
    MODEL_PATH = _with_suffix(MODEL_PATH)
if 'SCALER_PATH' not in os.environ:
    SCALER_PATH = _with_suffix(SCALER_PATH)
if 'ENCODER_PATH' not in os.environ:
    ENCODER_PATH = _with_suffix(ENCODER_PATH)
if 'HISTORY_PATH' not in os.environ:
    HISTORY_PATH = _with_suffix(HISTORY_PATH)
if 'OUTPUT_DIR' not in os.environ:
    OUTPUT_DIR = f"{OUTPUT_DIR}{_SUFFIX}"

# Place all run artifacts (model / scaler / encoder / history / plots) under one
# directory, OUT_ROOT. Default '.' preserves the legacy cwd behaviour. Absolute
# override paths are kept as-is (os.path.join ignores OUT_ROOT for an absolute
# second arg). Set e.g. OUT_ROOT=runs/tft_prithvi to isolate each run.
OUT_ROOT = os.environ.get('OUT_ROOT', '.')
MODEL_PATH = os.path.join(OUT_ROOT, MODEL_PATH)
SCALER_PATH = os.path.join(OUT_ROOT, SCALER_PATH)
ENCODER_PATH = os.path.join(OUT_ROOT, ENCODER_PATH)
HISTORY_PATH = os.path.join(OUT_ROOT, HISTORY_PATH)
OUTPUT_DIR = os.path.join(OUT_ROOT, OUTPUT_DIR)
for _d in {os.path.dirname(MODEL_PATH), os.path.dirname(SCALER_PATH),
           os.path.dirname(ENCODER_PATH), os.path.dirname(HISTORY_PATH),
           OUTPUT_DIR}:
    os.makedirs(_d or '.', exist_ok=True)
print(f"[prithvi] outputs -> MODEL={MODEL_PATH} | HISTORY={HISTORY_PATH} "
      f"| SCALER={SCALER_PATH} | ENCODER={ENCODER_PATH} | OUTDIR={OUTPUT_DIR}")

# Self-describing per-model config for inference: window/lead drive the serve-time
# data geometry (so serving uses THIS model's lead, not a hardcoded constant); the
# rest document the build. Read back by ndvi_core.model_io.load_train_config.
import json as _json
_train_cfg = {
    "window": WINDOW, "lead": LEAD, "emb_mode": EMB_MODE,
    "emb_proj_dim": EMB_PROJ_DIM, "emb_lag": EMB_LAG, "doy": DOY,
    "lora_r": LORA_R, "lora_alpha": LORA_ALPHA, "lora_dropout": LORA_DROPOUT,
    "lora_target": LORA_TARGET, "prithvi_norm": PRITHVI_NORM, "seed": SEED,
}
with open(os.path.join(OUT_ROOT, "train_config.json"), "w") as _f:
    _json.dump(_train_cfg, _f, indent=2)
print(f"[prithvi] wrote {os.path.join(OUT_ROOT, 'train_config.json')}: {_train_cfg}")

np.random.seed(SEED)
torch.manual_seed(SEED)


# ============================================================
# LOAD DATA
# ============================================================

print("Loading dataset...")

df_complete = pd.read_csv(DATA_PATH)
# df_complete = df_complete[df_complete["mws_id"].isin(train_ids)]
print(f"Initial rows: {len(df_complete):,}")

print(df_complete[(df_complete["latitude"] == 25.50784134357382) & (df_complete["date"] == "2022-07-15")])
# ============================================================
# RENAME + CLEANING + DATETIME + SORT  (ndvi_core.features)
# ============================================================

df_complete = ncf.rename_and_clean(df_complete)   # rename ERA5 cols, -9999->NaN, drop bad NDVI/temp
df_complete = ncf.parse_and_sort(df_complete)      # to_datetime(date) + sort by (mws_id, date)


# ============================================================
# TEMPORAL SPLIT  (per-MWS chronological 70/20/10)
# ============================================================

print("Creating temporal split...")

df_complete = ncf.assign_temporal_split(df_complete, TRAIN_RATIO, VAL_RATIO)


# ============================================================
# SPLIT DATAFRAMES  (TRAIN-fit copies, made PRE-imputation like the original;
# tensor generation still uses the full chronological frame)
# ============================================================

fit_train_df = df_complete[df_complete['split'] == 'train'].copy()
fit_val_df = df_complete[df_complete['split'] == 'val'].copy()
fit_test_df = df_complete[df_complete['split'] == 'test'].copy()


# ============================================================
# LEAKAGE-SAFE IMPUTATION  (ndvi_core.features — MWS->district->state->global)
# ============================================================

print("Running leakage-safe hierarchical imputation...")
df_complete = ncf.apply_hierarchical_imputation(df_complete)
print("✅ Fast imputation complete.")


# ============================================================
# TIME FEATURES  (ndvi_core.features)
# ============================================================

print("Creating time features...")
df_complete = ncf.add_time_features(df_complete)


# ============================================================
# ENGINEERED + HARDENED FEATURES  (ndvi_core.features)
# ============================================================

print("Creating engineered features...")
_rad_clip = ncf.fit_radiation_clip(fit_train_df)   # 99th-pct clip from PRE-imputation train
df_complete = ncf.add_engineered_features(df_complete, _rad_clip)


# ============================================================
# LABEL ENCODING  (ndvi_core.features)
# ============================================================

print("Encoding categorical variables...")
df_complete, encoders = ncf.fit_label_encoders(df_complete)
joblib.dump(encoders, ENCODER_PATH)


# ============================================================
# QUARTERLY BASELINES + PHENOLOGY + DEVIATION  (ndvi_core.features, TRAIN-fit)
# ============================================================

print("Creating leakage-safe quarterly baselines + phenology + deviation...")
quarterly_train = ncf.fit_quarterly_baselines(fit_train_df)   # per-(mws,quarter) NDVI mean/std
phenology_train = ncf.fit_phenology(fit_train_df)             # historical_max_ndvi + peak_month
df_complete = ncf.merge_static_baselines(df_complete, quarterly_train, phenology_train)
df_complete = ncf.add_ndvi_q_deviation(df_complete)


# ============================================================
# FEATURE LISTS + SCALE COLS  (ndvi_core.config — single source of truth)
# ============================================================

dynamic_features = ncfg.DYNAMIC_FEATURES
forecast_feature_cols = ncfg.FORECAST_FEATURE_COLS
static_base_features = ncfg.STATIC_BASE_FEATURES
static_numerical_features = ncfg.STATIC_NUMERICAL_FEATURES
static_categorical_features = ncfg.STATIC_CATEGORICAL_FEATURES
scale_cols = ncfg.SCALE_COLS


# ============================================================
# TARGETED NULL DROP
# ============================================================

critical_cols = list(set(
    dynamic_features +
    static_base_features +
    ['date', 'mws_id', 'split', 'NDVI']
))


initial_count = len(df_complete)


df_complete = df_complete.dropna(subset=critical_cols)


print("Targeted null filtering complete")
print(f"Rows before: {initial_count:,}")
print(f"Rows after : {len(df_complete):,}")


# ============================================================
# FINAL SPLIT DATAFRAMES
# ============================================================

# Again:
# ONLY for scaler fitting / reporting.
# Tensor generation uses full chronological dataframe.


df_train = df_complete[df_complete['split'] == 'train'].copy()
df_val = df_complete[df_complete['split'] == 'val'].copy()
df_test = df_complete[df_complete['split'] == 'test'].copy()

print(df_test[
    (df_test["latitude"] == 25.50784134357382) &
    (df_test["date"] == "2022-07-15")
]["NDVI"])
# ============================================================
# DEBUG LOGGING
# ============================================================

print("\n" + "=" * 60)
print("TEMPORAL SPLIT SUMMARY")
print("=" * 60)

for name, df in [
    ('TRAIN', df_train),
    ('VAL', df_val),
    ('TEST', df_test)
]:

    print(f"\n{name}")
    print(f"Rows: {len(df):,}")
    print(f"MWS IDs: {df['mws_id'].nunique():,}")
    print(f"Date range: {df['date'].min()} -> {df['date'].max()}")


# ============================================================
# EXPORT UNSCALED STATIC LOOKUP TABLE
# ============================================================

print("\nExporting UNscaled static lookup table...")

lookup_features = [

    'mws_id',

    'latitude',
    'longitude',

    'q1_mean_ndvi',
    'q2_mean_ndvi',
    'q3_mean_ndvi',
    'q4_mean_ndvi',

    'q1_std_ndvi',
    'q2_std_ndvi',
    'q3_std_ndvi',
    'q4_std_ndvi',

    'historical_max_ndvi',
    'peak_month',

    'baseline_ndvi_mean',
    'baseline_ndvi_std',

    'sensitivity_spei3'
]

# One row per MWS
mws_lookup_unscaled = (
    df_complete[lookup_features]
    .drop_duplicates(subset=['mws_id'])
    .sort_values('mws_id')
    .reset_index(drop=True)
)

mws_lookup_unscaled.to_csv(
    "mws_static_lookup_UNSCALED.csv",
    index=False
)

print("Saved -> mws_static_lookup_UNSCALED.csv")
print(mws_lookup_unscaled.head())

# ============================================================
# SCALING
# ============================================================

print("Scaling features...")

scaler = StandardScaler()

# TRAIN ONLY FIT

scaler.fit(df_train[scale_cols])

# Apply

for split_df in [df_train, df_val, df_test]:
    split_df[scale_cols] = scaler.transform(split_df[scale_cols])


# Merge scaled back

df_complete.loc[df_train.index, scale_cols] = df_train[scale_cols]
df_complete.loc[df_val.index, scale_cols] = df_val[scale_cols]
df_complete.loc[df_test.index, scale_cols] = df_test[scale_cols]


joblib.dump(scaler, SCALER_PATH)

# ============================================================
# EXPORT SCALED STATIC LOOKUP TABLE
# ============================================================

print("\nExporting SCALED static lookup table...")

lookup_features = [

    'mws_id',

    'latitude',
    'longitude',

    'q1_mean_ndvi',
    'q2_mean_ndvi',
    'q3_mean_ndvi',
    'q4_mean_ndvi',

    'q1_std_ndvi',
    'q2_std_ndvi',
    'q3_std_ndvi',
    'q4_std_ndvi',

    'historical_max_ndvi',
    'peak_month',

    'baseline_ndvi_mean',
    'baseline_ndvi_std',

    'sensitivity_spei3'
]

# After scaling has already been applied
mws_lookup_scaled = (
    df_complete[lookup_features]
    .drop_duplicates(subset=['mws_id'])
    .sort_values('mws_id')
    .reset_index(drop=True)
)

mws_lookup_scaled.to_csv(
    "mws_static_lookup_SCALED.csv",
    index=False
)

print("Saved -> mws_static_lookup_SCALED.csv")
print(mws_lookup_scaled.head())
# ============================================================
# PRODUCTION TENSOR GENERATION
# ============================================================

print("Building production temporal tensors...")


# IMPORTANT:
# Uses FULL chronological history.
# Enforces split-safe horizons.
# Allows expanding-history validation.


# build_temporal_tensors now lives in ndvi_core.tensors (shared with inference).
build_temporal_tensors = nct.build_temporal_tensors


# ============================================================
# GENERATE TENSORS
# ============================================================

X_train_dyn,\
X_train_stat_num,\
X_train_stat_cat,\
y_train_reg,\
train_basin_ids,\
train_target_dates,\
train_debug = build_temporal_tensors(
    df_complete,
    target_split='train',
    window=WINDOW,
    lead=LEAD
)


X_val_dyn, X_val_stat_num, X_val_stat_cat, y_val_reg,val_basin_ids,val_target_dates,val_debug = (
    build_temporal_tensors(
        df_complete,
        target_split='val',
        window=WINDOW,
        lead=LEAD
    )
)
# NOTE: train debug dump intentionally disabled — the full per-sample
# training_debug.json is huge and not needed downstream.

X_test_dyn, X_test_stat_num, X_test_stat_cat, y_test_reg,test_basin_ids,test_target_dates,test_debug = (
    build_temporal_tensors(
        df_complete,
        target_split='test',
        window=WINDOW,
        lead=LEAD
    )
)


print("\nTensor Summary")
print(f"Train: {len(X_train_dyn):,}")
print(f"Val  : {len(X_val_dyn):,}")
print(f"Test : {len(X_test_dyn):,}")


# ============================================================
# PRITHVI STATIC EMBEDDINGS -> per-sample row index (Phase 1)
# ============================================================
# Built from the basin_ids + target_dates the tensor builder already returns,
# so build_temporal_tensors is untouched. Each sample maps to the (mws_id,
# year(target)-EMB_LAG) embedding row. When --use_prithvi is OFF we use a
# zero index everywhere and the model ignores it (baseline reproduction).

# Phase 2: load embeddings.parquet ONLY to derive the (mu, sd) standardizer and
# the (mws_id, year) tile table. The per-sample index `X_*_emb_idx` is now a TILE
# index (same Y-1 mapping as Phase 1); tile VALUES are re-encoded live by the
# LivePrithviProjector. The missing-row index (zero_idx) keeps Phase-1 semantics
# (missing tile -> zero block), so the sample set stays identical to base/Phase-1.
# Tile table source: the composite FOLDER (GWL-style, no parquet) or the parquet's
# key registry. Either way TILE_KEYS/key_to_row/zero_idx feed the SAME pe.idx_for
# sample->tile mapping; tile VALUES are re-encoded live.
if TILE_SOURCE == 'folder':
    TILE_KEYS, key_to_row, zero_idx = pf.build_tile_table_from_folder(COMPOSITE_DIR)
    emb_matrix = None
    EMB_DIM = None                               # taken from the Prithvi cfg below
else:
    emb_matrix, key_to_row, zero_idx, EMB_DIM = pe.load_embeddings(EMB_PATH)
    TILE_KEYS = pf.tile_keys_from_parquet(key_to_row, zero_idx)

# Walk-back year bounds (used by halfyear/quarterly; ignored for annual).
EMB_MAX_YEAR = pe.max_year_of(key_to_row)
EMB_MIN_YEAR = pe.min_year_of(key_to_row)

X_train_emb_idx, miss_tr = pe.idx_for(
    train_basin_ids, train_target_dates, key_to_row, zero_idx, lag=EMB_LAG,
    mode=EMB_MODE, max_year=EMB_MAX_YEAR, min_year=EMB_MIN_YEAR)
X_val_emb_idx, miss_va = pe.idx_for(
    val_basin_ids, val_target_dates, key_to_row, zero_idx, lag=EMB_LAG,
    mode=EMB_MODE, max_year=EMB_MAX_YEAR, min_year=EMB_MIN_YEAR)
X_test_emb_idx, miss_te = pe.idx_for(
    test_basin_ids, test_target_dates, key_to_row, zero_idx, lag=EMB_LAG,
    mode=EMB_MODE, max_year=EMB_MAX_YEAR, min_year=EMB_MIN_YEAR)

# (mu, sd) ONLY for the musd path (reproduces a frozen Phase-1). The layernorm path
# learns its own normalization, so no parquet statistics are needed.
if PRITHVI_NORM == 'musd':
    emb_mu, emb_sd = pe.standardize_(emb_matrix, X_train_emb_idx, zero_idx)
else:
    emb_mu = emb_sd = None
if emb_matrix is not None:
    del emb_matrix                               # values not needed (re-encoded live)

# Optional one-time correctness check: folder tile resolution must equal the parquet's.
if VALIDATE_TILE_TABLE:
    _pm, _pk2r, _pz, _ = pe.load_embeddings(VALIDATE_TILE_TABLE)
    del _pm
    _pkeys = pf.tile_keys_from_parquet(_pk2r, _pz)
    _pmax, _pmin = pe.max_year_of(_pk2r), pe.min_year_of(_pk2r)
    def _resolve(idxs, keys, z):
        return [None if int(i) == z else keys[int(i)] for i in idxs]
    for _nm, _bids, _dts in (('train', train_basin_ids, train_target_dates),
                             ('val', val_basin_ids, val_target_dates),
                             ('test', test_basin_ids, test_target_dates)):
        _pidx, _ = pe.idx_for(_bids, _dts, _pk2r, _pz, lag=EMB_LAG,
                              mode=EMB_MODE, max_year=_pmax, min_year=_pmin)
        _fidx, _ = pe.idx_for(_bids, _dts, key_to_row, zero_idx, lag=EMB_LAG,
                              mode=EMB_MODE, max_year=EMB_MAX_YEAR, min_year=EMB_MIN_YEAR)
        if _resolve(_fidx, TILE_KEYS, zero_idx) != _resolve(_pidx, _pkeys, _pz):
            raise SystemExit(f"[ft][validate] FOLDER vs PARQUET tile resolution "
                             f"DIFFERS on {_nm} split — aborting.")
    print("[ft][validate] folder tile resolution == parquet on train/val/test ✓")

def _pct(m, n):
    return f"{m:,} ({100 * m / max(1, n):.2f}%)"
print(f"[ft] missing-tile samples -> zero block | "
      f"train {_pct(miss_tr, len(X_train_emb_idx))} | "
      f"val {_pct(miss_va, len(X_val_emb_idx))} | "
      f"test {_pct(miss_te, len(X_test_emb_idx))}")
print(f"[ft] unique tiles used (train): {len(np.unique(X_train_emb_idx)):,}")


# ============================================================
# DATASET
# ============================================================

class NDVIRegressionDataset(Dataset):

    def __init__(
        self,
        X_dyn,
        X_stat_num,
        X_stat_cat,
        y,
        X_emb_idx,
        train_mode=False
    ):

        self.X_dyn = torch.tensor(X_dyn, dtype=torch.float32)
        self.X_stat_num = torch.tensor(X_stat_num, dtype=torch.float32)
        self.X_stat_cat = torch.tensor(X_stat_cat, dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.float32).view(-1, 1)
        self.X_emb_idx = torch.tensor(X_emb_idx, dtype=torch.long)

        self.train_mode = train_mode

    def __len__(self):
        return len(self.X_dyn)

    def __getitem__(self, idx):

        x_dyn = self.X_dyn[idx]
        x_stat_num = self.X_stat_num[idx]
        x_stat_cat = self.X_stat_cat[idx]
        y = self.y[idx]
        emb_idx = self.X_emb_idx[idx]

        # Feature masking augmentation
        if self.train_mode and np.random.random() < 0.15:
            x_dyn = x_dyn.clone()
            x_dyn[:-1, 0] = 0.0  # mask history but keep the anchor

        return x_dyn, x_stat_num, x_stat_cat, emb_idx, y
    


# ============================================================
# DATASETS
# ============================================================

train_dataset = NDVIRegressionDataset(
    X_train_dyn,
    X_train_stat_num,
    X_train_stat_cat,
    y_train_reg,
    X_train_emb_idx,
    train_mode=True
)

val_dataset = NDVIRegressionDataset(
    X_val_dyn,
    X_val_stat_num,
    X_val_stat_cat,
    y_val_reg,
    X_val_emb_idx
)


test_dataset = NDVIRegressionDataset(
    X_test_dyn,
    X_test_stat_num,
    X_test_stat_cat,
    y_test_reg,
    X_test_emb_idx
)


# ============================================================
# DATALOADERS
# ============================================================

# Tile-grouped batches keep few unique tiles per step (fast live encoding). The
# per-sample fallback (--no_tile_grouped) is exact but re-encodes tiles ~spt x
# and needs a small TRAIN_BATCH to fit encoder activations in memory.
if TILE_GROUPED:
    _train_sampler = pf.TileGroupedBatchSampler(
        X_train_emb_idx, n_tiles=N_TILES,
        samples_per_tile=SAMPLES_PER_TILE, seed=SEED)
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=_train_sampler,
        num_workers=4,
        pin_memory=True
    )
    print(f"[ft] tile-grouped train loader: {len(_train_sampler)} batches/epoch "
          f"(~{N_TILES} unique tiles/batch, batch≈{N_TILES * SAMPLES_PER_TILE})")
else:
    train_loader = DataLoader(
        train_dataset,
        batch_size=TRAIN_BATCH,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
val_loader = DataLoader(
    val_dataset,
    batch_size=VAL_BATCH,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)


test_loader = DataLoader(
    test_dataset,
    batch_size=TEST_BATCH,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)


print("\nDatasets ready")
print(f"Train samples: {len(train_dataset):,}")
print(f"Val samples  : {len(val_dataset):,}")
print(f"Test samples : {len(test_dataset):,}")


# ============================================================
# EARLY STOPPING
# ============================================================

class EarlyStopping:

    def __init__(
        self,
        patience=12,
        min_delta=0,
        path='best_model.pt'
    ):

        self.patience = patience
        self.min_delta = min_delta
        self.path = path

        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss, model):

        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)

        elif val_loss > self.best_loss - self.min_delta:

            self.counter += 1

            print(
                f'EarlyStopping counter: '
                f'{self.counter} out of {self.patience}'
            )

            if self.counter >= self.patience:
                self.early_stop = True

        else:
            self.best_loss = val_loss
            self.save_checkpoint(model)
            self.counter = 0

    def save_checkpoint(self, model):
        torch.save(model.state_dict(), self.path)


# ============================================================
# TFT MODEL  (ndvi_core.models_tft — shared with inference)
# ============================================================

GatedLinearUnit = ncm.GatedLinearUnit
GatedResidualNetwork = ncm.GatedResidualNetwork
VariableSelectionNetwork = ncm.VariableSelectionNetwork
InterpretableMultiHeadAttention = ncm.InterpretableMultiHeadAttention
ProTFT_Elite = ncm.ProTFT_Elite


# ============================================================
# DEVICE
# ============================================================


device = torch.device(
    'cuda' if torch.cuda.is_available() else 'cpu'
)

print(f"Using device: {device}")


# ============================================================
# MODEL INIT
# ============================================================

emb_configs = [
    (len(encoders['state'].classes_), ncfg.CAT_EMB_DIM),
    (len(encoders['district'].classes_), ncfg.CAT_EMB_DIM),
    (len(encoders['tehsil'].classes_), ncfg.CAT_EMB_DIM)
]


# Build the LIVE Prithvi+LoRA projector: encodes each (mws,year-1) tile on the fly
# (LoRA in graph), standardizes with the Phase-1 (mu,sd), projects 1024->proj_dim.
_lora_model, _prithvi_cfg = pf.load_prithvi_lora(
    MODEL_DIR, r=LORA_R, alpha=LORA_ALPHA, dropout=LORA_DROPOUT,
    num_frames=1, device=device, target=LORA_TARGET)

if EMB_DIM is None:                              # folder mode: dim from the encoder
    EMB_DIM = int(_prithvi_cfg['embed_dim'])

_loc_df = pd.read_csv(MWS_CSV)
_latlon = {str(r.mws_id): (float(r.lat), float(r.lon)) for r in _loc_df.itertuples()}

_tile_store = pf.TileStore(
    TILE_KEYS, COMPOSITE_DIR, _latlon,
    _prithvi_cfg['mean'], _prithvi_cfg['std'], doy=DOY,
    cache_size=TILE_CACHE, n_threads=8)

prithvi_projector = pf.LivePrithviProjector(
    _lora_model, _tile_store, emb_mu, emb_sd, zero_idx,
    in_dim=EMB_DIM, proj_dim=EMB_PROJ_DIM, pool=ncfg.PRITHVI_POOL,
    encode_chunk=ENCODE_CHUNK, amp=True, device=device, norm=PRITHVI_NORM)
print(f"[ft] live projector: {EMB_DIM} -> {EMB_PROJ_DIM} | "
      f"{len(TILE_KEYS):,} tiles | doy={DOY} | encode_chunk={ENCODE_CHUNK}")


model_reg = ProTFT_Elite(
    dynamic_input_size=len(dynamic_features),
    num_numerical_static=len(static_numerical_features),
    embedding_configs=emb_configs,
    hidden_size=ncfg.HIDDEN_SIZE,
    num_heads=ncfg.NUM_HEADS,
    d_model=ncfg.D_MODEL,
    lstm_layers=ncfg.LSTM_LAYERS,
    tft_dropout=ncfg.TFT_DROPOUT,
    prithvi_projector=prithvi_projector
).to(device)

# Warm-start: load the Phase-1 +Prithvi forecaster + projection head. The Phase-1
# checkpoint has `prithvi.proj.*` (matches) and `prithvi.emb_matrix` (ignored);
# LoRA + (mu,sd) buffers are new (kept). strict=False handles both directions.
if WARMSTART:
    if not os.path.exists(WARMSTART):
        raise SystemExit(f"[ft] warmstart checkpoint not found: {WARMSTART}")
    _sd = torch.load(WARMSTART, map_location=device, weights_only=True)
    _miss, _unexp = model_reg.load_state_dict(_sd, strict=False)
    _proj_loaded = [k for k in _sd if k.startswith('prithvi.proj.')]
    _fc_loaded = [k for k in _sd if not k.startswith('prithvi.')]
    print(f"[ft] warm-started from {WARMSTART} | forecaster keys {len(_fc_loaded)} "
          f"+ proj keys {len(_proj_loaded)} loaded | new (LoRA/buffers) kept")
else:
    print("[ft] no warm-start: forecaster + projection trained from scratch")


# ============================================================
# HISTORY TRACKING
# ============================================================

history = {
    'train_loss': [],
    'val_loss': [],
    'train_r2': [],
    'val_r2': [],
    'train_rmse': [],
    'val_rmse': [],
    'train_trend_acc': [],
    'val_trend_acc': []
}


# ============================================================
# TRAINING LOOP
# ============================================================

# ============================================================
# BASIN-WISE R²
# ============================================================

from collections import defaultdict


def compute_basin_r2(
    preds,
    trues,
    basin_ids
):

    basin_pred = defaultdict(list)
    basin_true = defaultdict(list)

    for p, t, b in zip(preds, trues, basin_ids):

        basin_pred[b].append(float(p))
        basin_true[b].append(float(t))

    basin_r2_scores = []

    for b in basin_pred.keys():

        if len(basin_true[b]) < 2:
            continue

        try:

            r2 = r2_score(
                basin_true[b],
                basin_pred[b]
            )

            if np.isfinite(r2):
                basin_r2_scores.append(r2)

        except:
            continue

    if len(basin_r2_scores) == 0:
        return np.nan, []

    return np.median(basin_r2_scores), basin_r2_scores

def run_directional_training(
    model,
    train_loader,
    val_loader,
    epochs=100,
    weight_dir=0.02
):

    model = model.to(device)

    criterion_reg = torch.nn.HuberLoss(delta=1.0)

    # Two LR groups: Prithvi (LoRA + projection) at FT_LR; warm-started forecaster
    # at a small fraction so it gently re-tunes to the shifting embeddings.
    optimizer = torch.optim.AdamW(
        pf.split_param_groups(model, FT_LR, FORECASTER_LR_SCALE),
        weight_decay=1e-5
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5
    )

    early_stopping = EarlyStopping(
        patience=12,
        path=MODEL_PATH
    )

    for epoch in range(1, epochs + 1):

        # =====================================================
        # TRAIN
        # =====================================================

        model.train()

        t_loss = 0
        t_correct = 0
        t_total = 0

        train_preds = []
        train_trues = []
        train_basin_tracker = []

        for (
            x_dyn,
            x_stat_num,
            x_stat_cat,
            emb_idx,
            y_target
        ) in tqdm(
            train_loader,
            desc=f"Epoch {epoch} [Train]",
            leave=False
        ):
            x_dyn = x_dyn.to(device)
            x_stat_num = x_stat_num.to(device)
            x_stat_cat = x_stat_cat.to(device)
            emb_idx = emb_idx.to(device)
            y_target = y_target.to(device).view(-1, 1)

            optimizer.zero_grad()

            pred, _ = model(
                x_dyn,
                x_stat_num,
                x_stat_cat,
                emb_idx
            )
            batch_size = pred.shape[0]
            train_basin_tracker.extend(
                    train_basin_ids[
                        len(train_basin_tracker):
                        len(train_basin_tracker) + batch_size
                    ]
                )

            y_current = x_dyn[:, -1, 0].view(-1, 1)

            loss_reg = criterion_reg(pred, y_target)

            pred_move = pred - y_current
            true_move = y_target - y_current

            loss_dir = F.binary_cross_entropy_with_logits(
                pred_move * 10,
                (true_move > 0).float()
            )

            total_loss = loss_reg + (weight_dir * loss_dir)

            total_loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                0.5
            )

            optimizer.step()

            t_loss += total_loss.item()

            train_preds.append(pred.detach().cpu().numpy())
            train_trues.append(y_target.cpu().numpy())

            t_correct += (
                ((pred_move > 0) == (true_move > 0))
                .float()
                .sum()
                .item()
            )

            t_total += y_target.size(0)


        # =====================================================
        # VALIDATION
        # =====================================================

        model.eval()

        v_loss = 0
        v_correct = 0
        v_total = 0

        val_preds = []
        val_trues = []
        val_basin_tracker = []
        with torch.no_grad():

            for (
                x_dyn,
                x_stat_num,
                x_stat_cat,
                emb_idx,
                y_target
            ) in tqdm(
                val_loader,
                desc=f"Epoch {epoch} [Val]",
                leave=False
            ):

                x_dyn = x_dyn.to(device)
                x_stat_num = x_stat_num.to(device)
                x_stat_cat = x_stat_cat.to(device)
                emb_idx = emb_idx.to(device)
                y_target = y_target.to(device).view(-1, 1)

                pred, _ = model(
                    x_dyn,
                    x_stat_num,
                    x_stat_cat,
                    emb_idx
                )
                batch_size = pred.shape[0]

                val_basin_tracker.extend(
                    val_basin_ids[
                        len(val_basin_tracker):
                        len(val_basin_tracker) + batch_size
                    ]
                )
                y_current = x_dyn[:, -1, 0].view(-1, 1)

                loss_reg = criterion_reg(pred, y_target)

                pred_move = pred - y_current
                true_move = y_target - y_current

                loss_dir = F.binary_cross_entropy_with_logits(
                    pred_move * 10,
                    (true_move > 0).float()
                )

                total_loss = loss_reg + (weight_dir * loss_dir)

                v_loss += total_loss.item()

                val_preds.append(pred.cpu().numpy())
                val_trues.append(y_target.cpu().numpy())

                v_correct += (
                    ((pred_move > 0) == (true_move > 0))
                    .float()
                    .sum()
                    .item()
                )

                v_total += y_target.size(0)


        # =====================================================
        # METRICS
        # =====================================================

        def get_scores(
            preds,
            trues,
            basin_ids
        ):

            p = np.concatenate(preds).flatten()
            t = np.concatenate(trues).flatten()

            mask = ~np.isnan(p) & ~np.isnan(t)

            p = p[mask]
            t = t[mask]

            basin_ids = np.array(basin_ids)[mask]

            global_r2 = r2_score(t, p)

            rmse = np.sqrt(
                np.mean((t - p) ** 2)
            )

            median_basin_r2, basin_r2s = (
                compute_basin_r2(
                    p,
                    t,
                    basin_ids
                )
            )

            return (
                global_r2,
                rmse,
                median_basin_r2
            )



        train_r2,train_rmse,train_median_r2 = get_scores(
        train_preds,
        train_trues,
        train_basin_tracker)

    
        val_r2,val_rmse,val_median_r2= get_scores(
        val_preds,
        val_trues,
        val_basin_tracker)



        train_trend = t_correct / t_total
        val_trend = v_correct / v_total


        history['train_loss'].append(
            t_loss / len(train_loader)
        )

        history['val_loss'].append(
            v_loss / len(val_loader)
        )

        history['train_r2'].append(train_r2)
        history['val_r2'].append(val_r2)

        history['train_rmse'].append(train_rmse)
        history['val_rmse'].append(val_rmse)

        history['train_trend_acc'].append(train_trend)
        history['val_trend_acc'].append(val_trend)


        print(
            f"Epoch {epoch:02d} | "
            f"Train Global R²: {train_r2:.4f} | "
            f"Train Median Basin R²: {train_median_r2:.4f} | "
            f"Val Global R²: {val_r2:.4f} | "
            f"Val Median Basin R²: {val_median_r2:.4f} | "
            f"Val RMSE: {val_rmse:.4f} | "
            f"Trend: {100 * val_trend:.2f}%"
        )


        scheduler.step(v_loss / len(val_loader))

        early_stopping(
            v_loss / len(val_loader),
            model
        )

        if early_stopping.early_stop:
            print("Early stopping triggered")
            break


    model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
    print("Best model loaded")


# ============================================================
# TRAIN
# ============================================================

if EVAL_ONLY:
    # Backfill mode: load the already-trained FT checkpoint, skip training.
    model_reg.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
    print(f"[eval_only] loaded checkpoint {MODEL_PATH}; skipping training + history save")
else:
    run_directional_training(
        model_reg,
        train_loader,
        val_loader,
        epochs=EPOCHS,
        weight_dir=0.02
    )


    # ============================================================
    # SAVE HISTORY
    # ============================================================

    history_df = pd.DataFrame({
        'epoch': range(1, len(history['train_loss']) + 1),
        'train_loss': history['train_loss'],
        'val_loss': history['val_loss'],
        'train_r2': history['train_r2'],
        'val_r2': history['val_r2'],
        'train_rmse': history['train_rmse'],
        'val_rmse': history['val_rmse'],
        'train_trend_acc': history['train_trend_acc'],
        'val_trend_acc': history['val_trend_acc']
    })


    history_df.to_csv(HISTORY_PATH, index=False)

    print(f"History saved -> {HISTORY_PATH}")


# ============================================================
# TEST EVALUATION
# ============================================================


def evaluate_on_test(model, test_loader):

    model.eval()

    preds = []
    trues = []
    currents = []

    correct = 0
    total = 0
    test_basin_tracker = []
    with torch.no_grad():

        for (
            x_dyn,
            x_stat_num,
            x_stat_cat,
            emb_idx,
            y_target
        ) in tqdm(test_loader, desc="Testing"):

            x_dyn = x_dyn.to(device)
            x_stat_num = x_stat_num.to(device)
            x_stat_cat = x_stat_cat.to(device)
            emb_idx = emb_idx.to(device)
            y_target = y_target.to(device).view(-1, 1)

            pred, _ = model(
                x_dyn,
                x_stat_num,
                x_stat_cat,
                emb_idx
            )
            batch_size = pred.shape[0]

            test_basin_tracker.extend(
                test_basin_ids[
                    len(test_basin_tracker):
                    len(test_basin_tracker) + batch_size
                ]
            )
            y_current = x_dyn[:, -1, 0].view(-1, 1)

            preds.append(pred.cpu().numpy())
            trues.append(y_target.cpu().numpy())
            currents.append(y_current.cpu().numpy())

            correct += (
                (((pred - y_current) > 0) == ((y_target - y_current) > 0))
                .float()
                .sum()
                .item()
            )

            total += y_target.size(0)


    p = np.concatenate(preds).flatten()
    t = np.concatenate(trues).flatten()
    c = np.concatenate(currents).flatten()   # last observed NDVI = persistence prediction

    global_r2 = r2_score(t, p)

    median_test_r2, basin_r2s = (
        compute_basin_r2(
            p,
            t,
            test_basin_tracker
        )
    )
    rmse = np.sqrt(np.mean((t - p) ** 2))
    trend_acc = 100 * (correct / total)

    # Persistence baseline (predict next = last observed NDVI).
    pers_global_r2 = r2_score(t, c)
    pers_median_r2, _ = compute_basin_r2(c, t, test_basin_tracker)
    pers_rmse = np.sqrt(np.mean((t - c) ** 2))

    # Delta-only R² (skill on the residual actual-current) + median-basin delta.
    delta_true = t - c
    delta_pred = p - c
    model_delta_r2 = r2_score(delta_true, delta_pred)
    median_delta_r2, _ = compute_basin_r2(delta_pred, delta_true, test_basin_tracker)

    # Deseasonalized ANOMALY R²: NDVI minus per-basin-per-month climatology (fit on
    # TRAIN targets only, leak-free, scaled space). Denominator = anomaly variance
    # only -> the true drought-skill metric. Mirrors the _PRITHVI runner.
    try:
        _trm = pd.to_datetime(pd.Series(list(train_target_dates))).dt.month.to_numpy()
        _trdf = pd.DataFrame({'b': [str(x) for x in train_basin_ids], 'm': _trm,
                              'y': np.asarray(y_train_reg, dtype=float).ravel()})
        _clim = _trdf.groupby(['b', 'm'])['y'].mean().to_dict()
        _bmean = _trdf.groupby('b')['y'].mean().to_dict()
        _gmean = float(_trdf['y'].mean())
        _tem = pd.to_datetime(pd.Series(list(test_target_dates[:len(t)]))).dt.month.to_numpy()
        _climt = np.array([_clim.get((str(b), int(m)), _bmean.get(str(b), _gmean))
                           for b, m in zip(test_basin_tracker, _tem)])
        anom_true = t - _climt
        anom_pred = p - _climt
        global_anom_r2 = r2_score(anom_true, anom_pred)
        median_anom_r2, _ = compute_basin_r2(anom_pred, anom_true, test_basin_tracker)
    except Exception as _e:
        global_anom_r2 = float('nan'); median_anom_r2 = float('nan')
        print(f"[anomaly] skipped: {str(_e)[:140]}")

    print("\n" + "=" * 50)
    print("FINAL TEST RESULTS")
    print("=" * 50)
    print(f"Global Test R²        : {global_r2:.4f}")
    print(f"Median Basin Test R² : {median_test_r2:.4f}")
    print(f"Test RMSE             : {rmse:.4f}")
    print(f"Trend Accuracy        : {trend_acc:.2f}%")
    print("-" * 50)
    print("PERSISTENCE BASELINE (predict next = last observed NDVI)")
    print(f"Persistence Global R² : {pers_global_r2:.4f}")
    print(f"Persistence Median R² : {pers_median_r2:.4f}")
    print(f"Persistence RMSE      : {pers_rmse:.4f}")
    print("-" * 50)
    print(f"Model Delta-only R²   : {model_delta_r2:.4f}  "
          f"(skill on residual = actual - current; persistence = 0 by definition)")
    print(f"Median Basin Delta R²: {median_delta_r2:.4f}  "
          f"(per-basin R² on actual-current vs pred-current, then median)")
    print(f"Global Anomaly R²     : {global_anom_r2:.4f}  "
          f"(NDVI - per-basin-per-month climatology; true drought-skill metric)")
    print(f"Median Basin Anomaly R²: {median_anom_r2:.4f}  "
          f"(per-basin R² on anomalies, then median = north-star drought skill)")
    print("=" * 50)


# ============================================================
# RUN TEST
# ============================================================

evaluate_on_test(model_reg, test_loader)

if EVAL_ONLY:
    print("[eval_only] metrics printed; skipping sample export + per-MWS eval. Done.")
    raise SystemExit(0)

# ============================================================
# EXPORT REAL TEST SAMPLES FOR INFERENCE VALIDATION
# ============================================================

print("\nExporting real test samples...")

model_reg.eval()

export_rows = []

sample_counter = 0

with torch.no_grad():

    for (
        x_dyn,
        x_stat_num,
        x_stat_cat,
        emb_idx,
        y_target
    ) in test_loader:

        x_dyn_gpu = x_dyn.to(device)
        x_stat_num_gpu = x_stat_num.to(device)
        x_stat_cat_gpu = x_stat_cat.to(device)
        emb_idx_gpu = emb_idx.to(device)

        preds, _ = model_reg(
            x_dyn_gpu,
            x_stat_num_gpu,
            x_stat_cat_gpu,
            emb_idx_gpu
        )

        preds = preds.cpu().numpy().flatten()
        actuals = y_target.numpy().flatten()

        batch_size = len(preds)

        for i in range(batch_size):

            global_idx = sample_counter + i

            export_rows.append({

                "mws_id": str(test_basin_ids[global_idx]),

            "date": str(test_target_dates[global_idx]),

                "latitude": float(
                    df_test.iloc[global_idx]['latitude']
                ),

                "longitude": float(
                    df_test.iloc[global_idx]['longitude']
                ),

                "actual_ndvi": float(actuals[i]),
                "predicted_ndvi": float(preds[i])
            })

        sample_counter += batch_size

        # export first 200 samples
        if sample_counter >= 200:
            break


# ============================================================
# SAVE
# ============================================================

export_df = pd.DataFrame(export_rows)

export_df.to_csv(
    "test_inference_samples.csv",
    index=False
)

print("\nSaved -> test_inference_samples.csv")

print("\nSample Preview:")
print(export_df.head())

# ============================================================
# BEST / WORST MWS PLOTS
# ============================================================



def plot_best_worst_mws(
    model,
    full_df,
    filename='best_worst_temporal.png'
):

    model.eval()

    results = []

    unique_ids = full_df['mws_id'].unique()

    for mid in tqdm(unique_ids, desc='Per-MWS Evaluation'):

        group = (
            full_df[
                (full_df['mws_id'] == mid) &
                (full_df['split'] == 'test')
            ]
            .sort_values('date')
        )

        if len(group) < 20:
            continue

        try:

            preds = []
            trues = []
            dates = []

            group_all = (
                full_df[
                    full_df['mws_id'] == mid
                ]
                .sort_values('date')
                .reset_index(drop=True)
            )

            dyn_vals = (
                group_all[dynamic_features]
                .values
                .astype(np.float32)
            )

            base_static_vals = (
                group_all[static_base_features]
                .iloc[0]
                .values
                .astype(np.float32)
            )

            stat_cat_vals = (
                group_all[static_categorical_features]
                .iloc[0]
                .values
                .astype(np.int64)
            )

            for t in range(WINDOW + LEAD - 1, len(group_all)):

                row = group_all.iloc[t]

                if row['split'] != 'test':
                    continue

                history_start = t - LEAD - WINDOW + 1
                history_end = t - LEAD + 1

                forecast_start = t - LEAD + 1
                forecast_end = t + 1

                history = group_all.iloc[history_start:history_end]
                forecast = group_all.iloc[forecast_start:forecast_end]

                if len(history) != WINDOW:
                    continue

                if len(forecast) != LEAD:
                    continue

                if len(forecast['split'].unique()) != 1:
                    continue

                if forecast['split'].iloc[0] != 'test':
                    continue

                forecast_vector = np.array([
                    forecast['precipitation'].sum(),
                    forecast['temperature_2m'].mean(),
                    # forecast['runoff'].sum(),
                    # forecast['et'].mean()
                ], dtype=np.float32)

                static_vector = np.concatenate([
                    base_static_vals,
                    forecast_vector
                ]).astype(np.float32)

                x_dyn = torch.tensor(
                    history[dynamic_features].values,
                    dtype=torch.float32
                ).unsqueeze(0).to(device)

                x_stat_num = torch.tensor(
                    static_vector,
                    dtype=torch.float32
                ).unsqueeze(0).to(device)

                x_stat_cat = torch.tensor(
                    stat_cat_vals,
                    dtype=torch.long
                ).unsqueeze(0).to(device)

                emb_idx_t = None
                if USE_PRITHVI:
                    _yr = pd.Timestamp(row['date']).year - EMB_LAG
                    _ri = key_to_row.get((str(mid), int(_yr)), zero_idx)
                    emb_idx_t = torch.tensor([_ri], dtype=torch.long).to(device)

                with torch.no_grad():
                    pred, _ = model(
                        x_dyn,
                        x_stat_num,
                        x_stat_cat,
                        emb_idx_t
                    )

                preds.append(pred.cpu().numpy().flatten()[0])
                trues.append(row['NDVI'])
                dates.append(row['date'])


            if len(trues) > 10:

                score = r2_score(trues, preds)

                results.append({
                    'id': mid,
                    'r2': score,
                    'preds': preds,
                    'trues': trues,
                    'dates': dates
                })

        except Exception as e:
            print(f"MWS {mid} failed: {e}")
            continue


    if len(results) < 10:
        print("Not enough valid MWS IDs")
        return


    results = sorted(
        results,
        key=lambda x: x['r2'],
        reverse=True
    )


    best_5 = results[:5]
    worst_5 = results[-5:]

    plot_list = []

    for b, w in zip(best_5, worst_5):
        plot_list.append(b)
        plot_list.append(w)


    fig, axes = plt.subplots(
        nrows=5,
        ncols=2,
        figsize=(22, 28)
    )

    axes = axes.flatten()

    for i, data in enumerate(plot_list):

        ax = axes[i]

        is_best = (i % 2 == 0)

        color = 'blue' if is_best else 'red'
        status = 'BEST' if is_best else 'WORST'

        ax.plot(
            data['dates'],
            data['trues'],
            label='Actual',
            color='black',
            alpha=0.4
        )

        ax.plot(
            data['dates'],
            data['preds'],
            label='Predicted',
            color=color,
            linestyle='--'
        )

        ax.set_title(
            f"{status} | {data['id']} | R²={data['r2']:.4f}"
        )

        ax.legend()
        ax.grid(True, alpha=0.2)


    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), bbox_inches='tight')
    plt.close()

    print(f"Saved plot -> {filename}")


# ============================================================
# GENERATE PLOTS
# ============================================================

plot_best_worst_mws(
    model_reg,
    df_complete
)


# ============================================================
# DIAGNOSTIC TIME-SERIES PLOTS
#   -> diag_test_best_worst_timeseries.png
#   -> diag_worst_full_timeline.png
# Adapted from gwl_plots.py (_plot_best_worst_timeseries +
# generate_worst_full_timeline_plot) for the NDVI / MWS pipeline.
# Plots are in scaled-NDVI space (same as the rest of the pipeline).
# ============================================================

import matplotlib.dates as mdates
from collections import defaultdict

MIN_TS_SAMPLES = 10      # min test samples for an MWS to be ranked
N_PER_CATEGORY = 5       # number of best / worst panels

# Cache of worst test-MWS ids, populated by the best/worst plot and
# reused by the full-timeline plot so both show the SAME stations.
_WORST_TEST_MWS = []

_EMPTY_DATES = np.array([], dtype='datetime64[ns]')
_EMPTY_VALS = np.array([], dtype=np.float32)


def _predict_in_batches(model, X_dyn, X_num, X_cat, X_emb_idx=None, batch=4096):
    """Run the model over flat tensors -> absolute-NDVI preds (scaled space)."""
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(X_dyn), batch):
            xd = torch.tensor(X_dyn[i:i + batch], dtype=torch.float32).to(device)
            xn = torch.tensor(X_num[i:i + batch], dtype=torch.float32).to(device)
            xc = torch.tensor(X_cat[i:i + batch], dtype=torch.long).to(device)
            ei = None
            if X_emb_idx is not None:
                ei = torch.tensor(X_emb_idx[i:i + batch], dtype=torch.long).to(device)
            p, _ = model(xd, xn, xc, ei)
            preds.append(p.cpu().numpy().flatten())
    return np.concatenate(preds) if preds else _EMPTY_VALS


def _group_pred_by_mws(preds, trues, basin_ids, dates):
    """{mws_id: (dates_sorted, preds_sorted, trues_sorted)}."""
    bucket = defaultdict(lambda: {'p': [], 't': [], 'd': []})
    for p, t, b, d in zip(preds, trues, basin_ids, dates):
        bucket[b]['p'].append(float(p))
        bucket[b]['t'].append(float(t))
        bucket[b]['d'].append(pd.to_datetime(d))
    out = {}
    for b, v in bucket.items():
        order = np.argsort(v['d'])
        out[b] = (
            np.array(v['d'])[order],
            np.array(v['p'])[order],
            np.array(v['t'])[order],
        )
    return out


def _group_actual_by_mws(trues, basin_ids, dates):
    """{mws_id: (dates_sorted, trues_sorted)} — actual NDVI, no inference."""
    bucket = defaultdict(lambda: {'t': [], 'd': []})
    for t, b, d in zip(trues, basin_ids, dates):
        bucket[b]['t'].append(float(t))
        bucket[b]['d'].append(pd.to_datetime(d))
    out = {}
    for b, v in bucket.items():
        order = np.argsort(v['d'])
        out[b] = (np.array(v['d'])[order], np.array(v['t'])[order])
    return out


def _mask_predict_group(model, X_dyn, X_num, X_cat, trues, basin_ids, dates, keep,
                        X_emb_idx=None):
    """Predict + group, only for MWS in `keep` (efficient subset inference)."""
    keep = set(keep)
    basin_ids = np.asarray(basin_ids)
    mask = np.array([b in keep for b in basin_ids])
    if not mask.any():
        return {}
    ei = X_emb_idx[mask] if X_emb_idx is not None else None
    preds = _predict_in_batches(model, X_dyn[mask], X_num[mask], X_cat[mask], ei)
    return _group_pred_by_mws(
        preds,
        np.asarray(trues)[mask],
        basin_ids[mask],
        np.asarray(dates)[mask],
    )


def plot_best_worst_timeseries(
    model,
    n_per_category=N_PER_CATEGORY,
    filename='diag_test_best_worst_timeseries.png',
):
    """Best vs worst MWS (by test R²) as side-by-side time series."""
    global _WORST_TEST_MWS

    test_groups = _group_pred_by_mws(
        _predict_in_batches(model, X_test_dyn, X_test_stat_num, X_test_stat_cat,
                            X_test_emb_idx),
        y_test_reg, test_basin_ids, test_target_dates,
    )

    # Actual history (no inference needed — we already store the targets)
    train_actual = _group_actual_by_mws(y_train_reg, train_basin_ids, train_target_dates)
    val_actual = _group_actual_by_mws(y_val_reg, val_basin_ids, val_target_dates)

    mws_r2 = {}
    for b, (d, p, t) in test_groups.items():
        if len(t) < MIN_TS_SAMPLES or np.std(t) == 0:
            continue
        mws_r2[b] = r2_score(t, p)

    if len(mws_r2) < 2 * n_per_category:
        print(f"  Too few MWS for best/worst timeseries ({len(mws_r2)})")
        return

    ranked = sorted(mws_r2, key=lambda b: mws_r2[b])
    worst = ranked[:n_per_category]
    best = ranked[-n_per_category:][::-1]
    _WORST_TEST_MWS = list(worst)

    fig, axes = plt.subplots(
        n_per_category, 2,
        figsize=(16, 3.4 * n_per_category),
        squeeze=False,
    )

    for col, (label, mws_list, color) in enumerate([
        ("BEST", best, "#0D88ED"),
        ("WORST", worst, "#FF5722"),
    ]):
        for row, b in enumerate(mws_list):
            ax = axes[row, col]
            td, tp, tt = test_groups[b]

            ta = train_actual.get(b, (_EMPTY_DATES, _EMPTY_VALS))
            va = val_actual.get(b, (_EMPTY_DATES, _EMPTY_VALS))

            all_d = np.concatenate([ta[0], va[0], td])
            all_t = np.concatenate([ta[1], va[1], tt])
            if len(all_d):
                order = np.argsort(all_d)
                ax.plot(all_d[order], all_t[order], "b-", linewidth=1.0,
                        alpha=0.7, label="Actual")

            ax.plot(td, tp, "r--", linewidth=1.3, alpha=0.9, label="Predicted")

            # Background shading for the train / val regions
            test_start = td.min() if len(td) else None
            val_start = va[0].min() if len(va[0]) else None
            train_start = ta[0].min() if len(ta[0]) else None
            if train_start is not None and test_start is not None:
                end = val_start if val_start is not None else test_start
                ax.axvspan(train_start, end, color="0.92", zorder=0)
            if val_start is not None and test_start is not None:
                ax.axvspan(val_start, test_start, color="0.85", zorder=0)

            ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
            ax.xaxis.set_major_locator(mdates.AutoDateLocator())
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right",
                     fontsize=7)
            ax.set_title(
                f"{label} | {b}\nTest R²={mws_r2[b]:.3f}  (n={len(tt)})",
                fontsize=9, fontweight="bold", color=color,
            )
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)
            if col == 0:
                ax.set_ylabel("NDVI (scaled)")

    plt.suptitle("Best vs Worst MWS (by Test R²) — NDVI", fontsize=13, y=1.005)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved plot -> {filename}")


def plot_worst_full_timeline(
    model,
    n_per_split=N_PER_CATEGORY,
    filename='diag_worst_full_timeline.png',
):
    """Worst test MWS plotted across their val+test timeline (no train)."""
    worst = _WORST_TEST_MWS[:n_per_split]
    if not worst:
        print("  No cached worst MWS; run plot_best_worst_timeseries first.")
        return

    # Predict only for the worst MWS — val and test splits only (no train)
    val_p = _mask_predict_group(
        model, X_val_dyn, X_val_stat_num, X_val_stat_cat,
        y_val_reg, val_basin_ids, val_target_dates, worst,
        X_emb_idx=X_val_emb_idx)
    test_p = _mask_predict_group(
        model, X_test_dyn, X_test_stat_num, X_test_stat_cat,
        y_test_reg, test_basin_ids, test_target_dates, worst,
        X_emb_idx=X_test_emb_idx)

    val_actual = _group_actual_by_mws(y_val_reg, val_basin_ids, val_target_dates)
    test_actual = _group_actual_by_mws(y_test_reg, test_basin_ids, test_target_dates)

    n = len(worst)
    fig, axes = plt.subplots(n, 1, figsize=(16, 3.0 * n), squeeze=False)

    for i, b in enumerate(worst):
        ax = axes[i, 0]

        va = val_actual.get(b, (_EMPTY_DATES, _EMPTY_VALS))
        te = test_actual.get(b, (_EMPTY_DATES, _EMPTY_VALS))

        hx = np.concatenate([va[0], te[0]])
        hy = np.concatenate([va[1], te[1]])
        if len(hx):
            order = np.argsort(hx)
            hx = hx[order]
            hy = hy[order]
            ax.plot(hx, hy, color="#0D88ED", linewidth=1.2, alpha=0.85,
                    label="Actual")

        for split_label, groups, color in [
            ("val", val_p, "#FF9800"),
            ("test", test_p, "#D32F2F"),
        ]:
            if b in groups:
                d, p, _t = groups[b]
                ax.plot(d, p, "o--", color=color, markersize=3, linewidth=0.9,
                        alpha=0.75, label=f"Pred ({split_label}, n={len(d)})")

        # Region shading: val region (lighter) then test region (darker)
        val_end = va[0].max() if len(va[0]) else None
        if len(hx) and val_end is not None:
            lo, hi = hx.min(), hx.max()
            ax.axvspan(lo, val_end, color="gray", alpha=0.10, zorder=0)
            if hi > val_end:
                ax.axvspan(val_end, hi, color="gray", alpha=0.24, zorder=0)

        tr2 = (
            r2_score(test_p[b][2], test_p[b][1])
            if b in test_p and len(test_p[b][2]) > 1 else None
        )
        tr2_s = f"test R²={tr2:.3f}" if tr2 is not None else "test R²=—"
        ax.set_title(f"{b}  |  {tr2_s}", fontsize=10, color="#D32F2F")
        ax.set_ylabel("NDVI (scaled)")
        ax.grid(alpha=0.3)
        ax.legend(loc="best", fontsize=7, ncol=2)
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        for tick in ax.get_xticklabels():
            tick.set_rotation(45)

    plt.suptitle(
        f"Worst MWS Timeline (val + test, by Test R²) — top {n}",
        fontsize=13, y=1.005,
    )
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved plot -> {filename} ({n} stations)")


plot_best_worst_timeseries(model_reg)
plot_worst_full_timeline(model_reg)


print("\nPipeline complete.")
