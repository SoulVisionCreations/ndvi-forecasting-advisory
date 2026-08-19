"""
ndvi_core.config — the feature contract (single source of truth).

These lists/constants were previously duplicated verbatim across
Run_With_MWS_Split_Temporal_FT.py, ..._TFT_FT.py, app.py and unified_engine.py.
They are now defined ONCE here; every train/inference path imports from this
module so the model's input contract can never silently drift.

Verified identical in _FT.py and _TFT_FT.py on 2026-06-29.
"""

# ---------------------------------------------------------------------------
# Window geometry (champion = window 24 / lead 7). Drivers may still override
# via env (WINDOW/LEAD); these are the canonical defaults + documentation.
# ---------------------------------------------------------------------------
WINDOW = 24            # temporal lookback steps (fortnights); champion uses 24 (= 12 months)
LEAD = 7               # forecast lead in steps (3.5 months)
FORECAST_HORIZON = 7   # forecast window length used for the forecast_* aggregates

# Serve-time anchor grid (C1). The as-of / forecast-from date is snapped DOWN to
# the most recent 14-day fortnight on a grid anchored at FORTNIGHT_GRID_EPOCH ->
# that snapped date is the anchor s0; target = anchor + LEAD*14 days (mirrors
# training's grid geometry target_row = anchor_row + lead). Training resamples each
# series to its own 14-day phase (many phases), so the model is phase-robust and
# any fixed grid is faithful; the epoch below is a real training-grid start. A
# missing/latency reading AT the anchor is ffilled in build_window (previous value),
# never by sliding the anchor back. See ndvi_core.dates.resolve_forecast_dates.
FORTNIGHT_GRID_EPOCH = "2017-07-01"

# ---------------------------------------------------------------------------
# Raw GEE/source column harmonisation
# ---------------------------------------------------------------------------
# ERA5/MODIS raw names -> the names the rest of the pipeline expects.
RENAME_MAP = {
    "total_precipitation_sum": "precipitation",
    "runoff_sum": "runoff",
    "total_evaporation_sum": "et",
}

# Columns cleaned of poison values (-9999) and used for hierarchical imputation.
WEATHER_COLS = [
    "NDVI", "et", "runoff", "precipitation", "temperature_2m",
    "dewpoint_temperature_2m",
    "surface_solar_radiation_downwards_sum",
    "surface_net_thermal_radiation_sum",
    "sm_surface", "sm_rootzone",
]

# ---------------------------------------------------------------------------
# Model input contract
# ---------------------------------------------------------------------------
# Dynamic (per-timestep) sequence features — order matters: NDVI MUST be index 0
# (the model reads current_ndvi = x_dyn[:, -1, 0] for the residual/delta head).
DYNAMIC_FEATURES = [
    "NDVI",
    "ndvi_q_deviation",
    "month_sin",
    "month_cos",
    "et",
    "runoff",
    "precipitation",
    "dewpoint_temperature_2m",
    "sm_rootzone",
    "sm_surface",
    "surface_net_thermal_radiation_sum",
    "surface_solar_radiation_downwards_sum",
    "temperature_2m",
    "u_component_of_wind_10m",
    "v_component_of_wind_10m",
    "precip_et_ratio",
    "months_from_peak",
    "temp_precip_idx",
    "evap_stress",
    "precip_cum_3",
]

# Forecast-window aggregates (computed over the `lead` forecast steps), appended
# to the static numeric vector.
FORECAST_FEATURE_COLS = [
    "forecast_precip_sum",
    "forecast_temp_mean",
]

# Static numeric base features (one row per MWS).
STATIC_BASE_FEATURES = [
    "latitude",
    "longitude",
    "baseline_ndvi_mean",
    "baseline_ndvi_std",
    "sensitivity_spei3",
    "q1_mean_ndvi",
    "q2_mean_ndvi",
    "q3_mean_ndvi",
    "q4_mean_ndvi",
    "q1_std_ndvi",
    "q2_std_ndvi",
    "q3_std_ndvi",
    "q4_std_ndvi",
    "historical_max_ndvi",
    "peak_month",
]

# Full static numeric vector fed to the model = base + forecast aggregates.
STATIC_NUMERICAL_FEATURES = STATIC_BASE_FEATURES + FORECAST_FEATURE_COLS

# Static categorical features (label-encoded -> embeddings).
STATIC_CATEGORICAL_FEATURES = [
    "state_encoded",
    "district_encoded",
    "tehsil_encoded",
]

# Raw categorical source columns the LabelEncoders are fit on (in order).
CATEGORICAL_SOURCE_COLS = ["state", "district", "tehsil", "mws_id"]

# ---------------------------------------------------------------------------
# Scaling — columns standardised by the saved StandardScaler (fit on train).
# ---------------------------------------------------------------------------
SCALE_COLS = [
    "NDVI",
    "et",
    "runoff",
    "precipitation",
    "dewpoint_temperature_2m",
    "sm_rootzone",
    "sm_surface",
    "surface_net_thermal_radiation_sum",
    "surface_solar_radiation_downwards_sum",
    "temperature_2m",
    "u_component_of_wind_10m",
    "v_component_of_wind_10m",
    "precip_et_ratio",
    "latitude",
    "longitude",
    "baseline_ndvi_mean",
    "baseline_ndvi_std",
    "q1_mean_ndvi",
    "q2_mean_ndvi",
    "q3_mean_ndvi",
    "q4_mean_ndvi",
    "q1_std_ndvi",
    "q2_std_ndvi",
    "q3_std_ndvi",
    "q4_std_ndvi",
    "historical_max_ndvi",
    "peak_month",
    "months_from_peak",
    "temp_precip_idx",
    "evap_stress",
    "precip_cum_3",
    "sensitivity_spei3",
]

# Columns written to the per-MWS static lookup table (mws_static_lookup_*.csv),
# consumed at inference to fetch a location's static profile.
LOOKUP_FEATURES = [
    "mws_id",
    "latitude",
    "longitude",
    "q1_mean_ndvi",
    "q2_mean_ndvi",
    "q3_mean_ndvi",
    "q4_mean_ndvi",
    "q1_std_ndvi",
    "q2_std_ndvi",
    "q3_std_ndvi",
    "q4_std_ndvi",
    "historical_max_ndvi",
    "peak_month",
    "baseline_ndvi_mean",
    "baseline_ndvi_std",
    "sensitivity_spei3",
]

# NDVI is dynamic-feature index 0 (residual head anchor) and scale-col index 0.
NDVI_DYNAMIC_IDX = DYNAMIC_FEATURES.index("NDVI")
NDVI_SCALE_IDX = SCALE_COLS.index("NDVI")

# ---------------------------------------------------------------------------
# Model architecture (ProTFT_Elite) + Prithvi LoRA/projector — the EXACT values
# the champion bundle was trained with. Single source for BOTH the training
# MODEL INIT and the inference model_io, so the served graph can never drift
# from the trained one. (Mismatches also self-fail at load_state_dict.)
# ---------------------------------------------------------------------------
HIDDEN_SIZE = 256          # ProTFT_Elite arg (LSTM-parity; not used by TFT internals)
NUM_HEADS = 4
D_MODEL = 128
LSTM_LAYERS = 1
TFT_DROPOUT = 0.1
CAT_EMB_DIM = 2            # embedding dim per categorical (state/district/tehsil)

EMB_PROJ_DIM = 32          # Prithvi projector output (the 32-d static block)
PRITHVI_NORM = "layernorm" # GWL-style learned LayerNorm pre-projection (champion)
PRITHVI_POOL = "mean"
PRITHVI_DOY = 182          # ANNUAL fallback doy (quarterly tiles use per-period doy)

LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET = "qkv"
