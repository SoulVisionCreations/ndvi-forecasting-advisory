"""
ndvi_core.tensors — tensor/window construction.

build_temporal_tensors : the training tensor builder, extracted verbatim
  (behavior-preserving) from Run_With_MWS_Split_Temporal_TFT_FT.py. Walks each
  MWS chronologically and emits split-safe (window, lead) samples.

single_window_tensors : the inference counterpart — assemble ONE
  (x_dyn, x_stat_num, x_stat_cat) sample from a processed single-location
  history + a static-numeric vector + categorical ids, using the SAME feature
  ordering (config.DYNAMIC_FEATURES) so it matches what training produced.
"""

import numpy as np
from tqdm import tqdm

from . import config


def build_temporal_tensors(
    full_df,
    target_split,
    window=config.WINDOW,
    lead=config.LEAD,
    collect_debug=True,
):
    """Per-MWS split-safe sample builder. Returns
    (X_dyn, X_stat_num, X_stat_cat, y, basin_ids, target_dates, sample_debug).
    Verbatim from the training script (dynamic/static feature lists now sourced
    from config)."""
    dynamic_features = config.DYNAMIC_FEATURES
    static_base_features = config.STATIC_BASE_FEATURES
    static_categorical_features = config.STATIC_CATEGORICAL_FEATURES

    X_dyn, X_stat_num, X_stat_cat, y = [], [], [], []
    target_dates, basin_ids, sample_debug = [], [], []

    grouped = full_df.groupby("mws_id")

    for mws_id, group in tqdm(grouped, desc=f"Building {target_split}", disable=False):
        group = group.sort_values("date").reset_index(drop=True)

        if len(group) < (window + lead):
            continue

        base_static_vals = group[static_base_features].iloc[0].values.astype(np.float32)
        stat_cat_vals = group[static_categorical_features].iloc[0].values.astype(np.int64)

        for t in range(window + lead - 1, len(group)):
            target_row = group.iloc[t]
            if target_row["split"] != target_split:
                continue

            history_start = t - lead - window + 1
            history_end = t - lead + 1
            if history_start < 0:
                continue
            history = group.iloc[history_start:history_end]
            if len(history) != window:
                continue

            forecast_start = t - lead + 1
            forecast_end = t + 1
            forecast = group.iloc[forecast_start:forecast_end]
            if len(forecast) != lead:
                continue

            forecast_splits = forecast["split"].unique()
            if len(forecast_splits) != 1 or forecast_splits[0] != target_split:
                continue

            forecast_precip_sum = forecast["precipitation"].sum()
            forecast_temp_mean = forecast["temperature_2m"].mean()
            forecast_vector = np.array(
                [forecast_precip_sum, forecast_temp_mean], dtype=np.float32
            )

            history_values = history[dynamic_features].values.astype(np.float32)
            target_value = np.float32(target_row["NDVI"])
            full_static = np.concatenate(
                [base_static_vals, forecast_vector]
            ).astype(np.float32)

            if np.isnan(history_values).any() or np.isnan(full_static).any() or np.isnan(target_value):
                continue

            if collect_debug:
                sample_debug.append({
                    "mws_id": str(mws_id),
                    "target_date": str(target_row["date"]),
                    "latitude": float(group.iloc[0]["latitude"]),
                    "longitude": float(group.iloc[0]["longitude"]),
                    "actual_ndvi": float(target_value),
                    "forecast_precip_sum": float(forecast_precip_sum),
                    "forecast_temp_mean": float(forecast_temp_mean),
                    "history_dates": history["date"].astype(str).tolist(),
                    "history_ndvi": history["NDVI"].tolist(),
                    "history_precip": history["precipitation"].tolist(),
                    "history_temp": history["temperature_2m"].tolist(),
                    "scaled_tensor": history_values.tolist(),
                    "full_static": full_static.tolist(),
                    "categorical": stat_cat_vals.tolist(),
                })

            X_dyn.append(history_values)
            X_stat_num.append(full_static)
            X_stat_cat.append(stat_cat_vals)
            y.append(target_value)
            target_dates.append(target_row["date"])
            basin_ids.append(mws_id)

    return (
        np.array(X_dyn, dtype=np.float32),
        np.array(X_stat_num, dtype=np.float32),
        np.array(X_stat_cat, dtype=np.int64),
        np.array(y, dtype=np.float32),
        np.array(basin_ids),
        np.array(target_dates),
        sample_debug,
    )


def single_window_tensors(window_df, static_num_vec, cat_ids):
    """Assemble ONE inference sample.

    window_df    : a processed, scaled single-location history of exactly WINDOW
                   rows (most recent first->last) containing config.DYNAMIC_FEATURES.
    static_num_vec : np.array of the scaled static numeric vector, ordered as
                     config.STATIC_NUMERICAL_FEATURES (base + forecast aggregates).
    cat_ids      : [state_id, district_id, tehsil_id] (label-encoded ints).

    Returns (x_dyn, x_stat_num, x_stat_cat) as float32/int64 numpy arrays with a
    leading batch dim of 1, matching the training tensor layout."""
    x_dyn = window_df[config.DYNAMIC_FEATURES].values.astype(np.float32)[None, ...]
    x_stat_num = np.asarray(static_num_vec, dtype=np.float32)[None, ...]
    x_stat_cat = np.asarray(cat_ids, dtype=np.int64)[None, ...]
    return x_dyn, x_stat_num, x_stat_cat
