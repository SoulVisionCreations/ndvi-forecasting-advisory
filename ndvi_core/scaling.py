"""
ndvi_core.scaling — thin, shared helpers around the saved StandardScaler.

Training fits one StandardScaler on SCALE_COLS (train split only) and saves it.
Inference must apply the IDENTICAL transform and correctly invert the model's
NDVI output. Previously inference re-implemented this by hand
(_scale_dynamic / _scale_static_num / _unscale_prediction); those are replaced
by these functions so both paths share one definition.
"""

import numpy as np

from . import config


def apply_scaler(df, scaler, cols=None):
    """Standardise `cols` of `df` in place-style (returns a copy) using a fitted
    StandardScaler. Mirrors training: scaler.transform on exactly SCALE_COLS,
    NaN/inf coerced to 0 first (matches the inference contract)."""
    cols = list(cols) if cols is not None else list(config.SCALE_COLS)
    out = df.copy()
    block = out[cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    out[cols] = scaler.transform(block)
    return out


def scale_value(scaler, col, raw):
    """Standardise a single raw value for one SCALE_COLS column using that
    column's fitted (mean, scale). Used for forecast aggregates / q-means."""
    i = config.SCALE_COLS.index(col)
    return (raw - scaler.mean_[i]) / scaler.scale_[i]


def inverse_ndvi(scaler, pred_scaled_value):
    """Invert the model's scalar output back to raw NDVI.

    The model returns (current_ndvi_scaled + delta_scaled), a value in NDVI's
    standardised space. We invert it through the scaler's NDVI mean/scale by
    placing it in a dummy row's NDVI slot and inverse_transform-ing, then clip
    to [0, 1]. (Identical to the worker's former _unscale_prediction / FIX-1.)"""
    dummy = np.zeros((1, len(config.SCALE_COLS)), dtype=np.float32)
    dummy[0, config.NDVI_SCALE_IDX] = pred_scaled_value
    unscaled = scaler.inverse_transform(dummy)
    return float(np.clip(unscaled[0, config.NDVI_SCALE_IDX], 0.0, 1.0))
