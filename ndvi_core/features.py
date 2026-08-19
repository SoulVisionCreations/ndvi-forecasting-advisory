"""
ndvi_core.features — leakage-safe feature engineering + static profile.

Extracted (behavior-preserving) from the shared spine of
Run_With_MWS_Split_Temporal_TFT_FT.py (verified byte-identical to its _FT
sibling for the feature-eng region). This is the de-duplication target: the
training driver AND the inference path both call these functions, so the exact
same feature math runs at train and serve time.

Design — fit vs transform (so one codebase serves both, leakage-safe):
  * fit_*(...)        -> returns stats fitted on the TRAIN split (radiation
                        clip limits, quarterly baselines, phenology, label
                        encoders). Called once at train time; saved as artifacts.
  * add_*/apply_*(df) -> pure per-row transforms; run at BOTH train and inference,
                        taking the precomputed stats as inputs.

At inference a request's history is massaged into the same column schema
(mws_id, date, the weather cols) and these same functions are applied using the
SAVED stats (scaler/encoders from artifacts; quarterly/phenology/peak from the
mws_static_lookup row), guaranteeing parity with training.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from . import config


# ===========================================================================
# Cleaning + temporal structure
# ===========================================================================

def rename_and_clean(df, weather_cols=None, rename_map=None):
    """Rename raw ERA5 cols, null out -9999 poison values, drop rows with no
    NDVI/temperature. (Source: RENAME + BASIC CLEANING blocks.)"""
    weather_cols = weather_cols or config.WEATHER_COLS
    rename_map = rename_map or config.RENAME_MAP
    df = df.rename(columns=rename_map)
    df[weather_cols] = df[weather_cols].replace(-9999, np.nan)
    df = df.dropna(subset=["NDVI", "temperature_2m"])
    return df


def parse_and_sort(df):
    """Parse `date` to datetime and sort chronologically per MWS."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values(["mws_id", "date"]).reset_index(drop=True)


def assign_temporal_split(df, train_ratio, val_ratio):
    """Per-MWS chronological 70/20/10 split tag (TRAIN-only stats safety).
    (Source: TEMPORAL SPLIT block.)

    Implemented via explicit group iteration + concat rather than
    ``groupby(...).apply(func-returning-group)``: on pandas >= 3.0 the latter
    EXCLUDES the grouping column ('mws_id') from the frame it returns, which
    silently dropped mws_id and broke every downstream mws_id groupby. Iterating
    groups keeps all columns and is bit-identical to the original per-group
    slicing on every pandas version (2.x and 3.x)."""
    parts = []
    for _, group in df.groupby("mws_id", sort=True):
        g = group.copy()
        n = len(g)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        split = np.empty(n, dtype=object)
        split[:train_end] = "train"
        split[train_end:val_end] = "val"
        split[val_end:] = "test"
        g["split"] = split
        parts.append(g)
    return pd.concat(parts)


# ===========================================================================
# Hierarchical imputation
# ===========================================================================

def apply_hierarchical_imputation(df, weather_cols=None):
    """Fill weather-col NaNs by MWS -> district -> state -> global median.

    NOTE (behavior-preserving): the original script computes train-split medians
    first but then *recomputes and applies* medians from the full frame; we
    replicate the applied behavior exactly (medians from `df`). (Source:
    LEAKAGE-SAFE IMPUTATION 'APPLY' block.)"""
    weather_cols = weather_cols or config.WEATHER_COLS
    df = df.copy()

    mws_medians = df.groupby("mws_id")[weather_cols].median()
    for col in weather_cols:
        df[col] = df[col].fillna(df["mws_id"].map(mws_medians[col]))

    if "district" in df.columns:
        district_medians = df.groupby("district")[weather_cols].median()
        for col in weather_cols:
            df[col] = df[col].fillna(df["district"].map(district_medians[col]))

    if "state" in df.columns:
        state_medians = df.groupby("state")[weather_cols].median()
        for col in weather_cols:
            df[col] = df[col].fillna(df["state"].map(state_medians[col]))

    df[weather_cols] = df[weather_cols].fillna(df[weather_cols].median())
    return df


# ===========================================================================
# Time features
# ===========================================================================

def add_time_features(df):
    """year/month/doy/quarter/season/days_since_start + sin/cos encodings.
    (Source: TIME FEATURES block.)"""
    df = df.copy()
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day_of_year"] = df["date"].dt.dayofyear
    df["quarter"] = df["date"].dt.quarter
    df["season"] = df["month"].apply(
        lambda x: 1 if x in [12, 1, 2] else 2 if x in [3, 4, 5]
        else 3 if x in [6, 7, 8] else 4
    )
    df["days_since_start"] = (df["date"] - df["date"].min()).dt.days
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["doy_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365)
    df["doy_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365)
    return df


# ===========================================================================
# Engineered / hardened features
# ===========================================================================

def fit_radiation_clip(fit_train_df):
    """Per-radiation-col 99th-pct upper clip limit, fit on TRAIN only.
    (Source: HARDENED FEATURES radiation clipping.)"""
    rad_cols = [
        "surface_solar_radiation_downwards_sum",
        "surface_net_thermal_radiation_sum",
    ]
    return {c: fit_train_df[c].quantile(0.99) for c in rad_cols}


def add_engineered_features(df, rad_clip_limits):
    """precip_runoff_diff, NDVI/precip/et lags, precip_et_ratio, temp_precip_idx,
    evap_stress, radiation clip, NDVI rolling mean/std, NDVI_change_prev,
    precip_cum_3. (Source: BASIC ENGINEERED + HARDENED FEATURES blocks.)

    rad_clip_limits: {col: upper} from fit_radiation_clip (train) at train time;
    the SAME saved limits at inference."""
    df = df.copy()
    df["precip_runoff_diff"] = df["precipitation"] - df["runoff"]

    grouped = df.groupby("mws_id")
    df["precipitation_prev"] = grouped["precipitation"].shift(1)
    df["et_prev"] = grouped["et"].shift(1)
    df["NDVI_prev"] = grouped["NDVI"].shift(1)
    df["NDVI_prev2"] = grouped["NDVI"].shift(2)

    df["precip_et_ratio"] = (df["precipitation"] / (df["et"] + 0.01)).clip(0, 50)
    df["temp_precip_idx"] = df["temperature_2m"] * df["precipitation"]
    df["evap_stress"] = (df["temperature_2m"] / (df["sm_rootzone"] + 0.01)).clip(0, 1000)

    for rad_col, upper in rad_clip_limits.items():
        df[rad_col] = df[rad_col].clip(None, upper)

    grouped = df.groupby("mws_id")  # re-group after column edits
    for col in ["NDVI"]:
        df[f"{col}_rolling_mean"] = (
            grouped[col].shift(1).transform(lambda x: x.rolling(3, min_periods=1).mean())
        )
        df[f"{col}_rolling_std"] = (
            grouped[col].shift(1).transform(lambda x: x.rolling(3, min_periods=1).std()).fillna(0)
        )

    df["NDVI_change_prev"] = df["NDVI_prev"] - df["NDVI_prev2"]
    df["precip_cum_3"] = (
        grouped["precipitation"].transform(lambda x: x.rolling(3).sum()).fillna(0)
    )
    return df


# ===========================================================================
# Label encoding
# ===========================================================================

def fit_label_encoders(df, cols=None):
    """Fit a LabelEncoder per categorical source col and add `{col}_encoded`.
    Returns (df, encoders). (Source: LABEL ENCODING block.)"""
    cols = cols or config.CATEGORICAL_SOURCE_COLS
    df = df.copy()
    encoders = {}
    for col in cols:
        le = LabelEncoder()
        df[f"{col}_encoded"] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
    return df, encoders


def apply_label_encoders(df, encoders, cols=None):
    """Inference counterpart of fit_label_encoders: apply SAVED encoders to add
    `{col}_encoded`. Unseen/missing categories -> 0 (neutral fallback, matching
    the deployed _safe_encode). Encoded on the raw str (same as training's
    fit_transform on .astype(str))."""
    cols = cols or config.CATEGORICAL_SOURCE_COLS
    df = df.copy()
    for col in cols:
        le = encoders.get(col)
        if le is None:
            df[f"{col}_encoded"] = 0
            continue
        classes = set(le.classes_.tolist())
        df[f"{col}_encoded"] = df[col].map(
            lambda v: int(le.transform([str(v)])[0]) if str(v) in classes else 0
        )
    return df


# ===========================================================================
# Quarterly baselines + phenology (TRAIN-fit) + deviation
# ===========================================================================

def fit_quarterly_baselines(fit_train_df):
    """Per-(mws, quarter) NDVI mean/std pivoted to q{n}_{mean,std}_ndvi, fit on
    TRAIN. (Source: TRAIN-ONLY QUARTERLY BASELINES.)"""
    df = fit_train_df.copy()
    df["quarter"] = ((df["month"] - 1) // 3) + 1
    stats = df.groupby(["mws_id", "quarter"])["NDVI"].agg(["mean", "std"]).reset_index()
    pivot = stats.pivot(index="mws_id", columns="quarter", values=["mean", "std"])
    pivot.columns = [f"q{c[1]}_{c[0]}_ndvi" for c in pivot.columns]
    return pivot.reset_index()


def fit_phenology(fit_train_df):
    """Per-MWS historical_max_ndvi + peak_month, fit on TRAIN.
    (Source: TRAIN-ONLY PHENOLOGY.)"""
    pheno = fit_train_df.groupby("mws_id")["NDVI"].agg(["max", "idxmax"]).reset_index()
    pheno["peak_month"] = fit_train_df.loc[pheno["idxmax"], "month"].values
    return pheno.rename(columns={"max": "historical_max_ndvi"})[
        ["mws_id", "historical_max_ndvi", "peak_month"]
    ]


def merge_static_baselines(df, quarterly_train, phenology_train):
    """Left-merge the train-fit quarterly + phenology tables and add
    months_from_peak. (Source: the two merges + months_from_peak.)"""
    df = df.merge(quarterly_train, on="mws_id", how="left")
    df = df.merge(phenology_train, on="mws_id", how="left")
    df["months_from_peak"] = df["month"] - df["peak_month"]
    return df


def add_ndvi_q_deviation(df):
    """ndvi_q_deviation = NDVI - that row's quarter mean (inf/NaN -> 0).
    (Source: QUARTERLY DEVIATION block.)"""
    df = df.copy()

    def _q_mean(row):
        return row.get(f"q{int(row['quarter'])}_mean_ndvi", np.nan)

    df["ndvi_q_deviation"] = df["NDVI"] - df.apply(_q_mean, axis=1)
    df["ndvi_q_deviation"] = (
        df["ndvi_q_deviation"].replace([np.inf, -np.inf], np.nan).fillna(0)
    )
    return df


# ===========================================================================
# Null drop + static lookup export
# ===========================================================================

def drop_critical_nulls(df):
    """Drop rows missing any critical (dynamic + static-base + key) column.
    (Source: TARGETED NULL DROP.)"""
    critical_cols = list(set(
        config.DYNAMIC_FEATURES
        + config.STATIC_BASE_FEATURES
        + ["date", "mws_id", "split", "NDVI"]
    ))
    return df.dropna(subset=critical_cols)


def static_lookup_table(df, features=None):
    """One row per MWS of the static profile columns (for the inference lookup).
    Call before scaling -> UNSCALED table; after scaling -> SCALED table.
    (Source: EXPORT (UN)SCALED STATIC LOOKUP TABLE.)"""
    features = features or config.LOOKUP_FEATURES
    return (
        df[features]
        .drop_duplicates(subset=["mws_id"])
        .sort_values("mws_id")
        .reset_index(drop=True)
    )
