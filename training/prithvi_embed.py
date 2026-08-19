"""
prithvi_embed.py
================
Shared helper for wiring FROZEN Prithvi static embeddings into the NDVI
forecasters (LSTM / TFT). Phase 1: the Prithvi encoder is run offline by
encode_static.py -> embeddings.parquet (mws_id, year, e0..e1023). Here we only
LOOK UP those vectors and project them (1024 -> proj_dim) with a small TRAINABLE
head learned end-to-end toward the NDVI loss. The Prithvi weights never appear.

Design notes
------------
- Memory: there are millions of training samples but only ~150-174k unique
  (mws_id, year) embeddings. So we store ONE matrix of unique vectors and a
  per-sample int index into it (not a 1024-wide vector per sample).
- Leak-safe pairing (Fix A): a sample with target date in year Y uses the
  composite from year Y-1. `idx_for(...)` applies the `lag` (default 1).
- Missing key (no composite for that mws_id/year): mapped to a reserved ZERO
  row, which is forced to 0 AFTER standardization so it stays neutral. We keep
  the sample (zeros) rather than dropping it, so the sample set is IDENTICAL
  whether --use_prithvi is on or off (fair ablation).
- Standardization is fit on TRAIN rows only (z-score), then applied to all rows.

Typical use in a training script
---------------------------------
    import prithvi_embed as pe
    emb_matrix, key_to_row, zero_idx, emb_dim = pe.load_embeddings(EMB_PATH)
    Xtr_idx, miss = pe.idx_for(train_basin_ids, train_target_dates, key_to_row, zero_idx)
    Xva_idx, _    = pe.idx_for(val_basin_ids,   val_target_dates,   key_to_row, zero_idx)
    Xte_idx, _    = pe.idx_for(test_basin_ids,  test_target_dates,  key_to_row, zero_idx)
    pe.standardize_(emb_matrix, Xtr_idx, zero_idx)        # in place, train-fit
    projector = pe.PrithviProjector(emb_matrix, proj_dim=32)   # nn.Module
    ... model receives emb_idx, calls projector(emb_idx) -> (B, proj_dim)
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


def load_embeddings(emb_path):
    """Read embeddings.parquet -> (emb_matrix, key_to_row, zero_idx, emb_dim).

    emb_matrix : float32 (K+1, emb_dim); the last row (zero_idx) is the reserved
                 "missing" row (zeros). key_to_row maps (str mws_id, int year,
                 str half) -> row.

    BACKWARD COMPAT: the key is always a 3-tuple (mws_id, year, half). Annual
    parquets predate the half-year work and have NO `half` column -> we fill it
    with "ANNUAL". So an annual lookup (mws, year-lag) becomes (mws, year-lag,
    "ANNUAL") and matches exactly as before. Half-year parquets carry "H1"/"H2".
    """
    df = pd.read_parquet(emb_path)
    df["mws_id"] = df["mws_id"].astype(str)
    df["year"] = df["year"].astype(int)
    if "half" not in df.columns:                              # old annual parquet
        df["half"] = "ANNUAL"
    df["half"] = df["half"].fillna("ANNUAL").astype(str)
    ecols = sorted((c for c in df.columns if c.startswith("e") and c[1:].isdigit()),
                   key=lambda c: int(c[1:]))
    mat = df[ecols].to_numpy(dtype=np.float32)               # (K, emb_dim)
    mat = np.concatenate([mat, np.zeros((1, mat.shape[1]), np.float32)], axis=0)
    zero_idx = mat.shape[0] - 1
    key_to_row = {(m, y, h): i for i, (m, y, h)
                  in enumerate(zip(df["mws_id"], df["year"], df["half"]))}
    halves = sorted(df["half"].unique().tolist())
    print(f"[prithvi_embed] loaded {len(df):,} embeddings "
          f"({df['mws_id'].nunique():,} MWS, years {sorted(df['year'].unique().tolist())}, "
          f"halves {halves}), dim={len(ecols)}")
    return mat, key_to_row, zero_idx, len(ecols)


def _season_of(ts, mode):
    """Period label for a timestamp under a seasonal emb_mode.
    halfyear -> "H1" (Jan-Jun) / "H2" (Jul-Dec);
    quarterly -> "Q1".."Q4" (Jan-Mar / Apr-Jun / Jul-Sep / Oct-Dec).
    Both modes share the SAME prior-year same-season walk-back; only the period
    granularity differs."""
    if mode == "quarterly":
        return "Q" + str((ts.month - 1) // 3 + 1)
    return "H1" if ts.month <= 6 else "H2"          # halfyear


def make_key(basin_id, target_date, lag=1, mode="annual", max_year=None):
    """The (mws_id, year, half) lookup key for one sample. Single source of truth
    for BOTH the bulk idx_for path and the per-MWS eval path, so they never drift.

    mode="annual"  (default, backward compatible): (mws, year(target)-lag, "ANNUAL").
    mode="halfyear"/"quarterly": (mws, min(year(target)-lag, max_year), season) with
        season = H1/H2 (halfyear) or Q1-Q4 (quarterly) of the TARGET month
        (prior-year SAME-SEASON analog; same-year same-season would overlap the
        target -> leak).
    """
    ts = pd.Timestamp(target_date)
    if mode in ("halfyear", "quarterly"):
        cyear = ts.year - lag
        if max_year is not None:
            cyear = min(cyear, max_year)
        return (str(basin_id), int(cyear), _season_of(ts, mode))
    return (str(basin_id), int(ts.year - lag), "ANNUAL")


def max_year_of(key_to_row):
    """Latest year present in the embedding table (half-year walk-back ceiling)."""
    return max((k[1] for k in key_to_row), default=None)


def min_year_of(key_to_row):
    """Earliest year present in the embedding table (half-year walk-back floor)."""
    return min((k[1] for k in key_to_row), default=None)


def resolve_row(basin_id, target_date, key_to_row, lag=1, mode="annual",
                min_year=None, max_year=None):
    """Row index into emb_matrix for ONE sample, or None if it should be dropped.

    Single source of truth for BOTH the bulk idx_for path and the per-MWS eval
    path, so they never drift.

    mode="annual" (default, backward compatible): the single key
        (mws, year(target)-lag, "ANNUAL"); returns None if that key is absent
        (callers in annual mode map None -> zero_idx and KEEP the sample).

    mode="halfyear"/"quarterly": "latest seasonal match with walk-back." season =
        H1/H2 (halfyear) or Q1-Q4 (quarterly) of the target month. Start at
        year(target)-lag (clamped to max_year so future/undownloaded years are
        skipped) and walk BACKWARDS year by year, SAME season, returning the first
        year whose tile exists. If NO year in [min_year, year-lag] has a tile for
        this (mws, season), returns None -> the caller DROPS the sample. lag=1
        guarantees we never read the same-year same-season tile (overlap = leak).
    """
    ts = pd.Timestamp(target_date)
    if mode in ("halfyear", "quarterly"):
        half = _season_of(ts, mode)
        start = ts.year - lag
        if max_year is not None:
            start = min(start, int(max_year))
        lo = int(min_year) if min_year is not None else -(10 ** 9)
        for y in range(int(start), lo - 1, -1):
            r = key_to_row.get((str(basin_id), int(y), half))
            if r is not None:
                return r
        return None
    return key_to_row.get(make_key(basin_id, target_date, lag=lag, mode="annual"))


def idx_for(basin_ids, target_dates, key_to_row, zero_idx, lag=1,
            mode="annual", max_year=None, min_year=None, return_keep=False):
    """Per-sample row index into emb_matrix. See resolve_row for the keying rule.

    Returns (idx int64 array, n_missing) by default. If return_keep=True, returns
    (idx, n_missing, keep_mask) — a bool array marking samples to RETAIN.

    annual (default, backward compatible): a missing key -> zero_idx and the
        sample is KEPT (keep=True) as a neutral zero vector, exactly as before.
    halfyear/quarterly: resolve_row walks back same-season; a sample with NO tile
        in any year gets idx=zero_idx (placeholder) and keep=False so the caller
        can DROP it.

    The default 2-tuple return keeps existing callers (e.g. the _FT scripts and
    the docstring example) working unchanged.
    """
    if mode in ("halfyear", "quarterly"):
        if max_year is None:
            max_year = max_year_of(key_to_row)
        if min_year is None:
            min_year = min_year_of(key_to_row)
    idx = np.empty(len(basin_ids), dtype=np.int64)
    keep = np.ones(len(basin_ids), dtype=bool)
    miss = 0
    for i, (b, d) in enumerate(zip(basin_ids, target_dates)):
        r = resolve_row(b, d, key_to_row, lag=lag, mode=mode,
                        min_year=min_year, max_year=max_year)
        if r is None:
            idx[i] = zero_idx
            miss += 1
            if mode in ("halfyear", "quarterly"):
                keep[i] = False          # DROP: no seasonal tile in any year
        else:
            idx[i] = r
    if return_keep:
        return idx, miss, keep
    return idx, miss


def standardize_(emb_matrix, train_idx, zero_idx, eps=1e-6):
    """In-place z-score of emb_matrix, fit on UNIQUE train rows only.

    The zero (missing) row is forced back to 0 so missing stays neutral.
    Returns (mu, sd) for saving/inference reuse.
    """
    rows = np.unique(train_idx)
    rows = rows[rows != zero_idx]
    if len(rows) == 0:
        raise SystemExit("[prithvi_embed] no train embedding rows to fit standardizer")
    mu = emb_matrix[rows].mean(axis=0)
    sd = emb_matrix[rows].std(axis=0) + eps
    emb_matrix[:] = (emb_matrix - mu) / sd
    emb_matrix[zero_idx] = 0.0
    return mu, sd


class PrithviProjector(nn.Module):
    """Frozen embedding lookup + trainable 1024->proj_dim projection head.

    The embedding matrix is a non-trainable buffer (moves with .to(device)).
    Only the projection Linear/LayerNorm learn.
    """

    def __init__(self, emb_matrix, proj_dim=32, dropout=0.2):
        super().__init__()
        self.register_buffer(
            "emb_matrix", torch.as_tensor(emb_matrix, dtype=torch.float32))
        in_dim = emb_matrix.shape[1]
        self.out_dim = proj_dim
        self.proj = nn.Sequential(
            nn.Linear(in_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, emb_idx):
        e = self.emb_matrix[emb_idx]        # (B, in_dim), frozen
        return self.proj(e)                 # (B, proj_dim), trainable
