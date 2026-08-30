"""
prithvi_finetune.py
===================
Phase 2: JOINT supervised fine-tune of Prithvi (LoRA) + the 1024->proj_dim head
+ the forecaster, all toward the NDVI loss. Unlike Phase 1 (frozen encoder, an
offline embeddings.parquet looked up by index), here the Prithvi encoder is IN
the training graph: each sample's (mws_id, year-1) annual composite tile is
loaded, encoded through Prithvi+LoRA, standardized with the Phase-1 (mu, sd),
projected, and concatenated into the forecaster statics.

Why the forecaster code is untouched
------------------------------------
The Phase-1 forecaster already threads a per-sample integer `emb_idx` end to end
and calls `self.prithvi(emb_idx) -> (B, proj_dim)`. Phase 2 simply REINTERPRETS
that integer as a TILE index and swaps `self.prithvi` for `LivePrithviProjector`,
which loads + encodes the tile live. Dataset, DataLoaders, train/val/test loops,
metrics, and the forecaster modules are all unchanged -> apples-to-apples.

Warm-start exactness
--------------------
LoRA initializes to zero, so the live encoder reproduces the EXACT frozen vectors
that produced embeddings.parquet. Standardizing with the same (mu, sd) and loading
the Phase-1 projection head makes step-0 behaviour identical to Phase 1; LoRA then
adapts the encoder from that optimum.

LoRA scope
----------
timm ViT blocks use a FUSED `attn.qkv` Linear, so LoRA on `qkv` adapts q, k, v
together (q,v cannot be cleanly isolated in the fused layer). We target ONLY the
encoder blocks (the decoder is never run by forward_features).

Compute
-------
~156k unique TRAIN tiles vs ~2.16M samples. `LivePrithviProjector.forward` always
de-duplicates tiles within the batch and encodes each unique tile once. With the
tile-grouped sampler (recommended) a batch covers few unique tiles, so each is
encoded ~once per epoch instead of ~14x.
"""

import os
import re
import sys
import glob
import json
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn as nn

FNAME_RE = re.compile(r"^composite_(?P<mws>.+)_(?P<year>\d{4})\.tif$")


def amp_ctx(device, enabled=True):
    """bf16 autocast on CUDA (matches encode_static.py); no-op elsewhere."""
    if enabled and str(device).startswith("cuda"):
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return nullcontext()


# ---------------------------------------------------------------------------
# Prithvi base + LoRA on encoder attention (fused qkv)
# ---------------------------------------------------------------------------
def load_prithvi_lora(model_dir, r=16, alpha=32, dropout=0.05, num_frames=1,
                      device="cuda", target="qkv", exclude_ckpt=None):
    """Instantiate PrithviMAE, load .pt weights, freeze base, attach LoRA to the
    encoder attention (`encoder.blocks.*.attn.<target>`). Returns (peft_model, cfg).

    exclude_ckpt: a .pt path to skip when globbing model_dir for the base
    checkpoint (i.e. the caller's FUSED bundle). In a self-contained weights/ dir
    the only .pt is that bundle; once excluded no base checkpoint remains, so
    PrithviMAE is left at init and the bundle (loaded downstream by build_model,
    strict=False) supplies the backbone. The backbone is FROZEN in training, so
    the bundle's backbone tensors are byte-identical to the base weights -> the
    result is identical to loading a base .pt here and overwriting it.
    """
    sys.path.insert(0, model_dir)
    from prithvi_mae import PrithviMAE
    from peft import LoraConfig, get_peft_model

    with open(os.path.join(model_dir, "config.json")) as f:
        cfg = json.load(f)["pretrained_cfg"]

    args = dict(cfg)
    args["num_frames"] = num_frames                 # T=1 spatial encoder
    model = PrithviMAE(**args)

    ckpts = [c for c in glob.glob(os.path.join(model_dir, "*.pt"))
             if exclude_ckpt is None
             or os.path.abspath(c) != os.path.abspath(exclude_ckpt)]
    if not ckpts:
        if exclude_ckpt is None:
            raise SystemExit(f"No .pt checkpoint in {model_dir}")
        # Self-contained dir: the only .pt was the fused bundle (excluded). Leave
        # PrithviMAE at init; build_model's bundle load fills the frozen backbone.
        print("[ft] no separate base checkpoint in model_dir (fused bundle "
              "excluded) -> backbone comes from the bundle downstream.")
    else:
        ckpt0 = ckpts[0]
        if os.path.getsize(ckpt0) < 10_000_000:
            raise SystemExit(
                f"{ckpt0} looks like a Git LFS pointer, not the model. Run:\n"
                f"  cd {model_dir} && git lfs install && git lfs pull")
        state = torch.load(ckpt0, map_location="cpu", weights_only=False)
        state = state.get("model", state) if isinstance(state, dict) else state
        for k in [k for k in state if "pos_embed" in k]:   # recomputed for T=1
            del state[k]
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"[ft] loaded {os.path.basename(ckpt0)} | missing={len(missing)} "
              f"unexpected={len(unexpected)} (pos_embed expected)")

    # LoRA only on encoder attention; decoder is never used by forward_features.
    n_blocks = len(model.encoder.blocks)
    targets = [f"encoder.blocks.{i}.attn.{target}" for i in range(n_blocks)]
    lora_cfg = LoraConfig(r=r, lora_alpha=alpha, lora_dropout=dropout,
                          target_modules=targets, bias="none")
    model = get_peft_model(model, lora_cfg)

    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"[ft] LoRA r={r} alpha={alpha} drop={dropout} on {n_blocks} encoder "
          f"blocks ({target}) | trainable {n_train:,} / {n_total:,} "
          f"({100*n_train/n_total:.3f}%)")
    return model.to(device), cfg


def encode_tokens(peft_model, img, tcoords, loc, pool="mean"):
    """forward_features -> last (LayerNorm'd) tokens -> pooled (B, embed_dim).
    Grad flows when called outside no_grad (training); identical pooling to
    encode_static.py so warm-start matches the frozen embeddings."""
    feats = peft_model.forward_features(img, tcoords, loc)
    tokens = feats[-1]                                  # (B, 1+num_patches, D)
    if pool == "cls":
        return tokens[:, 0, :]
    return tokens[:, 1:, :].mean(dim=1)


# ---------------------------------------------------------------------------
# Tile store: tile_idx -> normalized (img, temporal_coords, location_coords)
# ---------------------------------------------------------------------------
class TileStore:
    """Maps a tile index (row in the (mws_id, year) table) to a normalized tensor
    triple, matching encode_static.CompositeDataset exactly. Threaded batch loads
    + a bounded warm cache (helpful when tiles repeat across batches)."""

    def __init__(self, keys, composite_dir, latlon, mean, std, doy=182,
                 cache_size=20000, n_threads=8):
        self.keys = keys                                # list[(mws_id:str, year:int)]
        self.dir = composite_dir
        self.latlon = latlon
        self.mean = np.asarray(mean, np.float32)[:, None, None]
        self.std = np.asarray(std, np.float32)[:, None, None]
        self.doy = float(doy)
        self.cache = {}
        self.cache_size = cache_size
        self.pool = ThreadPoolExecutor(max_workers=n_threads)

    # Representative day-of-year per period (matches encode_static.py); annual
    # tiles fall back to self.doy. Feeds the Prithvi temporal coordinate.
    _PERIOD_DOY = {"H1": 90, "H2": 274, "Q1": 46, "Q2": 136, "Q3": 227, "Q4": 319}

    def _path(self, mws, year, half="ANNUAL"):
        if half and half != "ANNUAL":
            return os.path.join(self.dir, f"composite_{mws}_{year}_{half}.tif")
        return os.path.join(self.dir, f"composite_{mws}_{year}.tif")

    def _load_one(self, idx):
        hit = self.cache.get(idx)
        if hit is not None:
            return hit
        import rasterio
        # key_to_row yields (mws, year, half) 3-tuples (annual -> half="ANNUAL",
        # half-year -> H1/H2, quarterly -> Q1-4); tolerate the legacy 2-tuple too.
        _k = self.keys[idx]
        mws, year = _k[0], _k[1]
        half = _k[2] if len(_k) > 2 else "ANNUAL"
        with rasterio.open(self._path(mws, year, half)) as src:
            arr = src.read().astype(np.float32)         # (6, 224, 224)
            nd = src.nodata
        nodata = (arr == 0)
        if nd is not None:
            nodata |= (arr == nd)
        arr = (arr - self.mean) / self.std
        arr[nodata] = 0.0                               # neutral after norm
        lat, lon = self.latlon[mws]
        doy = float(self._PERIOD_DOY.get(half, self.doy))  # per-period doy
        rec = (arr,
               np.array([[float(year), doy]], np.float32),   # (1,2)
               np.array([float(lat), float(lon)], np.float32))    # (2,)
        if len(self.cache) < self.cache_size:
            self.cache[idx] = rec
        return rec

    def load_batch(self, idxs, device):
        recs = list(self.pool.map(self._load_one, idxs))
        img = torch.from_numpy(np.stack([r[0] for r in recs])).to(device)
        tc = torch.from_numpy(np.stack([r[1] for r in recs])).to(device)
        lc = torch.from_numpy(np.stack([r[2] for r in recs])).to(device)
        return img, tc, lc


# ---------------------------------------------------------------------------
# Live projector — drop-in replacement for prithvi_embed.PrithviProjector
# ---------------------------------------------------------------------------
class LivePrithviProjector(nn.Module):
    """forward(tile_idx) -> (B, proj_dim). Encodes each UNIQUE tile in the batch
    once (LoRA in graph), standardizes with (mu, sd), projects 1024->proj_dim.
    Missing tiles (tile_idx == zero_idx) -> zeros, matching Phase-1 semantics.
    The `proj` head mirrors prithvi_embed.PrithviProjector so Phase-1 weights load.
    """

    def __init__(self, lora_model, tile_store, mu, sd, zero_idx, in_dim=1024,
                 proj_dim=32, dropout=0.2, pool="mean", encode_chunk=256,
                 amp=True, device="cuda", norm="musd"):
        super().__init__()
        self.encoder = lora_model                       # PEFT-wrapped PrithviMAE
        self.store = tile_store
        self.zero_idx = int(zero_idx)
        self.pool = pool
        self.encode_chunk = int(encode_chunk)
        self.amp = amp
        self.dev = device
        self.out_dim = proj_dim
        # Pre-projection standardization of the pooled 1024-vector. Two modes:
        #  - "musd"      : fixed offline (mu, sd) from embeddings.parquet — reproduces
        #                  a frozen Phase-1 exactly (the warm-start / two-phase path).
        #  - "layernorm" : inline LEARNED LayerNorm(in_dim) (GWL-style) — for the
        #                  from-scratch joint train; needs no parquet (mu, sd).
        self.norm = norm
        if norm == "layernorm":
            self.pre_norm = nn.LayerNorm(in_dim)
        elif norm == "musd":
            self.register_buffer("mu", torch.as_tensor(mu, dtype=torch.float32))
            self.register_buffer("sd", torch.as_tensor(sd, dtype=torch.float32))
        else:
            raise ValueError(f"norm must be 'musd' or 'layernorm', got {norm!r}")
        self.proj = nn.Sequential(
            nn.Linear(in_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def _encode_unique(self, uniq_idxs):
        outs = []
        for s in range(0, len(uniq_idxs), self.encode_chunk):
            chunk = uniq_idxs[s:s + self.encode_chunk]
            img, tc, lc = self.store.load_batch(chunk, self.dev)
            with amp_ctx(self.dev, self.amp):
                emb = encode_tokens(self.encoder, img, tc, lc, self.pool)
            outs.append(emb.float())
        return torch.cat(outs, 0)                        # (U, in_dim)

    def forward(self, tile_idx):
        w = self.proj[0].weight
        idx = tile_idx.detach().to("cpu").numpy().astype(np.int64)
        B = len(idx)
        out = torch.zeros(B, self.out_dim, device=w.device, dtype=w.dtype)

        present = idx != self.zero_idx
        if not present.any():
            return out
        pos = np.nonzero(present)[0]
        uniq, inv = np.unique(idx[pos], return_inverse=True)
        emb_u = self._encode_unique([int(x) for x in uniq])      # (U, in_dim)
        if self.norm == "layernorm":
            emb_u = self.pre_norm(emb_u)                          # learned (GWL-style)
        else:
            emb_u = (emb_u - self.mu) / self.sd                  # fixed offline (mu,sd)
        proj_u = self.proj(emb_u)                                # (U, proj_dim)
        scattered = proj_u[torch.as_tensor(inv, device=proj_u.device)]
        out[torch.as_tensor(pos, device=out.device)] = scattered.to(out.dtype)
        return out


# ---------------------------------------------------------------------------
# Tile-grouped batch sampler — few unique tiles per batch (fast encoding)
# ---------------------------------------------------------------------------
class TileGroupedBatchSampler(torch.utils.data.Sampler):
    """Each batch = n_tiles tiles x samples_per_tile samples, so a batch of
    n_tiles*samples_per_tile rows touches only n_tiles unique tiles. Reshuffles
    every epoch. This makes joint training ~ (samples_per_tile)x cheaper than
    per-sample random batching, at the cost of exact epoch composition.

    Missing-tile samples (tile_idx == zero_idx) are grouped under the zero key and
    still trained (they contribute a zero Prithvi block, as in Phase 1).
    """

    def __init__(self, tile_idx, n_tiles=256, samples_per_tile=32, seed=42):
        self.by_tile = {}
        for i, t in enumerate(np.asarray(tile_idx).astype(np.int64)):
            self.by_tile.setdefault(int(t), []).append(i)
        self.tiles = list(self.by_tile.keys())
        self.n_tiles = int(n_tiles)
        self.spt = int(samples_per_tile)
        self.seed = int(seed)
        self.epoch = 0
        self._len = max(1, len(tile_idx) // (self.n_tiles * self.spt))

    def __len__(self):
        return self._len

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        self.epoch += 1
        tiles = self.tiles.copy()
        rng.shuffle(tiles)
        for b in range(self._len):
            start = (b * self.n_tiles) % len(tiles)
            chosen = tiles[start:start + self.n_tiles]
            if len(chosen) < self.n_tiles:
                chosen = chosen + tiles[:self.n_tiles - len(chosen)]
            batch = []
            for t in chosen:
                pool = self.by_tile[t]
                sel = rng.choice(pool, size=self.spt, replace=len(pool) < self.spt)
                batch.extend(int(x) for x in sel)
            rng.shuffle(batch)
            yield batch


# ---------------------------------------------------------------------------
# Helpers for the training scripts
# ---------------------------------------------------------------------------
def tile_keys_from_parquet(key_to_row, n_rows):
    """Invert key_to_row -> ordered list[(mws_id, year)] of length n_rows (row ==
    tile_idx). The reserved zero row has no key (None)."""
    keys = [None] * n_rows
    for k, r in key_to_row.items():
        if 0 <= r < n_rows:
            keys[r] = k
    return keys


def build_tile_table_from_folder(composite_dir):
    """Build the tile table by scanning the composite FOLDER directly (GWL-style),
    so the live FT path needs NO embeddings.parquet.

    Returns (tile_keys, key_to_row, zero_idx) — the SAME triple the parquet path
    yields (tile_keys_from_parquet + load_embeddings), so downstream
    ``prithvi_embed.idx_for(...)`` maps samples -> tiles identically. Tile VALUES
    are re-encoded live, so only the key registry is needed here.

    Each ``composite_<mws>_<year>[_<H1|H2|Q1-4>].tif`` is parsed into a
    (mws_id:str, year:int, half:str) key (annual files -> half="ANNUAL"), exactly
    matching ``prithvi_embed.make_key``'s output. Row index == tile_idx (files
    sorted for determinism); zero_idx == n_tiles (reserved missing-tile row). The
    absolute row numbers differ from the parquet's, but the key->tile resolution is
    identical (verify with --validate_tile_table).
    """
    import re
    fname_re = re.compile(
        r"^composite_(?P<mws>.+)_(?P<year>\d{4})(?:_(?P<half>H[12]|Q[1-4]))?\.tif$")
    files = sorted(os.path.basename(p)
                   for p in glob.glob(os.path.join(composite_dir, "*.tif")))
    tile_keys, key_to_row, n_bad, n_dup = [], {}, 0, 0
    for f in files:
        m = fname_re.match(f)
        if m is None:
            n_bad += 1
            continue
        key = (str(m.group("mws")), int(m.group("year")),
               m.group("half") or "ANNUAL")
        if key in key_to_row:                        # duplicate file for same key
            n_dup += 1
            continue
        key_to_row[key] = len(tile_keys)
        tile_keys.append(key)
    zero_idx = len(tile_keys)                         # reserved missing-tile row
    if n_bad or n_dup:
        print(f"[ft][folder] note: skipped {n_bad} non-matching + {n_dup} duplicate "
              f".tif names in {composite_dir}")
    print(f"[ft][folder] scanned {composite_dir}: {len(tile_keys):,} tiles "
          f"(zero_idx={zero_idx})")
    return tile_keys, key_to_row, zero_idx


def split_param_groups(model, ft_lr, forecaster_lr_scale):
    """Two LR groups: Prithvi (LoRA + proj) at ft_lr; forecaster at a scaled LR."""
    prithvi, forecaster = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (prithvi if n.startswith("prithvi.") else forecaster).append(p)
    groups = [{"params": prithvi, "lr": ft_lr},
              {"params": forecaster, "lr": ft_lr * forecaster_lr_scale}]
    n_p = sum(p.numel() for p in prithvi)
    n_f = sum(p.numel() for p in forecaster)
    print(f"[ft] trainable: prithvi(LoRA+proj) {n_p:,} @ lr {ft_lr:g} | "
          f"forecaster {n_f:,} @ lr {ft_lr * forecaster_lr_scale:g}")
    return groups
