# data/

Holds the **training dataset** — i.e. the **AIKosh Dataset asset**. NOT committed to
git; downloaded / regenerated here.

Contents:

```
final_spei_output.csv                # numerical NDVI + weather + SPEI-3 history (download)
all_mws_locations_dedup.csv          # MWS locations (mws_id, lat, lon) — tile-generation manifest
```

- **Inference does NOT need this folder** — only `training/` (fine-tune / retrain) does.
- The 755k image tiles are **not** shipped: regenerate them from the manifest with the tile
  generator `training/download_composites.py`. Ships the *recipe*, not the tiles.

See **[../DATA.md](../DATA.md)** for the schema, provenance, and licensing (CC BY 4.0).
