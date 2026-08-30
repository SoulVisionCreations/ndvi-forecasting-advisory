# Setup — NDVI_Forecasting

Validated on: Python 3.11, CUDA 12.8, NVIDIA H200 (linux-x86_64).

## 1. Environment (uv)
```bash
uv venv --python 3.11 .venv
source .venv/bin/activate
uv pip install -e .              # inference + serving
# uv pip install -e ".[train]"  # + fine-tune/eval tools (matplotlib, seaborn, tqdm)
```
`torch==2.10.0` from PyPI is the CUDA-12.8 (`+cu128`) build on linux-x86_64 —
GPU works out of the box, **no special index needed**.

(plain `pip` works too: `python -m venv .venv && pip install -e .`)

## 2. Model weights
Download the 7-file model asset into `weights/` (see `weights/README.md`). It is
self-contained — no Hugging Face / base-model download required.

## 3. Credentials (env — never committed)
```bash
earthengine authenticate      # Google Earth Engine (satellite fetch)
export CORESTACK_KEY=...       # CoreStack admin lookup
```

## 4. Run a forecast
```bash
python inference/infer_cli.py --lat 25.44 --lon 91.71 --date 2024-08-15 \
    --run_dir weights --model_dir weights \
    --lookup_csv weights/mws_static_lookup_UNSCALED.tsv
```
Or serve: `uvicorn inference.serve_api:app` (env vars set the same paths).

## 5. (Optional) Vegetation Outlook advisory
The flagship use case — reuses the same model in-process; a leashed Gemma-3-4B (via Ollama) rewords the
message (template fallback).
```bash
curl -fsSL https://ollama.com/install.sh | sh   # Ollama (Linux); serves :11434
ollama pull gemma3:4b                           # ~3.3 GB

# from the repo root; needs weights/ + a GPU + CORESTACK_KEY
RUN_DIR=weights MODEL_DIR=weights LOOKUP_CSV=weights/mws_static_lookup_UNSCALED.tsv \
  ADVISORY_ENGINE=inprocess ADVISORY_SLM=1 ADVISORY_SLM_MODEL=gemma3:4b CORESTACK_KEY=... \
  python -m uvicorn advisory.serve_advisory:app --port 8011   # run inside the activated venv
# POST /advisory {"lat":20.17,"lon":78.32,"regime":"rain-fed","date":"2018-06-15"}
```
`ADVISORY_SLM=0` = template-only (no Ollama). See `advisory/README.md`.
