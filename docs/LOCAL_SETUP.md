# Local setup — run the Vegetation Outlook advisory end-to-end

Stand up the advisory on your own machine (macOS/Linux, **CPU or GPU**). ~10 min; the first
request then takes ~1–2 min (CPU + live data fetch). Every step below runs in the **activated venv**.

## 0. Prerequisites
- **`uv`** (https://docs.astral.sh/uv/) and **`git`**.
- Internet access (reaches Google Earth Engine, CoreStack, Open-Meteo).
- A **Google account** + **Earth Engine registered to a Google Cloud project** (free for research /
  non-commercial). This registration produces the **project id** you'll pin — set it up in **step 4**.
- A **CoreStack API key** — REQUIRED (drives cropland sampling). To get one: **register at
  <https://dashboard.core-stack.org/>**, then **generate an API key** from the user/management
  interface. It's a **~41-character** string — copy the **whole** value (a truncated key fails auth).
  Guide: <https://core-stack.org/use-apis/> · API docs: <https://api-doc.core-stack.org/> · questions:
  contact@core-stack.org.
- **Optional:** Ollama + `gemma3:4b`, only if you want the message reworded by the LLM (otherwise a
  deterministic template is used).
- The **Model asset** `ndvi_aikosh_model.zip` — download it from the AIKosh **Model** page:
  <https://aikosh.indiaai.gov.in/web/models/details/ndvi_forecasting_model.html>. This is the **only**
  asset the advisory needs. The AIKosh **Dataset** asset is a *separate* download and is **not**
  required to serve the advisory or run inference — only the train/fine-tune pipeline uses it.

## 1. Get the code
```bash
git clone https://github.com/SoulVisionCreations/ndvi-forecasting-advisory.git
cd ndvi-forecasting-advisory
```

## 2. Fresh Python 3.11 env + install
```bash
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e .
python -c "import ee, uvicorn, torch; print('deps OK')"   # sanity-check the deps landed in THIS venv
```
Keep this venv **active** for every step below (env vars + the server must run in it). On macOS this
installs the CPU/MPS build of torch automatically.

## 3. Unzip the Model bundle into `weights/`
`weights/` is a folder **at the repo root** — i.e. `ndvi-forecasting-advisory/weights/`, right
next to `advisory/`, `ndvi_core/`, and `README.md`. It already exists in the clone (currently just a
`README.md` placeholder); the command below fills it with the 7 model files. **Run it from the repo
root** (the `ndvi-forecasting-advisory/` directory you `cd`'d into in step 1) so the relative
`weights/` path resolves there. Replace `/path/to/ndvi_aikosh_model.zip` with wherever you downloaded
the Model asset (e.g. `~/Downloads/ndvi_aikosh_model.zip`).
```bash
pwd                                    # should end in /ndvi-forecasting-advisory  (the repo root)
python -m zipfile -e /path/to/ndvi_aikosh_model.zip weights/   # -> extracts INTO ./weights/
ls weights/
# tft_temporal_production_ft.pt  standard_scaler_temporal_tft_ft.pkl  label_encoders_temporal_tft_ft.pkl
# train_config.json  mws_static_lookup_UNSCALED.tsv  prithvi_mae.py  config.json
```
The env vars in step 6 (`RUN_DIR=weights`, `MODEL_DIR=weights`, `LOOKUP_CSV=weights/…`) are
**relative to the repo root too**, so start the server (step 7) from the same directory. If you'd
rather keep the model elsewhere, extract to an absolute path (e.g.
`python -m zipfile -e … /data/ndvi_weights/`) and set `RUN_DIR`/`MODEL_DIR`/`LOOKUP_CSV` to that
absolute path instead.
(The pre-existing `weights/README.md` is harmless — the loader reads the 7 files by name.)

## 4. Credentials

**First, register for Earth Engine and get a Cloud project id** — a one-time prerequisite.
`earthengine authenticate` (below) only signs your *user* in; it does **not** create a project or
register you for Earth Engine. Do this first:
1. Have a **Google account**.
2. Go to **<https://earthengine.google.com>** → **Get Started** → sign in. The flow walks you through
   **creating or selecting a Google Cloud project and registering it for Earth Engine** — choose
   **noncommercial / research** (free; commercial needs a paid plan). It enables the **"Earth Engine
   API"** on that project as part of the flow. *(First-time access may need a short approval.)*
3. Note the **project id** it gives you (looks like `ee-yourname` or `my-project-123456`) — you pin it
   in the next step.

This registration is the only real GEE prerequisite, and it's what produces the id you'll use.

**Then authenticate the CLI and pin that project:**
```bash
earthengine authenticate                        # one-time browser OAuth for your USER (saves ~/.config/earthengine/)
earthengine set_project <your-gcp-project-id>   # REQUIRED — pins the project you registered above
export CORESTACK_KEY='<your-corestack-key>'
```
Earth Engine (2023+) runs every request under that Cloud project, and the advisory calls
`ee.Initialize()` with your credentials' **default** project — so pinning it with `set_project` is
required; without it the first request fails with `ee.EEException: … specify a project`. Verify:
`python -c "import ee; ee.Initialize(); print('EE ok')"`.

The **`CORESTACK_KEY`** above comes from CoreStack — three steps:
1. **Register** at <https://dashboard.core-stack.org/>.
2. **Generate an API key** from the user/management interface (guide: <https://core-stack.org/use-apis/>).
3. It's a **~41-character** string — paste the **full** value into `CORESTACK_KEY`. A truncated key
   fails auth (HTTP 502 *"…check your CORESTACK_KEY…"*).

## 5. (Optional) local LLM for the message wording
```bash
ollama pull gemma3:4b      # ~3.3 GB;  skip this and set ADVISORY_SLM=0 to use the template
ollama serve               # if not already running (serves 127.0.0.1:11434)
```

## 6. Environment variables
```bash
export RUN_DIR=weights
export MODEL_DIR=weights
export LOOKUP_CSV=weights/mws_static_lookup_UNSCALED.tsv
export ADVISORY_ENGINE=inprocess          # load the forecaster in-process (no separate service)
export DEVICE=cpu                          # or: cuda
export CACHE_DIR=advisory_cache            # imagery download cache (delete it to force fully-live)
export ADVISORY_FETCH_WORKERS=13           # concurrent satellite-tile downloads per request (see step 9)
# CORESTACK_KEY was exported in step 4.

# message phraser:
export ADVISORY_SLM=0                       # deterministic template (no Ollama needed)
# -- OR, to use Gemma (step 5): --
# export ADVISORY_SLM=1
# export ADVISORY_SLM_MODEL=gemma3:4b
# export ADVISORY_SLM_TEMP=0
# export ADVISORY_SLM_URL=http://localhost:11434/api/generate
```

## 7. Start the service
```bash
python -m uvicorn advisory.serve_advisory:app --host 127.0.0.1 --port 8011
```
Use `python -m uvicorn` (with the venv active) — a bare `uvicorn` can fall through to a global Python
that lacks the deps ("No module named 'ee'"). Wait for
`[serve_advisory] mode=inprocess — forecaster loaded; batched forward`.

## 8. Use it
**In a browser (no curl needed):** open **http://127.0.0.1:8011/docs** (Swagger UI) → expand
**POST /advisory** → **Try it out** → edit the JSON → **Execute**. `/health` is a plain GET;
`/redoc` is a read-only reference.
```bash
curl -s localhost:8011/health
# historical date = a backtest ("verbose": true for the internals):
curl -s localhost:8011/advisory -H 'content-type: application/json' \
  -d '{"lat":20.17,"lon":78.32,"regime":"rain-fed","date":"2018-06-15","verbose":true}'
# a LIVE ~3-month forecast: use today's date (or omit "date"):
curl -s localhost:8011/advisory -H 'content-type: application/json' \
  -d '{"lat":20.17,"lon":78.32,"regime":"rain-fed"}'
```
Response: `{ location, risk_level, confidence, message }` (+ `derived`/`sampling` if verbose).

## 9. Notes
- **First request is slow** (~1–2 min): CPU model forward + ~10 yrs of live data. Later requests reuse `CACHE_DIR`.
- **CPU is fine:** the decision layer is identical to a GPU box; the neural forecast matches to within CPU-vs-GPU float rounding.
- **Date** must be today or earlier (future → rejected). Today/omitted = a live ~3-month forecast; a past date = a backtest.
- **All-live** (no caches): `rm -rf advisory_cache` before running.
- **Rate-limit warnings:** each request fetches up to `ADVISORY_FETCH_WORKERS` satellite tiles **concurrently** (default 13 — the plot + ~12 sampled points). If Earth Engine throttles that burst, the response carries a top-level **`"warnings"`** array (e.g. *"Earth Engine rate-limited N of M … downloads … lower `ADVISORY_FETCH_WORKERS` … and restart"*). The request still completes (throttled tiles are retried with backoff). If you see it often, **lower `ADVISORY_FETCH_WORKERS`** (e.g. `export ADVISORY_FETCH_WORKERS=6`) and **restart the service** — you trade a little latency for a smaller concurrent burst. It also matters at scale: N simultaneous requests = `N × ADVISORY_FETCH_WORKERS` concurrent pulls.
- **regime:** `rain-fed` | `irrigated` | omit for both.
- **Remote host:** the service binds `127.0.0.1`; to browse a remote host's `/docs`, tunnel first: `ssh -L 8011:127.0.0.1:8011 <host>`.

## 10. Troubleshooting & failure modes

**The service won't start (exits immediately, non-zero).** It now **fails fast** on a bad model instead of starting in a degraded state.
- **`STARTUP ABORTED — model artifact path(s) missing or incorrect`** (it lists each one) → a model file/dir is missing. Extract the Model zip into `weights/` and check `RUN_DIR` / `MODEL_DIR` / `LOOKUP_CSV`; start from the repo root.
- **`STARTUP ABORTED — the forecaster failed to load` / `No module named 'ee'`** → you ran a bare `uvicorn` outside the venv (it hit a global Python). Use **`python -m uvicorn`** with the venv active; verify `python -c "import ee, uvicorn, torch"`; re-run `uv pip install -e .` if needed.
- **`ee.EEException: … specify a project`** → run `earthengine set_project <id>` (step 4).

**A request returns an error** (the message is categorized):
- **HTTP 422 "…in the future" / "expected format YYYY-MM-DD"** → `date` must be today-or-earlier, `YYYY-MM-DD`.
- **HTTP 422 "…outside the supported coverage"** → the point is off-land or outside India's CoreStack map; try an agricultural point in India.
- **HTTP 422 "No usable satellite / weather data…"** → usually the date is older than the satellite record — the usable window is **~2017 → today**; try a more recent date, or a nearby point.
- **HTTP 503 "…a data service … is unreachable"** → transient network/DNS to Earth Engine / CoreStack / Open-Meteo; retry.
- **HTTP 502 "…check your CORESTACK_KEY…"** → the key is missing/wrong/**truncated** (it's ~41 chars — paste the whole value), or an Earth Engine auth/project problem.

**A request succeeds but the result looks limited** (these are *not* errors):
- **`risk_level: out-of-coverage`** (a soft message) → no cropland **and** no vegetation near the point (barren / desert / water). Rare.
- **the message notes a nearby-cropland or general-vegetation read** → your plot's tehsil isn't in the (sparse) CoreStack cropland map, so it sampled the **nearest cropland (≤50 km)** or **local vegetation (≤10 km**; confidence capped, "not crop-specific"). Still a valid area-relative read.
- **lower confidence than expected** → confidence is the **worse** of the spatial point-spread and the clear-view (cloud) data quality; a cloudy/monsoon window or a far/vegetation proxy lowers it.
- **a `"warnings"` array in the response** → Earth Engine rate-limited the concurrent tile pulls (still retried). If frequent, lower `ADVISORY_FETCH_WORKERS` (e.g. `6`) and restart — see §9.
- **the message reads like a fixed template** → the optional Gemma phraser is off/unreachable (`ADVISORY_SLM=0`, or Ollama not running). The advisory is still correct — only the wording is templated.

**Credentials / tooling:**
- **GEE auth errors** → re-run `earthengine authenticate` (then `earthengine set_project <id>`).
- **Ollama "connection refused"** (only if `ADVISORY_SLM=1`) → run `ollama serve` + `ollama pull gemma3:4b`, or set `ADVISORY_SLM=0`.
