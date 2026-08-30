# advisory — Vegetation Outlook layer (UC1)

A farmer-facing **advisory layer** on top of the NDVI forecaster. It turns a ~3-month
NDVI forecast into a soft, area-relative **vegetation-outlook opinion** with
stress-mitigation levers.

**Positioning:** an *opinion / indicator* — **not** a drought detector, **not** a
source of truth. No "drought" wording in farmer-facing text.

## Separate service, forecaster untouched
This package is **additive**: it ships as its **own** FastAPI service
(`serve_advisory.py`, on a separate port) and never edits the forecaster's own service
(`inference/serve_api.py`). It has two backends (env `ADVISORY_ENGINE`):

- **`inprocess`** (default, and what the deployment runs) — loads the forecaster
  **once** in-process (`advisory.engine.AdvisoryEngine`) and runs the plot + sampled
  points as a single **batched** forward (~1.5–2 min end-to-end). Satellite tiles for the
  ~13 points download **concurrently** — env **`ADVISORY_FETCH_WORKERS`** (default 13). If
  Earth Engine throttles the burst, the response adds a top-level **`"warnings"`** array
  (throttled tiles retry with backoff, so the request still completes); lower
  `ADVISORY_FETCH_WORKERS` and restart if it recurs.
- **`http`** (fallback) — the black-box path: serial calls to the forecaster's
  `/forecast` endpoint (~7–8 min). Used only if the in-process engine fails to load.

Either way the forecaster's weights and its `/forecast` service are unchanged — the
advisory only adds a layer on top.

## Three stages
```
MODEL        the forecaster emits FACTS: z = how far below/above the area's OWN normal
RULE ENGINE  deterministic — owns every number + decision (severity, confidence,
             data-quality, attribution, risk row). Auditable.
PHRASER      template (default) OR a leashed Gemma 3 that ONLY rewords the matched row
             (+ optional local-language translation). Invents nothing; template fallback.
```

## Modules
| file | role |
|---|---|
| `config.py` | every pinned threshold + endpoint in one place |
| `forecast_client.py` | thin HTTP client for the black-box `/forecast` |
| `sampling.py` | Step 1–2: locate village + sample ≥12 cropland points (CoreStack) |
| `aggregate.py` | median z, %below, **spatial confidence = population SD**, attribution |
| `data_quality.py` | **separate** clear-view axis; overall = worse-of-two; poor → gate |
| `season_lens.py` | target date → **local wet/dry** → driver / stress-mitigation levers |
| `rule_engine.py` | facts → the one matched row + lens + gates (the brain) |
| `phraser.py` | template + leashed Gemma 3 + faithfulness validation |
| `advisory.py` | orchestrator (`build_advisory`, `build_advisory_from_facts`) |
| `serve_advisory.py` | separate FastAPI `/advisory` service |

**Changing the advisory?** See **[`CUSTOMIZE.md`](CUSTOMIZE.md)** — which file to edit (config
thresholds · the rules CSV · the season-lens levers · the phraser template) and in what format, with
examples, all **without retraining the model**.

## Pinned definitions (match use_case_ndvi.txt)
- **severity** (area median z): `≤−2 well_below | −2..−1 below | −1..1 normal | ≥1 above`
- **spatial confidence** = **population SD** of per-point z: `<0.5 high | 0.5–1.0 medium | ≥1.0 low`
- **data quality** = clear-view fraction: `≥0.75 good | 0.5–0.75 fair | <0.5 poor` — a
  **separate** axis; **overall confidence = worse of the two**; **poor → watch/low**.
- **attribution**: `plot_z≤−1 & %below≥50 & median_z≤−1 → regional | plot_z≤−1 → local | else none`
- **sampling**: village polygon, ≤5 km buffer, **min 12 points**.
- **recent trend**: shown as **context only** — never changes the row.

## Run
```bash
# from the repo root
python -m advisory.tests.test_core        # 9 pure-logic tests, no network
python -m advisory.examples.advisory_demo # offline demo (wet/dry lens across locations)

# live (needs /forecast on :8001 + CORESTACK_KEY)
python -m uvicorn advisory.serve_advisory:app --port 8011
# POST /advisory {"lat":20.17,"lon":78.32,"regime":"rain-fed","date":"2018-06-15"}
```

## SLM (Gemma 3 4B) — the phraser only
The phraser can OPTIONALLY use a small local model to reword the matched row (off by default in
code; the deployment sets `ADVISORY_SLM=1`). Uses **`gemma3:4b`** (Gemma 3, 4B, ~3.3 GB) locally
via Ollama (`ADVISORY_SLM_MODEL`, `ADVISORY_SLM_URL`; temperature 0). Output is
**faithfulness-validated** (no forbidden words, no invented numbers, magnitude preserved); on any
failure it **falls back to the deterministic template**. `ADVISORY_SLM=0` skips it entirely.

## Scaffold status — wired vs stubbed
Working now: sampling, aggregation, data-quality math + gate, season lens, rule engine,
template phraser, orchestrator, service, tests.
Stubbed / follow-ups (clearly marked in code):
- **clear-view fraction** uses a season prior + anchor-staleness proxy until `/forecast`
  exposes a true valid-observation count.
- **recent trend** is often `unknown` at serve time (the forecaster computes the window
  but doesn't return it yet).
- **local wet/dry** uses a coarse SW/NE-belt month rule until per-location rainfall
  climatology is wired (`local_wet_dry(..., precip_normals=...)` already accepts it).
- **Gemma 3n phraser** interface is complete but disabled by default.
