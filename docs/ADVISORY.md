# Vegetation Outlook advisory (the Use Case)

The **release face** of this project: a farmer-facing advisory built on the NDVI forecaster.
For any `(lat, lon, date)` in India it turns the ~3-month NDVI forecast into a soft,
area-relative **vegetation-outlook opinion** with practical, low-regret stress-mitigation steps.

**Positioning:** an *opinion / indicator* — **not** a drought detector, **not** a source of
truth. No "drought" wording in farmer-facing text. It's meant to be weighed alongside a farmer's
own experience and local extension advice (Krishi Vigyan Kendra / KVK).

---

## What it's actually doing (in plain terms)

For your location it forecasts vegetation greenness (**NDVI**) ~3 months ahead — using a Temporal
Fusion Transformer on a **frozen Prithvi-EO-2.0-300M** encoder **adapted with LoRA (r16/α32, on the
q/k/v projections)**, which encodes the satellite image into the forecaster — then compares that
forecast to **the same spot's own seasonal normal** — the *typical* greenness for this time of year,
built from ~10 years of its own history. The comparison is a **standardized seasonal anomaly** (a
z-score): *how far above/below usual* the forecast is, measured in units of that spot's own
year-to-year variability. So it's a **relative** read — "greener or browner than usual **here, for
this season**" — not an absolute NDVI number. (Because each point is scored against its own history, a
dry-region plot and a lush plot both read "normal" when each is at its own usual.)

It samples **~12 cropland points** around your plot and turns their z-scores into the outlook — three
separate statistics, each answering a different question:

- the **median** of the z's → the area's **severity** — how far from usual (→ **Below / Near / Above
  normal**). This is the *level* of the signal.
- the **spread** (population SD) of the z's → **spatial confidence** — do the points *agree* (tight →
  high confidence) or is the field patchy (wide → low)? This is *not* the level; a patchy field can
  have a moderate median but low confidence.
- your **plot vs the area** (your plot's own z compared to the neighbours) → **attribution** — is a
  below-usual read **area-wide (regional)** or **just your plot (local)**?

The result is a soft **Below / Near / Above normal** opinion for the next ~3 months, with a confidence
and a short message — an indicator to weigh with local experience, **not** a calibrated drought
measurement. For the stage-by-stage mechanics (rule table, season lens, phraser) see *How it works*
below; for the exact thresholds, [`advisory/README.md`](../advisory/README.md).

---

## What each number does — and how they combine

A common intuition is that *your plot's own z* sets how bad things are. It doesn't — **severity
comes only from the surrounding area's median**. Your plot's z is used *relative to the area* to
answer a different question: *whose* problem it is. Three independent axes come out of the numbers:

| axis | answers | computed from | label |
|---|---|---|---|
| **severity** | *how serious* (the level) | `median_z` of the ~12 **area** points | well_below / below / normal / above |
| **confidence** | *how much to trust it* | `sd` (point agreement) **worse-of** clear-view (data quality) | low / medium / high |
| **attribution** | *whose problem it is* | your **plot's own z** vs the area (`plot z`, `pct_below`, `median_z`) | regional / local / none |

- **Area z's → severity + confidence.** The **median** of the neighbours' z's is the *level*
  (severity); their **spread** (SD) is whether they *agree* (spatial confidence). These are the only
  sources of severity and spatial confidence — the plot's own z plays no part in either.
- **Plot z (vs the area) → attribution.** Your own z is compared against the neighbourhood to decide
  whether a below reading is **area-wide (regional)** or **just your plot (local)** — not to set its
  own severity.

**How they combine into the advisory.** The decision (`risk_level`) is a single rule-table lookup
keyed by **four** things at once:

```
(attribution, regime, severity, confidence) → exactly one rule row → risk_level + sentence
      │          │         │          │
  plot-z-vs-area user   area median  area sd + clear-view
```

Then the season lens adds the concrete steps and the phraser words it — neither can change the
`risk_level`.

**Where the plot's own z single-handedly steers the decision.** Because `median_z ≤ −1` forces
`pct_below ≥ 50`, a below/well_below **area** can only be `regional` or `none` — never `local`.
`local` attribution happens only when the **area is normal** but **your plot's own z is low**, and
that is exactly the case that flips the outcome to **`check-your-field`** ("the neighbourhood looks
fine, but *your* plot is lagging — go look"). Everywhere else severity (the area) leads, and your
plot's z only adjusts attribution.

In one line: **area median → how serious · area spread + clear-view → how much to trust it · your
plot vs the area → whose problem it is** — and the rule table folds all four (plus regime) into the
one `risk_level` and message.

---

## Quickstart

```bash
# 1. install (Python 3.11), then sanity-check the deps landed in the venv
uv venv --python 3.11 && source .venv/bin/activate && uv pip install -e .
python -c "import ee, uvicorn, torch; print('deps OK')"

# 2. populate weights/  (download the AIKosh Model asset -> unzip into weights/)

# 3. (optional) local language model for nicer wording — Ollama + Gemma 3 4B
ollama pull gemma3:4b            # ~3.3 GB;  ADVISORY_SLM=0 skips it (deterministic template)

# 4. serve the advisory (loads the forecaster in-process)
export RUN_DIR=weights MODEL_DIR=weights LOOKUP_CSV=weights/mws_static_lookup_UNSCALED.tsv \
       ADVISORY_ENGINE=inprocess CORESTACK_KEY=<key> \
       ADVISORY_SLM=1 ADVISORY_SLM_MODEL=gemma3:4b DEVICE=cuda
python -m uvicorn advisory.serve_advisory:app --host 127.0.0.1 --port 8011

# 5. ask
curl -s localhost:8011/advisory -H 'content-type: application/json' \
  -d '{"lat":20.17,"lon":78.32,"regime":"rain-fed","date":"2018-06-15"}'
#  -> { location, risk_level, confidence, message }
```

**Prefer a browser?** The service ships FastAPI's interactive docs — no need to craft a POST:
open **`http://127.0.0.1:8011/docs`** (Swagger UI), expand **`POST /advisory`**, click **Try it
out**, edit the JSON, and **Execute**. `/health` is a plain GET; `/redoc` is a read-only reference.
The service binds `127.0.0.1`; to reach a remote host's `/docs`, tunnel first
(`ssh -L 8011:127.0.0.1:8011 <host>`).

**GPU optional.** Set `DEVICE=cpu` to run without a GPU — the deterministic decision layer is
identical and the neural forecast matches to within CPU-vs-GPU float rounding (just slower).

**Credentials:** Earth Engine — one-time register at [earthengine.google.com](https://earthengine.google.com)
(*Get Started*; noncommercial = free) to create/register a **Cloud project** and get its **id**, then
`earthengine authenticate` **+ `earthengine set_project <id>`** (`authenticate` only signs the *user*
in — the project step is separate) — plus `CORESTACK_KEY` (**required** for the advisory — it drives
the cropland sampling; register at [dashboard.core-stack.org](https://dashboard.core-stack.org/) →
generate an API key, guide at [core-stack.org/use-apis](https://core-stack.org/use-apis/)). See the
repo README "Configuration & credentials" table and [`LOCAL_SETUP.md`](LOCAL_SETUP.md) step 4.

**Concurrency & rate limits.** Each request pulls up to **`ADVISORY_FETCH_WORKERS`** satellite tiles
concurrently (default `13` — the plot + ~12 sampled points). If Earth Engine throttles that burst, the
response carries a top-level **`"warnings"`** array (the request still completes — throttled tiles retry
with backoff). If it recurs, **lower `ADVISORY_FETCH_WORKERS`** (e.g. `6`) and **restart** the service.

**Full step-by-step** (unzip · uv env · every env var · start · browse the API): [`LOCAL_SETUP.md`](LOCAL_SETUP.md).

---

## How it works (6 stages)

The design principle: **the model only produces NUMBERS; a fixed rule table makes the DECISION;
a season lens supplies the concrete STEPS; the language model only REWORDS (on a leash).** Nothing
downstream can overturn the numbers, and the phraser can't invent facts.

1. **Predictions (numbers)** — sample ~12 cropland points around the plot (CoreStack); forecast
   NDVI ~3 months ahead at each; convert to a **z-score** = how far the forecast greenness is from
   *that spot's own* ~10-yr seasonal normal (robust-std units). Also read the plot's own z, a
   recent-trend series, and a clear-view fraction.
2. **Aggregate (labels)** — median z → **severity**; population SD of the z's → **spatial
   confidence**; clear-view → **data quality**; **overall confidence = worse of the two**; plot z +
   %below + median → **attribution** (regional / local / none).
3. **Lens (concrete steps)** — target date → local **wet/dry** season → regime → the practical
   levers (e.g. conserve moisture / life-saving irrigation vs. secure your water source). Never
   changes the decision.
4. **Rules (the decision)** — a CSV rule table; match exactly **one row** by
   `(attribution, regime, severity, confidence)` → a `risk_level` + a canonical sentence.
5. **Decision (structured, auditable)** — `derived{…}` + the matched row (+ regime split if regime
   is omitted and the two regimes differ) + lens + trend. No prose yet.
6. **Advisory (words)** — a deterministic **template** composes the bulleted message; an optional
   **Gemma 3 4B** rewords it under a hard faithfulness check (`validate()`); any drift → the
   template is served. Optional local-language translation.

For the full component-by-component walk-through and a line-by-line worked example, see
[`advisory/PIPELINE.md`](../advisory/PIPELINE.md); for the module map + pinned thresholds,
[`advisory/README.md`](../advisory/README.md).

---

## The decision layer — summary & how to modify it

**Summary.** The model only produces NUMBERS (per-point z-scores). A small, **deterministic decision
layer** turns them into the labels and the final decision — it owns every threshold and the risk
level, and nothing downstream (season lens, phraser) can overturn it:

1. **Numbers → labels** — `advisory/aggregate.py`, thresholds in `advisory/config.py`:
   `median_z` → **severity**; population **SD** of the z's → **spatial confidence**; clear-view
   fraction → **data quality**; **overall confidence = worse of the two**; plot z vs the area
   (`median_z`, `%below`) → **attribution** (regional / local / none).
2. **Labels → decision** — `advisory/rule_engine.py` + the rule table
   `advisory/rules/uc1_advisory_rules.csv`: match **exactly one** row by
   `(attribution, regime, severity, confidence)` → a **`risk_level`** + a canonical sentence
   (17 rows cover every reachable combination).
3. **Concrete steps** — `advisory/season_lens.py` (wet/dry + regime levers; never change the decision).
4. **Wording** — `advisory/phraser.py`: a deterministic template (default) or a leashed Gemma reword
   under a faithfulness check; never invents facts.

**How to modify it** (all data/config — **no model retrain needed**):

- **Retune thresholds** → `advisory/config.py`: severity bins (`SEVERITY_WELL_BELOW=-2.0`,
  `SEVERITY_BELOW=-1.0`, `SEVERITY_ABOVE=1.0`), spatial-confidence SD bands (`SPATIAL_SD_HIGH=0.5`,
  `SPATIAL_SD_LOW=1.0`), attribution cutoffs (`ATTR_Z=-1.0`, `ATTR_PCT_BELOW=50`), `MIN_POINTS=12`,
  vegetation-fallback cap (`VEG_FALLBACK_CONF_CAP="medium"`), etc.
- **Change the decision or its wording** → the rule table `advisory/rules/uc1_advisory_rules.csv`
  (columns `attribution,regime,severity,confidence,risk_level,action_tier,active_advisory,escalate_if,driver_check`):
  add/adjust a row to remap a `(attribution, regime, severity, confidence)` combination to a different
  `risk_level` or sentence.
- **Change the concrete steps** → `advisory/season_lens.py` (season/regime levers).
- **Guardrails stay enforced regardless:** `FORBIDDEN_WORDS` (no "drought"/"famine"/… in output) + the
  phraser's `validate()` faithfulness check + the contradiction audit. The unit tests in
  `advisory/tests/test_core.py` pin the threshold→label→row behaviour — **run them after any change.**

Because the whole layer is pure functions + one CSV + one config file, changes are auditable and
testable **without touching the model weights**. **Full step-by-step customization guide (which file,
what format, examples): [`advisory/CUSTOMIZE.md`](../advisory/CUSTOMIZE.md).** Module map + all pinned
values: [`advisory/README.md`](../advisory/README.md).

---

## Worked example — MURLI, `2018-06-15` (regime omitted)

Request `{"lat":20.17,"lon":78.32,"date":"2018-06-15"}` → target **2018-09-08** (~3 months out).
CoreStack names the spot **MURLI, Ghatanji, Yavatmal**. The 12 sampled points read strongly below
their seasonal normal (median z ≈ −2.5, all points below, good clear-view) →

- **severity** = well_below · **attribution** = regional · **confidence** = medium
- **risk_level** = `below-usual`; because regime was omitted and the rain-fed vs. irrigated advice
  differs, the message carries **both** branches (rainfall levers vs. water-source levers).

Message (abridged): *"MURLI — vegetation may run **well below** its usual across your area over the
next ~3 months (medium confidence). For now low-regret stress-mitigation; be ready to escalate if
it firms up… conserve soil moisture · a life-saving irrigation at the critical stage · a foliar
potassium spray…"*

---

## Coverage, dates & confidence — what to expect

- **Usable dates: ~2017 → today.** Today (or omitting `date`) → a live ~3-month forecast; a past date →
  a backtest. Future dates are rejected; dates before the satellite record (~2016) return *"no usable data"*.
- **Coverage / fallback.** CoreStack cropland is sparse. If your plot's tehsil has cropland you get an
  in-village read; otherwise the advisory falls back — **nearest cropland (≤50 km)**, then **local
  vegetation (≤10 km**, flagged *not crop-specific*, confidence capped) — and only when there's **neither**
  does it return a graceful **`out-of-coverage`** message. Off-land / outside-India points are rejected (422).
- **Confidence = the worse of two axes:** the spatial point-spread and the clear-view (cloud) data quality.
  A cloudy/monsoon window or a distant/vegetation proxy reads as **lower confidence**.
- **`warnings` field.** On Earth Engine throttling, the response adds a top-level **`warnings`** array (the
  request still completes — tiles retry with backoff); lower `ADVISORY_FETCH_WORKERS` if it recurs.
- **Errors are categorized:** **422** (your input/date, or an off-coverage point), **503** (an upstream is
  unreachable — retry), **502** (rejected — check `CORESTACK_KEY` / Earth Engine auth). Full symptom → fix
  list: [`LOCAL_SETUP.md` §10](LOCAL_SETUP.md).

## Response fields (what the API returns)

Every request returns four fields; `"verbose": true` adds the internals. The **decision** is
`risk_level`; `severity` is just *one input* to it (see the note at the end).

**Always returned**

| field | meaning |
|---|---|
| `location.lat` / `.lon` | the point you asked about (echoed back) |
| `location.village` / `.district` / `.tehsil` | administrative names (CoreStack) for the plot |
| `location.target_date` | the date being forecast — ~3 months (7 fortnights) ahead |
| `location.regime` | `rain-fed` / `irrigated` — **omitted** when you didn't specify one |
| **`risk_level`** | **the decision** — `normal` · `watch` · `below-usual` · `check-your-field` (rule-engine output) |
| **`confidence`** | surfaced confidence — `low` / `medium` / `high` (the **worse** of spatial agreement and data quality) |
| `message` | the farmer-facing bulleted text (deterministic template, or a Gemma reword) |

**Added when `verbose: true`** — `derived{…}` turns the numbers into labels:

| field | meaning |
|---|---|
| `derived.median_z` | median of the ~12 sampled points' z-scores = the **area anomaly** (how far from usual) |
| `derived.severity` | the area **level** from `median_z` alone: `well_below` · `below` · `normal` · `above` |
| `derived.sd` | population SD of the point z's = how much the points **spread / disagree** |
| `derived.point_agreement` | spatial confidence from `sd`: `high` (tight) · `medium` · `low` (patchy) |
| `derived.data_quality` | clear-view (cloud) axis: `good` · `fair` · `poor` |
| `derived.pct_below` | % of sampled points below their own normal |
| `derived.attribution` | `regional` (area-wide) · `local` (just your plot) · `none` |
| `derived.n_points` / `.small_sample` | points sampled (target 12) / flag if fewer |
| `trend_context` | recent-trend context: `recovering` · `declining` · `unknown` |
| `sampling{…}` | where the points came from: `state/district/tehsil/village`, `n_points`, `proxy` (nearest-cropland fallback), `veg_fallback` (vegetation fallback) |
| `lens{…}` | concrete steps when the reading warrants them: `driver`, `escalate_if`, `levers[]`, `resource`, `season` (wet/dry) — **only on a below-usual reading** |
| `lens_irrigated{…}` | the irrigated-regime steps — **only when `regime` was omitted and the two regimes differ** |
| `warnings[]` | operational notices (e.g. Earth Engine rate-limiting) — only when something occurred |

> **`severity` vs `risk_level`.** `severity` is the raw *level* from `median_z` alone; `risk_level`
> is the **decision**, chosen by matching **all four** of `(attribution, regime, severity,
> confidence)` to one rule row. So the same `severity` can produce different `risk_level`s — e.g.
> `below` severity with `low` confidence → `watch`, but `well_below` with `medium` confidence +
> `regional` attribution → `below-usual`. Severity is an *ingredient*; risk_level is the *verdict*.

## Guarantees & limitations

- The decision layer is **contradiction-free across all reachable states** (audit harness), and the
  served wording can never contradict the decision (`validate()` guards; template fallback).
- **Field values are deterministic** (re-run → exact match). The **message** is Gemma at
  temperature 0 (reproducible in practice, guard-checked) — it is the one non-deterministic field.
- Known simplifications (marked in code): the clear-view fraction uses a season prior until the
  forecaster exposes a true valid-observation count; recent-trend is sometimes `unknown` at serve
  time; local wet/dry uses a coarse monsoon-belt month rule until per-location rainfall climatology
  is wired.
- It is a ~3-month **vegetation** outlook, not a calibrated drought product; treat it as one input.
