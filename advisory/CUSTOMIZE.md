# Customizing the Vegetation Outlook advisory

How to change the advisory's behaviour **without retraining the model**. The forecaster only emits
NUMBERS (per-point z-scores); everything a user typically wants to tune — the thresholds, the
decision, the wording, the on-the-ground steps — lives in a handful of **config files, one CSV, and
small pure-Python functions** in `advisory/`.

> **Golden rule:** the **rule engine owns every decision**; the season lens only adds steps; the
> phraser only rewords (on a leash). So a change is almost always a *data/config* edit, not a code
> rewrite — and the behaviour is pinned by `advisory/tests/test_core.py`. **Run the tests after any
> change** (command at the bottom).

## Quick index — "I want to change X → edit Y"

| I want to change… | Edit | Format |
|---|---|---|
| **Severity / confidence / attribution thresholds** | `config.py` | Python constants |
| **The decision (`risk_level`) or its canonical sentence** for a situation | `rules/uc1_advisory_rules.csv` | CSV row |
| **The concrete steps / stress-mitigation levers** (+ the wet/dry driver) | `season_lens.py` | Python dict (`_LENS`) + belt rule |
| **The message template / phrasing** | `phraser.py` | Python (`render_template`) |
| **Banned words** in any output | `config.py` (`FORBIDDEN_WORDS`) | tuple of strings |
| **Cropland sampling** (radii, min points, vegetation fallback) | `config.py` (+ `sampling.py`) | Python constants |
| **How the numbers become labels** (the aggregation maths) | `aggregate.py` | Python functions |
| **The phraser LLM** (which model, on/off, temperature) | env vars | `ADVISORY_SLM*` |
| **Device / concurrency / cache** | env vars | `DEVICE`, `ADVISORY_FETCH_WORKERS`, `CACHE_DIR` |

All paths are under `advisory/`. To change the **forecast itself** (the numbers — a different
horizon, inputs, or accuracy) you retrain the model instead — see [`../docs/TRAINING.md`](../docs/TRAINING.md).

---

## 1. Thresholds — `advisory/config.py`
The single home for every pinned number. Edit a constant, restart the service. The main ones:

```python
# severity from the AREA median z (advisory/aggregate.py::severity)
SEVERITY_WELL_BELOW = -2.0   # median_z <= this  -> "well_below"
SEVERITY_BELOW      = -1.0   # <= this -> "below" ; between BELOW and ABOVE -> "normal"
SEVERITY_ABOVE      =  1.0   # >= this -> "above"

# spatial confidence from the population SD of the point z's (how much they AGREE)
SPATIAL_SD_HIGH = 0.5        # sd < this -> "high"
SPATIAL_SD_LOW  = 1.0        # sd < this -> "medium" ; else "low"

# data-quality (clear-view) axis ; overall confidence = WORSE of spatial vs data-quality
DATA_GOOD = 0.75 ; DATA_FAIR = 0.50     # good / fair / poor -> high / medium / low

# attribution: is a below-normal read AREA-WIDE (regional) or just the plot (local)?
ATTR_Z         = -1.0        # plot_z / median_z must be <= this ...
ATTR_PCT_BELOW = 50          # ... AND >= this % of area points below normal -> "regional"

# sampling
MIN_POINTS            = 12          # target cropland points ; fewer -> small_sample flag
BUFFER_KM_MAX         = 5.0         # widen the in-village search up to this if short
VEG_FALLBACK_CONF_CAP = "medium"   # a non-cropland (vegetation) read can never exceed this
```
Also here: the CoreStack endpoints, the `FORBIDDEN_WORDS` tuple, and the SLM env defaults.
*Example:* to make "below" trigger sooner, raise `SEVERITY_BELOW` from `-1.0` to `-0.7`.

## 2. The decision table — `advisory/rules/uc1_advisory_rules.csv`
The rule engine matches **exactly one** row by the 4-tuple `(attribution, regime, severity,
confidence)` and returns that row's `risk_level` + sentence. To change *what decision* a situation
gets — or *what it says* — edit/add a row. Columns:

```
attribution,regime,severity,confidence,risk_level,action_tier,active_advisory,escalate_if,driver_check
```
| column | values |
|---|---|
| `attribution` | `none` · `local` · `regional` (or `any`) |
| `regime` | `rain-fed` · `irrigated` (or `any`) |
| `severity` | `well_below` · `below` · `normal` · `above` (or `any`) |
| `confidence` | `low` · `medium` · `high` (or `any`) |
| `risk_level` | the decision the API returns (e.g. `normal`, `watch`, `below-usual`, `check-your-field`) |
| `active_advisory` | the **canonical sentence** — the phraser rewords THIS and may not go beyond it |
| `escalate_if`, `driver_check` | optional hints (may be blank) |

Rules of thumb: keep the table **exhaustive** (every reachable combo must match a row — `any`
wildcards help), keep sentences within the `FORBIDDEN_WORDS` policy, and re-run the tests +
contradiction audit. ~17 rows today; comment lines start with `#`.
*Example:* to soften `regional + below + medium` from `below-usual` to `watch`, change that row's
`risk_level` and sentence.

## 3. The steps / levers — `advisory/season_lens.py`
Turns the target date + location into the **wet/dry** state, then picks the concrete
stress-mitigation levers. Two tunable pieces:
- **The wet/dry rule** — `local_wet_dry(...)`: a coarse belt rule (`_NE_BELT` bounding box,
  `_SW_WET_MONTHS = {6,7,8,9}`, `_NE_WET_MONTHS = {10,11,12}`). Adjust the belt/months, or wire real
  per-location rainfall climatology (it's a marked stub).
- **The levers** — the `_LENS` dict, keyed by season/state; each entry has `driver`, `escalate_if`,
  `levers` (the bulleted steps), `resource`. Edit the `levers` lists to change the advice.
  `apply(season, regime=…)` picks the key (e.g. `irrigated` → the water-source levers).

The lens **never changes the decision** — it only supplies steps, and is surfaced only when the
message actually carries levers.

## 4. The wording — `advisory/phraser.py`
- **`render_template(advisory)`** — the deterministic message (the floor; always works). Edit this to
  change the default phrasing/structure.
- **`FORBIDDEN_WORDS`** (in `config.py`) + **`validate(text, advisory)`** — the guardrails: an LLM
  reword may not add numbers/words the facts don't support, or any forbidden term; on any drift it
  **falls back to the template**. Tighten/loosen `validate()` to change what the LLM may say.
- **The LLM is env-controlled:** `ADVISORY_SLM` (1/0), `ADVISORY_SLM_MODEL` (e.g. `gemma3:4b`),
  `ADVISORY_SLM_TEMP` (`0` = deterministic), `ADVISORY_SLM_URL`. Set `ADVISORY_SLM=0` for a pure,
  fully-deterministic template message.

## 5. How numbers → labels — `advisory/aggregate.py`
If you need to change the *maths* (not just the thresholds): `severity(median_z)`,
`spatial_confidence(sd)`, `attribution(plot_z, pct_below, median_z)`, and `summarise(point_zs,
plot_z)` are small pure functions. (The thresholds they read come from `config.py` — prefer editing
those first.)

## 6. Sampling & coverage — `advisory/config.py` + `advisory/sampling.py`
Radii and fallback behaviour live as constants: `MIN_POINTS`, `BUFFER_KM_MAX`, `_MAX_PROXY_KM`
(nearest-cropland ≤ 50 km), `VEG_SEARCH_KM` (vegetation fallback ≤ 10 km), `VEG_DIST_PENALTY_KM`
(handicap so near cropland beats far vegetation), `VEG_FALLBACK_CONF_CAP`, `WORLDCOVER_VEG_CLASSES`.
Change these to widen/narrow coverage or the cropland-vs-vegetation preference.

---

## After ANY change
1. **Run the tests** (fast, no network — pins threshold→label→row behaviour + the guardrails):
   ```bash
   python -c "import advisory.tests.test_core as t; t._run_all()"
   ```
2. **Restart the service** (`:8011`) — `config.py` / the rules CSV / the lens / the phraser are read
   at load or first use, so a running server won't pick up edits until it restarts.
3. **For rule-table edits**, confirm the decision layer stays **contradiction-free**: no reachable
   `(attribution, regime, severity, confidence)` should resolve to conflicting rows.

## What you can NOT change here
The **forecast numbers** (the z-scores) come from the trained Prithvi + TFT model. To change *those* —
a different horizon, different inputs, or better accuracy — you **retrain**: see
[`../docs/TRAINING.md`](../docs/TRAINING.md) and [`../MODEL_CARD.md`](../MODEL_CARD.md). Module map:
[`README.md`](README.md); pipeline walk-through: [`PIPELINE.md`](PIPELINE.md).
