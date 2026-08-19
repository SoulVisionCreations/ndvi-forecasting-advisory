# UC1 Vegetation Outlook — end-to-end pipeline

How a request becomes an advisory, and which component owns what.

**Design principle:** the model only produces NUMBERS; a fixed rule table makes the
DECISION; the season lens supplies the concrete STEPS; the language model only REWORDS
(on a leash). Nothing downstream can overturn the numbers, and the phraser can't invent
facts.

```
 request (lat, lon, date, [regime])
        │
        ▼
 ┌──────────────────┐   engine.py / sampling.py / forecast
 │ 1. PREDICTIONS   │   ~12 cropland points → forecast NDVI (~3 mo) → per-point z-scores
 │    (numbers)     │   + own-plot z, recent-trend series, clear-view fraction
 └──────────────────┘
        │  point z's, plot z, clear-view, trend
        ▼
 ┌──────────────────┐   aggregate.py / data_quality.py  (pure threshold lookups; config.py)
 │ 2. AGGREGATE     │   median_z→severity · sd→point_agreement · plot/pct/median→attribution
 │    (labels)      │   clear-view→data_quality · confidence = worse(agreement, data)
 └──────────────────┘
        │  labels: severity, attribution, confidence, point_agreement, data_quality, trend
        ├───────────────────────────────┐
        ▼                               ▼
 ┌──────────────────┐            ┌──────────────────┐   season_lens.py
 │ 4. RULES         │            │ 3. LENS          │   season = wet/dry (region RULE)
 │ rule_engine.py   │            │ (concrete steps) │   + regime → driver / levers /
 │ uc1_..._rules.csv│            │                  │     escalate_if / resource
 │ match ONE row by │            └──────────────────┘   (NEVER changes the decision)
 │ (attribution,    │                     │
 │  regime,severity,│◄────────────────────┘  lens attaches to the decision
 │  confidence)     │
 └──────────────────┘
        │  risk_level + canonical sentence (+ regime split if regime omitted & they differ)
        ▼
 ┌──────────────────┐   the auditable answer, still structured (no prose yet)
 │ 5. DECISION      │   derived{...} + matched row (+ matched_irrigated) + lens (+ lens_irrigated) + trend
 └──────────────────┘
        │
        ▼
 ┌──────────────────┐   phraser.py
 │ 6. ADVISORY      │   template composes bullets → Gemma 3 4B rewords (guard-checked;
 │    (words)       │   drift → fall back to template)
 └──────────────────┘
        │
        ▼
   {location, risk_level, confidence, message, [verbose: derived, lens, trend, sampling]}
```

---

## Worked example — MURLI, `2018-06-15`, regime omitted

Request: `{"lat":20.17, "lon":78.32, "date":"2018-06-15"}`

### 1. PREDICTIONS — `engine.py`, `sampling.py`
1. Snap the date to the model grid + lead → **target 2018-09-08** (~3 months out).
2. Sample **~12 cropland points** around the plot (buffer outward if fewer than `MIN_POINTS=12`).
   CoreStack names the spot → **village MURLI, Ghatanji, Yavatmal**.
3. Forecast NDVI at each point, then convert to a **z-score** = how far the forecast greenness
   is from *that spot's own* ~10-yr seasonal-normal median, in robust-std units (1.4826×MAD,
   floored 0.05). `z < 0` = below its usual.
4. Also read: the farmer's **own plot z**, a **recent-trend** series, a **clear-view fraction**
   (could the satellite actually see the ground?).

Raw numbers: 12 negative point z's · plot z ≈ −2.9 · clear-view ≈ good · trend ↑.

### 2. AGGREGATE — `aggregate.py`, `data_quality.py` (edges in `config.py`)

| computed | from | value → label |
|---|---|---|
| `median_z = -2.485` | median of the 12 z's | ≤ −2 → **severity = well_below** |
| `sd = 0.814` | population SD of the z's | 0.5–1.0 → **point_agreement = medium** |
| `pct_below = 100` | % of points with z ≤ −1 | — |
| `attribution` | plot z ≤ −1 AND %below ≥ 50 AND median ≤ −1 | all true → **regional** (area-wide) |
| `data_quality = good` | clear-view ≥ 0.75 | good |
| **confidence = medium** | *worse of* (point_agreement, data_quality) | worse(medium, good) = **medium** |
| `trend = recovering` | recent obs vs normal, trending up | context only |
| `small_sample = false` | n_points < 12 | 12 points → false |

Reachability note: `median_z ≤ −1` forces `pct_below ≥ 50`, so `below/well_below` can only be
`regional` or `none` (never `local`); `local` only pairs with `normal/above`.

### 3. LENS — `season_lens.py` (concrete steps; does NOT decide anything)
- `local_wet_dry(20.17, 78.32, month=9)`: MURLI is outside the NE-belt box → SW monsoon →
  Sep ∈ {6,7,8,9} → **season = wet**. *(Region RULE by default; a per-location rainfall-
  climatology path exists but is intentionally unwired.)*
- `apply(season="wet", regime=…)` picks a hand-authored row:
  - **rain-fed** → "wet" row: driver *rainfall deficit*; levers *conserve moisture / life-saving
    irrigation / foliar potassium*; watch *the rains underperform*.
  - **irrigated** → always the water-source row: levers *micro-irrigate / mulch / kaolin*; watch
    *your water source draws down* (an irrigated farmer manages the SOURCE, not rainfall).

### 4. RULES — `rule_engine.py`, `rules/uc1_advisory_rules.csv`
The CSV is a spreadsheet; each row is keyed by **(attribution, regime, severity, confidence)**.
Regime was **omitted**, so the engine evaluates BOTH regimes and merges:

- `(regional, rain-fed, well_below, medium)` → **risk_level = below-usual**, tier *low_regret+prepare*,
  sentence *"Vegetation may run WELL BELOW its usual… (medium confidence)… low-regret
  stress-mitigation; be ready to escalate…"*
- `(regional, irrigated, well_below, medium)` → **risk_level = below-usual**, sentence
  *"May run well below… (irrigated)… Check + secure your water source…"*

The two sentences differ → `_merge_regimes` sets **`regime_split = True`** and keeps both
(each with its own lens). If they'd matched (e.g. a *normal* reading, whose rule is
`regime="any"`), it collapses to one branch.

> The row is chosen by **all four keys together** — severity alone does not decide it. Lower
> confidence or a different attribution lands on a different row (e.g. a soft "watch" instead
> of "below-usual"). If nothing matches, `_synth_watch` degrades to a neutral watch (with the
> `none+below` / `none+well_below` rows added, this is now unreachable for real aggregator output).

### 5. DECISION (structured, auditable — no prose yet)
```
risk_level = below-usual · confidence = medium · severity = well_below · attribution = regional
matched (rain-fed row) + matched_irrigated (irrigated row)
lens (rain-fed levers) + lens_irrigated (water-source levers)
trend_context = recovering · point_agreement = medium · data_quality = good
```

### 6. ADVISORY — `phraser.py`
1. **Template** (deterministic floor): `levers_shown` is true (severity ∈ {below,well_below}
   AND attribution = regional), so it prints the levers; `regime_split` → an **"If rain-fed"**
   branch (rainfall levers) and an **"If irrigated"** branch (water-source levers); adds the
   trend line, **reconciled** ("recovering… *even though* the outlook still leans well below").
   Also appends caveats when relevant (fair/poor data, unknown clear-view, small sample).
2. **Gemma 3 4B** rewords it into warm language, but `validate()` is a hard leash: can't soften
   "well below" → "a little below", can't invent an "if rain-fed/if irrigated" branch the rules
   didn't choose, must keep the confidence word / small-sample & data caveats, watch-items stay
   "may", the past trend is never projected forward. On any drift → serve the template.

Final `message` = the two-branch, levers-and-trend bulleted advisory.

---

## Who owns what (the mental model)

| component | owns | file(s) |
|---|---|---|
| **Model** | the numbers (per-point z-scores) — the only source of "how green" | forecaster + `engine.py` |
| **Aggregate** | numbers → labels (severity / point_agreement / attribution / confidence) | `aggregate.py`, `data_quality.py`, `config.py` |
| **Rules** | the **decision**: risk_level + canonical sentence, from the label combination. Season-neutral, auditable, exactly one row | `rule_engine.py`, `rules/uc1_advisory_rules.csv` |
| **Lens** | the **concrete steps** that fit the local season & regime. Never changes the decision | `season_lens.py` |
| **Phraser** | **wording only**, on a validate() leash; template is the guaranteed fallback | `phraser.py` |

**Rules vs. Lens (they're grouped but distinct):** rules = *what's happening & how serious*;
lens = *which practical levers fit*. The lens can swap advice (rainfall vs. water-source) without
ever moving the risk_level.

## How other cases flow through the same pipeline
- **Normal / all-clear** (`none`, `normal`): rules match a `regime="any"` row → no split, no
  levers → `levers_shown` false → the lens is **not surfaced** in verbose. Only a data caveat
  (if any) is added.
- **Check-your-field** (`local`): the AREA is normal but the farmer's own plot z ≤ −1 →
  attribution flips the whole outcome to "check-your-field"; no area-wide levers, lens suppressed,
  and the phraser guard forbids inventing regime branches.
- **Area below but your plot fine** (`none` + below/well_below): the `none+below` /
  `none+well_below` rows state the area severity while noting your plot reads normal.

## Guarantees (verified)
- Decision layer is **contradiction-free across all reachable states** (audit harness).
- The served wording can never contradict the decision (validate() guards; template fallback).
- FIELD values are deterministic (re-run → exact match); the MESSAGE is Gemma at temperature 0
  (reproducible), guard-checked.
