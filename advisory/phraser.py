"""
advisory.phraser — Stage 3: turn the matched rule row into the farmer-facing message.

Two backends:
  * render_template(advisory)  — deterministic, always works, ALWAYS the fallback.
  * phrase_with_slm(...)       — a LEASHED Gemma 3n call that ONLY rewords the row and
                                 (optionally) translates it. It invents nothing; if the
                                 output fails validation we fall back to the template.

The rule engine owns every fact. The SLM is a wordsmith/translator on a short leash.
"""
import json
import re
import urllib.request

from . import config as C


# ---------------------------------------------------------------------------
# v0 — deterministic template (the floor)
# ---------------------------------------------------------------------------
def levers_shown(advisory):
    """The single source of truth for 'does this advisory carry stress-mitigation levers?'
    Levers (and therefore the season lens) are only relevant when the AREA reads below its
    usual AND that below-ness is area-wide (regional). Used by BOTH the phraser (to add the
    lever bullets) and the serve layer (to decide whether to surface the lens block), so the
    message and the JSON can never disagree about whether a lens applies."""
    d = advisory["derived"]
    return d.get("severity") in ("below", "well_below") and d.get("attribution") == "regional"


def render_template(advisory):
    """Compose a soft, honest message as bullet POINTS (a title line + one point per
    line). Softness comes from the HONEST parts (confidence, data caveat, low-regret
    levers, 'one opinion') — never from hiding the magnitude."""
    d = advisory["derived"]
    row = advisory["matched"]
    lens = advisory["lens"]
    loc = advisory["input"]
    where = loc.get("village") or "your area"
    B, SUB = "• ", "    – "        # "• " bullet, "  – " sub-bullet
    show_levers = levers_shown(advisory)

    def _levers(l):
        """The SPECIFIC stress-mitigation steps + the watch trigger for a given lens."""
        out = [f"{B}Low-regret steps to consider:"]
        out += [f"{SUB}{lv}" for lv in l["levers"]]
        out.append(f"{B}Watch: {l['escalate_if']}.")
        return out

    # bullet 1 = the canonical rule sentence (it already carries the outlook, confidence,
    # "one opinion", "confirm with ..."). The bullets below add only what it lacks.
    lines = [f"{where} — vegetation outlook, next ~3 months:"]
    if advisory.get("regime_split"):        # regime unspecified & the two regimes diverge
        # each branch carries its OWN levers: rain-fed = rainfall/moisture; irrigated = water source
        lines.append(f"{B}If your field is RAIN-FED: {row['active_advisory']}")
        if show_levers:
            lines += _levers(lens)
        lines.append(f"{B}If your field is IRRIGATED: {advisory['matched_irrigated']['active_advisory']}")
        if show_levers:
            lines += _levers(advisory.get("lens_irrigated", lens))
    else:
        lines.append(f"{B}{row['active_advisory']}")
        if show_levers:
            lines += _levers(lens)          # lens is already regime-appropriate

    # data-quality caveat (additive — the rule sentence never mentions data quality)
    if d.get("data_poor_gate"):
        lines.append(f"{B}Note: limited clear satellite views this window — treat this "
                     "as a low-confidence read.")
    elif d["data_quality"] == "fair":
        lines.append(f"{B}Note: a fair-quality read — some cloud limited the clear "
                     "satellite views.")
    elif d["data_quality"] == "unknown":
        lines.append(f"{B}Note: clear satellite-view quality for this window could not be "
                     "assessed — read this greenness outlook with extra caution.")

    # small-sample caveat — few fields could be sampled, so temper the read no matter how
    # tightly they agree (point_agreement / data_quality don't account for sample SIZE).
    if d.get("small_sample"):
        lines.append(f"{B}Note: only a few fields could be sampled here (a small sample) — "
                     "read this with extra caution.")

    # recent trend, as CONTEXT only (only the actionable directions surface a line). When the
    # trend points OPPOSITE the ~3-month outlook we say so explicitly, so the past-context line
    # and the forecast never read as a bare contradiction.
    trend = advisory.get("trend_context")
    if trend in ("recovering", "declining"):
        sev = d["severity"]
        _phrase = {"recovering": "recovering toward normal",
                   "declining": "declining lately"}[trend]
        opposes = ((trend == "recovering" and sev in ("below", "well_below"))
                   or (trend == "declining" and sev in ("normal", "above")))
        if opposes:
            _outlook = {"well_below": "still leans well below", "below": "still leans below",
                        "normal": "is about normal", "above": "is favourable"}[sev]
            lines.append(f"{B}Recent trend (context): the area has been {_phrase} recently, "
                         f"even though the ~3-month outlook {_outlook} — weigh both.")
        else:
            lines.append(f"{B}Recent trend (context): the area has been {_phrase} — "
                         "weigh that in.")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# faithfulness validation — the SLM output must not drift from the facts
# ---------------------------------------------------------------------------
def _allowed_numbers(advisory):
    """The only numbers the message may contain (so the SLM can't invent stats)."""
    d = advisory["derived"]
    nums = {"3", "12"}                       # "~3 months", ">=12 points" are baseline
    for v in (d["median_z"], d["sd"], d["pct_below"], advisory.get("plot_z")):
        if v is not None:
            nums.add(str(v))
            nums.add(str(abs(v)))
    return nums


# A "watch"/"may" driver (rains, monsoon, water source) must stay a POSSIBILITY — it may
# NOT be asserted as currently happening ("rains ARE underperforming", "water source IS
# running low"). Keys on an auxiliary verb (is/are/has been/...) between the driver and a
# negative word; the hedged "... may underperform" has no such auxiliary, so it passes.
_WATCH_DRIFT = re.compile(
    r"\b(rains?|monsoon)\b[^.]{0,30}\b(is|are|has been|have been|was|were)\b[^.]{0,15}"
    r"(underperform|fail|weak|poor|insufficient)"
    r"|\b(water source|borewell|canal|water table)\b[^.]{0,30}\b(is|are|has been|have been)\b"
    r"[^.]{0,15}(low|running low|depleted|declin|drying)",
    re.I)

# The recent trend is PAST context, not a forecast. On a NORMAL/ABOVE outlook the message
# must not project a FUTURE decline of the vegetation (e.g. "may experience a decline over
# the next months"). Keys on a forward modal followed by a decline word; the past-tense
# trend line ("has been declining lately") has no forward modal, so it passes.
_FWD_DECLINE = re.compile(
    r"\b(may|might|could|will|expect\w*|likely|going to|set to)\b[^.]{0,25}"
    r"\b(decline|declining|drop|dropping|fall|falling|worsen|deteriorat\w*|get worse|go below|run below)\b",
    re.I)


def validate(text, advisory):
    """True iff the text is faithful & safe: no forbidden words, no untraceable numbers,
    no over/under-statement of severity, and no watch-item asserted as a present fact."""
    low = text.lower()
    if any(w in low for w in C.FORBIDDEN_WORDS):
        return False
    # Positioning: the message should READ as an opinion (it keeps "one opinion"/"may"),
    # but must never use the literal label "second opinion". We don't add a "do not say
    # X" prompt rule (naming it PRIMES the SLM to emit it); we strip the phrase from the
    # prompt and reject it here -> fall back to the (opinion-worded) template.
    if "second opinion" in low or "2nd opinion" in low:
        return False
    allowed = _allowed_numbers(advisory)
    for tok in re.findall(r"-?\d+\.?\d*", text):
        if tok not in allowed and tok.lstrip("-") not in allowed:
            return False
    # possibility framing — don't let a "watch"/"may" driver be stated as happening now.
    if _WATCH_DRIFT.search(low):
        return False
    # a regime-split message must keep BOTH branch labels so the reader can tell which is which.
    if advisory.get("regime_split") and (
            "irrigated" not in low
            or not any(p in low for p in ("rain-fed", "rain fed", "rainfed"))):
        return False
    # ...and conversely, a NON-split advisory must not INVENT regime branches the rule engine
    # never chose. A larger SLM (gemma3:4b) sometimes sprouts "if rain-fed / if irrigated"
    # with fabricated levers on a single-branch reading (e.g. a check-your-field/local case).
    if not advisory.get("regime_split"):
        rf_branch = any(p in low for p in ("if your field is rain-fed", "if rain-fed",
                                           "if rain fed", "if rainfed"))
        ir_branch = "if your field is irrigated" in low or "if irrigated" in low
        if rf_branch and ir_branch:
            return False
    # magnitude fidelity — the message must not UNDER- or OVER-state the severity.
    # Under-stating (softening a real below-normal read) was the team's #1 concern;
    # over-stating (a NORMAL/ABOVE read hallucinated into a "below" alarm) is just as
    # bad and would read as a false alarm. Either way -> reject -> fall back to template.
    d = advisory["derived"]
    sev = d["severity"]
    attr = d["attribution"]
    # These hold for a NORMAL/ABOVE outlook regardless of attribution: never invent a
    # below-normal alarm, and never project the PAST trend as a FUTURE decline.
    if sev in ("normal", "above"):
        if "below" in low:
            return False
        if _FWD_DECLINE.search(low):
            return False
    # POSITIVE magnitude fidelity (the message must actually STATE the area severity) applies
    # only when the AREA severity is the subject of the message. For attribution=='local' the
    # message is about the farmer's PLOT (the area itself reads normal/above), so it does NOT
    # restate the area severity -> skip. Also skipped when already gated to a low-confidence watch.
    if attr != "local" and not d.get("data_poor_gate"):
        wellbelow = ("well below", "well-below", "notably below", "noticeably below",
                     "significantly below")
        if sev == "well_below" and not any(p in low for p in wellbelow):
            return False                                  # don't soften well-below
        if sev == "below":
            if "below" not in low:
                return False                              # must state below
            if any(p in low for p in ("well below", "well-below")):
                return False                              # don't overstate below -> well-below
        if sev == "above" and "above" not in low:
            return False
    # --- phrasing must stay aligned with the FINAL decision, not just the severity ---
    eff = d["effective_confidence"]
    # (a) a confidence adjective in the text must match the headline confidence.
    for lvl, phrases in (("high", ("high confidence",)), ("medium", ("medium confidence",)),
                         ("low", ("low confidence", "low-confidence"))):
        if lvl != eff and any(p in low for p in phrases):
            return False
    # (b) a small sample (n<12) must stay acknowledged, else it reads as over-confident.
    if d.get("small_sample") and not any(p in low for p in
                                         ("few", "small sample", "limited sample", "handful")):
        return False
    # (c) when data quality pulled confidence BELOW the point agreement, the reason must survive.
    #     A deliberate conf_cap (e.g. a non-cropland veg-fallback) is its OWN reason — explained by the
    #     appended veg note — so it does not need a data-quality word.
    if (C.CONF_ORDER.index(eff) < C.CONF_ORDER.index(d["point_agreement"])
            and not d.get("conf_capped")
            and not any(p in low for p in ("fair", "cloud", "limited", "quality",
                                           "could not be assessed", "low-confidence", "low confidence"))):
        return False
    return True


# ---------------------------------------------------------------------------
# v1 — leashed Gemma 3n rephraser/translator
# ---------------------------------------------------------------------------
_PROMPT = """You are a translator/rephraser for a farm advisory. Rewrite the MESSAGE \
below into {lang}, in warm, simple language a farmer understands. Keep the SAME number of \
bullet points — reword each one; never add or remove a bullet.

EXAMPLE
MESSAGE:
KOTHA — vegetation outlook, next ~3 months:
• Vegetation looks about NORMAL for your area over the next ~3 months — a greenness read only. Nothing to flag on the vegetation itself; keep an eye on your local conditions (water, weather). One opinion only.
REWRITE:
KOTHA — vegetation outlook, next ~3 months:
• Your area's greenery looks about normal for the next ~3 months — just a greenness read. Nothing stands out; keep a normal eye on your local water and weather. One satellite opinion only.

HARD RULES:
- Do NOT add any fact, number, crop name, or recommendation that is not already in the MESSAGE.
- Do NOT use the word "drought" or any alarming word.
- Keep it a gentle OPINION ("may", "one opinion", "confirm locally").
- Preserve the STRENGTH of the assessment exactly: if it says "well below", keep
  "well below" (do NOT soften it to just "below" or "a little below").
- The reading is about the area's VEGETATION being below its usual — keep that subject;
  do NOT re-cast it as moisture, rainfall, or yield being below.
- This is a VEGETATION (greenness) read ONLY. If the MESSAGE says vegetation is normal or
  above, do NOT broaden it into a general all-clear like "nothing to worry about" or
  "everything is fine" — keep it scoped to the vegetation and KEEP any "watch your local
  conditions (water, weather)" note.
- Keep "watch"/"may" items as POSSIBILITIES; do NOT restate them as things happening now
  or already happened. NOT "the rains are underperforming", NOT "your water source is low";
  say "the rains MAY underperform", "your water source MAY run low".
- The "recent trend" is PAST context (what already happened), NOT a forecast. Do NOT turn
  it into a forward statement (e.g. NOT "vegetation may decline over the next months").
- If the MESSAGE has "If rain-fed" and "If irrigated" branches, KEEP BOTH and KEEP the explicit
  labels "If your field is rain-fed:" and "If your field is irrigated:" at the start of each —
  the reader MUST be able to tell which advice is which. Do not drop a label or merge them.
- Keep each branch's steps WITH that branch: the rain-fed steps (conserve moisture, a
  life-saving irrigation) stay under "If rain-fed"; the water-source steps (secure the source,
  micro-irrigate, ration) stay under "If irrigated". NEVER move a step across branches.
- Say "confirm locally" / "with your KVK" / "one opinion" AT MOST ONCE (at the very end) —
  do NOT repeat it on every bullet or branch.
- FORMAT as short bullet POINTS — a title line, then one point per line, each point
  line starting with "• " (keep the same points as the MESSAGE; do NOT merge into a paragraph).
- Keep it concise (under ~120 words; a regime-split message with two branches may use the
  upper end). Output only the rewritten message, nothing else.

MESSAGE:
{message}
"""


def phrase_with_slm(advisory, lang="English", model=None, url=None, timeout=60):
    """Call Gemma 3n (via Ollama) to reword the template message. Falls back to the
    template on any error or if the output fails faithfulness validation."""
    base = render_template(advisory)                 # the ground truth to reword
    model = model or C.SLM_MODEL
    url = url or C.SLM_URL
    try:
        payload = json.dumps({
            "model": model,
            "prompt": _PROMPT.format(lang=lang, message=base),
            "stream": False,
            "options": {"temperature": C.SLM_TEMP},
        }).encode()
        req = urllib.request.Request(url, data=payload,
                                     headers={"Content-Type": "application/json"})
        resp = json.load(urllib.request.urlopen(req, timeout=timeout))
        out = (resp.get("response") or "").strip()
        out = out.replace("▁", " ")                    # strip sentencepiece artifacts
        out = re.sub(r"[ \t]+", " ", out)              # collapse spaces/tabs, KEEP newlines
        out = re.sub(r"\n{3,}", "\n\n", out).strip()   # cap blank lines (keep the bullets)
    except Exception:
        return base                                  # model down -> template
    # only translation is allowed to introduce non-ASCII; validate the numbers/words
    return out if (out and validate(out, advisory)) else base


def phrase(advisory, lang="English", use_slm=None):
    """Dispatcher: template by default; SLM only when explicitly enabled."""
    use = C.SLM_ENABLED if use_slm is None else use_slm
    return phrase_with_slm(advisory, lang=lang) if use else render_template(advisory)
