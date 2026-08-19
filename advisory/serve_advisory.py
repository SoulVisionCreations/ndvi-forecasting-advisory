"""
advisory.serve_advisory — a thin, SEPARATE FastAPI service for the advisory layer.

It is deliberately its own app (not a new route on inference/serve_api.py) so the
forecasting service stays byte-for-byte unchanged.

Two backends (env ADVISORY_ENGINE):
  * "inprocess" (default) — load the forecaster ONCE (advisory.engine.AdvisoryEngine)
    and run the plot + points as one BATCHED forward (~1.5-2 min end-to-end). Needs
    the model weights (RUN_DIR/MODEL_DIR/LOOKUP_CSV) + a GPU.
  * "http" — the original path: 13 serial calls to /forecast on :8001 (~7-8 min).
If the engine fails to load at startup, we fall back to "http" rather than brick :8011.

Run (from repo root):
    ADVISORY_ENGINE=inprocess RUN_DIR=<weights> python -m uvicorn advisory.serve_advisory:app --port 8011
POST /advisory  {"lat":20.17,"lon":78.32,"regime":"rain-fed","date":"2018-06-15"}
    optional: "lang":"Marathi", "use_slm":true
"""
import os

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from . import advisory as adv_mod
from . import phraser

app = FastAPI(title="Vegetation Outlook — advisory layer", version="0.2.0")

# in-process engine (loaded at startup when ADVISORY_ENGINE=inprocess); None -> HTTP path.
_ENGINE = None
_MODE = os.environ.get("ADVISORY_ENGINE", "inprocess")


@app.on_event("startup")
def _load_engine():
    global _ENGINE, _MODE
    if _MODE != "inprocess":
        print("[serve_advisory] mode=http — advisory calls /forecast over HTTP")
        return
    from .engine import AdvisoryEngine, preflight_paths
    # FAIL FAST on a bad model path: abort startup with a clear message rather than
    # silently falling back to HTTP mode (a wrong RUN_DIR/MODEL_DIR would otherwise
    # look "up" but never actually serve the in-process forecaster).
    problems = preflight_paths()
    if problems:
        msg = ("[serve_advisory] STARTUP ABORTED — model artifact path(s) missing or incorrect:\n  - "
               + "\n  - ".join(problems)
               + "\nFix RUN_DIR / MODEL_DIR / LOOKUP_CSV (or extract the Model zip into weights/), "
                 "then restart. Not starting the service.")
        print(msg)
        raise RuntimeError(msg)                       # aborts uvicorn startup (non-zero exit)
    try:
        _ENGINE = AdvisoryEngine()
        print("[serve_advisory] mode=inprocess — forecaster loaded; batched forward")
    except Exception as e:
        # FAIL FAST: paths were present (preflight passed) but the model still failed to
        # load (corrupt bundle, torch/arch mismatch, or a missing dependency). Do NOT
        # silently fall back to HTTP — abort with the real error so the problem is visible
        # instead of the service coming "up" but never serving the in-process forecaster.
        # (Deliberate HTTP mode is still available by setting ADVISORY_ENGINE=http.)
        msg = (f"[serve_advisory] STARTUP ABORTED — the forecaster failed to load: {e}\n"
               "Model files were present (path preflight passed), so this is a LOAD error "
               "(corrupt bundle, torch/architecture mismatch, or a missing dependency). Fix it and "
               "restart; or set ADVISORY_ENGINE=http to use the /forecast HTTP backend on purpose. "
               "Not starting the service.")
        print(msg)
        raise RuntimeError(msg) from e                # aborts uvicorn startup (non-zero exit)


class AdvisoryRequest(BaseModel):
    lat: float
    lon: float
    regime: str | None = Field(None, description="rain-fed | irrigated | omit/null = show BOTH")
    date: str | None = Field(None, description="as-of date YYYY-MM-DD (default: today)")
    lang: str = Field("English", description="output language for the phraser")
    use_slm: bool | None = Field(None, description="override: use the SLM phraser")
    n_points: int | None = Field(None, description="override the min sampled points "
                                 "(e.g. 4 for a quick test; default 12 = the spec)")
    verbose: bool = Field(False, description="also return the full derived/matched block")


@app.get("/health")
def health():
    return {"status": "ok", "layer": "advisory", "version": app.version,
            "mode": _MODE, "engine_loaded": _ENGINE is not None}


def _advise(lat, lon, regime, date, *, lang="English", use_slm=None, min_points=None):
    """Run the advisory via the in-process engine (preferred) or the HTTP fallback.
    Shared by /advisory and the OpenAI-compatible shim."""
    if _ENGINE is not None:                          # in-process batched engine (~1.5-2 min)
        return _ENGINE.advise(lat, lon, regime, date,
                              lang=lang, use_slm=use_slm, min_points=min_points)
    return adv_mod.build_advisory(lat, lon, regime, date,   # HTTP fallback: 13 /forecast calls
                                  lang=lang, use_slm=use_slm, min_points=min_points)


def _classify_error(e):
    """Map an internal exception to (status_code, clean user-facing message), so a failed
    request returns a short, categorized message instead of a raw stack trace."""
    low = str(e).lower()
    # bad / future / malformed date (the guard messages are already clear) -> 422
    if any(m in low for m in ("in the future", "invalid date", "must be today", "expected format")):
        return 422, str(e)
    # location outside coverage (CoreStack admin 404 / off-land) -> 422
    if "404" in low or "not found" in low:
        return 422, ("This location looks outside the supported coverage (off-land, or not in the "
                     "CoreStack admin / cropland map). Try a point inside an agricultural district in India.")
    # a data service is unreachable (DNS / connection / timeout) -> 503, retryable
    net = ("nameresolution", "failed to resolve", "getaddrinfo", "max retries", "httpsconnectionpool",
           "httpconnectionpool", "connection refused", "connection reset", "connection aborted",
           "network is unreachable", "timed out", "timeout", "temporary failure in name resolution",
           "connectionerror", "read timed out")
    if any(m in low for m in net) or isinstance(e, (ConnectionError, TimeoutError)):
        return 503, ("A data service the advisory depends on is unreachable right now (Earth Engine / "
                     "CoreStack / weather). Check your network connection and try again in a moment.")
    # no usable satellite / weather data at the point -> 422
    if any(m in low for m in ("no usable data", "plot forecast unavailable", "no point forecasts")):
        return 422, ("No usable satellite / weather data at this location for this date. "
                     "Try a nearby agricultural point, or an earlier date.")
    # a data service rejected the request (bad/missing key or auth) -> 502 with a config hint
    if any(m in low for m in ("401", "403", "unauthorized", "forbidden")):
        return 502, "A data service rejected the request — check your CORESTACK_KEY and Earth Engine auth."
    # anything else -> generic 502 (no stack trace surfaced)
    return 502, "The advisory could not be produced due to an internal error. Please try again."


@app.post("/advisory")
def advisory(req: AdvisoryRequest):
    if req.regime is not None and req.regime not in ("rain-fed", "irrigated"):
        raise HTTPException(422, "regime must be 'rain-fed', 'irrigated', or omitted (both)")
    try:
        a = _advise(req.lat, req.lon, req.regime, req.date,
                    lang=req.lang, use_slm=req.use_slm, min_points=req.n_points)
    except Exception as e:
        status, msg = _classify_error(e)
        raise HTTPException(status, msg)

    # drop regime from location when it was unspecified (null)
    loc = {k: v for k, v in a["input"].items() if not (k == "regime" and v is None)}
    out = {
        "location": loc,
        "risk_level": a["matched"]["risk_level"],
        "confidence": a["derived"]["effective_confidence"],   # the single surfaced confidence
        "message": a["message"],
    }
    if req.verbose:
        # internal flags + the confidence intermediates (top-level "confidence" == effective) — not surfaced
        _hidden = ("data_poor_gate", "clear_view_source",
                   "overall_confidence", "effective_confidence", "conf_capped")
        derived = {k: v for k, v in a["derived"].items() if k not in _hidden}
        out.update(derived=derived, trend_context=a["trend_context"], sampling=a.get("sampling"))
        # Surface the season lens ONLY when the message actually carries its levers (i.e. an
        # area-wide below-usual read). Otherwise the deficit-framed lens would contradict a
        # normal / above / your-plot-only message. On a regime-split (regime unspecified) the
        # message shows both branches, so surface both lenses.
        if phraser.levers_shown(a):
            out["lens"] = a["lens"]
            if a.get("regime_split"):
                out["lens_irrigated"] = a.get("lens_irrigated")
    if a.get("warnings"):        # operator-facing (e.g. rate-limit) — surfaced even when not verbose
        out["warnings"] = a["warnings"]
    return out


# ---- OpenAI-compatible shim (Open WebUI) --------------------------------------------
# Lets Open WebUI (or any OpenAI-API client) drive the advisory as a "chat model": it POSTs
# a chat message, we parse lat/lon/date/regime out of it, run advise(), and return the
# bulleted message as the assistant reply. Purely additive: reuses the already-loaded
# engine — one service now speaks /advisory AND /v1/chat/completions (no second model load).
import json as _json
import re as _re
import time as _time
from fastapi.responses import StreamingResponse

_MODEL_ID = os.environ.get("ADVISORY_OPENAI_MODEL_ID", "vegetation-outlook")
_NUM = _re.compile(r"(-?\d{1,3}\.\d+)")          # decimals -> first two = lat, lon
_DATE = _re.compile(r"(\d{4}-\d{2}-\d{2})")       # optional ISO date


def _parse_query(text):
    """Pull (lat, lon, date, regime) from free chat text, e.g.
    'vegetation outlook for 20.17, 78.32 on 2018-06-15 irrigated'. Regime defaults rain-fed."""
    t = text or ""
    nums = _NUM.findall(t)
    if len(nums) < 2:
        return None
    m = _DATE.search(t)
    if _re.search(r"irrigat", t, _re.I):
        regime = "irrigated"
    elif _re.search(r"rain.?fed", t, _re.I):
        regime = "rain-fed"
    else:
        regime = None                                # unspecified -> show BOTH
    return float(nums[0]), float(nums[1]), (m.group(1) if m else None), regime


def _run_advisory_text(text):
    q = _parse_query(text)
    if q is None:
        return ("Please give a location as lat, lon (and an optional date + rain-fed/"
                "irrigated), e.g.\n`vegetation outlook for 20.17, 78.32 on 2018-06-15`")
    lat, lon, date, regime = q
    try:
        a = _advise(lat, lon, regime, date)
    except Exception as e:
        return "Sorry — " + _classify_error(e)[1]
    return a.get("message") or _json.dumps(a)


@app.get("/v1/models")                            # Open WebUI calls this to fill the model dropdown
def list_models():
    return {"object": "list",
            "data": [{"id": _MODEL_ID, "object": "model", "created": 0, "owned_by": "ndvi"}]}


@app.post("/v1/chat/completions")                 # ...and this to chat
def chat_completions(body: dict):
    msgs = body.get("messages") or []
    user = next((m.get("content", "") for m in reversed(msgs) if m.get("role") == "user"), "")
    content = _run_advisory_text(user if isinstance(user, str) else str(user))
    now = int(_time.time())
    cid = f"chatcmpl-{now}"
    if body.get("stream"):                         # Open WebUI streams by default
        def sse():
            chunk = {"id": cid, "object": "chat.completion.chunk", "created": now, "model": _MODEL_ID,
                     "choices": [{"index": 0, "delta": {"role": "assistant", "content": content},
                                  "finish_reason": None}]}
            yield f"data: {_json.dumps(chunk)}\n\n"
            end = {"id": cid, "object": "chat.completion.chunk", "created": now, "model": _MODEL_ID,
                   "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
            yield f"data: {_json.dumps(end)}\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(sse(), media_type="text/event-stream")
    return {"id": cid, "object": "chat.completion", "created": now, "model": _MODEL_ID,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": content},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}}
