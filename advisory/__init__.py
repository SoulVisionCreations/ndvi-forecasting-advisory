"""
advisory — Vegetation Outlook advisory layer (UC1).

An ADDITIVE layer on top of the NDVI forecaster. It never modifies the forecasting
core: it calls the forecaster's HTTP /forecast endpoint as a black box, then runs a
deterministic rule engine and (optionally) a small language model that ONLY rephrases
the matched rule row.

Three stages (the design the team signed off on):
    MODEL        the forecaster emits FACTS (z vs the area's own seasonal normal)
    RULE ENGINE  deterministic: owns every number + decision (auditable)
    PHRASER      template (default) or a leashed SLM that only rewords the matched row

Positioning: an OPINION / vegetation-outlook indicator — NOT a drought detector,
NOT a source of truth. No "drought" wording in farmer-facing text.

Run from the repo root, e.g.:  python -m advisory.examples.advisory_demo
"""

__version__ = "0.1.0"
