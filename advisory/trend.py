"""
advisory.trend — the recent-trend CONTEXT (recovering / holding / declining).

IMPORTANT: this is DATA, not the model, and it is CONTEXT ONLY. A backtest showed that
using it to inflate/deflate confidence did NOT improve reliability, so the rule engine
never lets it change the row — it is only shown to the farmer to weigh.

It needs the location's recent observed NDVI vs its per-fortnight normal. The deployed
/forecast does not currently return that series, so at serve time the trend is often
'unknown'; wiring it is a follow-up (the forecaster already computes the window).
"""
import statistics as st


def classify(recent_anomalies):
    """recent_anomalies: list of recent (NDVI - seasonal_normal) values, oldest->newest.
    Returns 'recovering' | 'declining' | 'holding' | 'unknown'."""
    a = [x for x in (recent_anomalies or []) if x is not None]
    if len(a) < 4:
        return "unknown"
    delta = st.mean(a[-2:]) - st.mean(a[:2])
    if delta > 0.03:
        return "recovering"
    if delta < -0.03:
        return "declining"
    return "holding"
