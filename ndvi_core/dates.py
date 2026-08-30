"""
ndvi_core.dates — serve-time anchor/target date geometry (C1).

The model predicts LEAD fortnights ahead of its last input observation (the
anchor). At serve time we snap the as-of / forecast-from date DOWN to the most
recent 14-day fortnight on a fixed grid (config.FORTNIGHT_GRID_EPOCH) -> the
anchor s0; the forecast target = anchor + LEAD*14 days. This mirrors training's
grid geometry (target_row = anchor_row + lead on the fortnightly grid). A missing
/ latency reading AT the anchor is ffilled in ndvi_core.inference.build_window
(previous value) -- the anchor is NEVER slid back to the last real reading.

Shared by infer_cli, serve_api (and any batch tool) so CLI and API resolve the
anchor/target identically (no duplicated date arithmetic).
"""

from datetime import datetime, timedelta

from . import config

FORTNIGHT_DAYS = 14


def _epoch(epoch=None):
    e = epoch if epoch is not None else config.FORTNIGHT_GRID_EPOCH
    return datetime.strptime(e, "%Y-%m-%d") if isinstance(e, str) else e


def snap_to_grid(dt, epoch=None):
    """Snap `dt` DOWN to the most recent 14-day fortnight <= dt on the grid
    anchored at `epoch`. Result is always within one fortnight of dt."""
    days = (dt - _epoch(epoch)).days
    return dt - timedelta(days=days % FORTNIGHT_DAYS)


def resolve_forecast_dates(date_str=None, lead=None, epoch=None, allow_future=False):
    """Resolve the three serve-time dates from an as-of date.

    date_str : the as-of / forecast-from date 'YYYY-MM-DD'; default = today.
    Returns (forecast_from, anchor, target) as datetimes where
      anchor = snap_to_grid(forecast_from)  -- nearest fortnight <= date (the s0),
      target = anchor + lead*14 days        -- LEAD fortnights ahead of the anchor.

    `date` must be TODAY OR EARLIER; a future date is rejected (allow_future=False).
    The model can only forecast ~lead*14 days ahead of the latest REAL data (~today),
    so a future `date` would pull today-anchored history yet label it with a future
    target season (walk-back tile + climatology weather) -- a temporal mismatch the
    model never saw in training. Use today's date for a live ~3-month forecast, or a
    past date for a backtest. allow_future=True is an escape hatch for controlled tests.

    Raises ValueError on an unparseable date_str, or on a future date_str when not allowed.
    """
    lead = config.LEAD if lead is None else lead
    now = datetime.now()
    if date_str:
        try:
            ff = datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError:
            raise ValueError(f"Invalid date '{date_str}' -- expected format YYYY-MM-DD.")
    else:
        ff = now
    ff = ff.replace(hour=0, minute=0, second=0, microsecond=0)
    if not allow_future and ff.date() > now.date():
        raise ValueError(
            f"date '{date_str}' is in the future; it must be today "
            f"({now.date()}) or earlier.")
    anchor = snap_to_grid(ff, epoch)
    target = anchor + timedelta(days=lead * FORTNIGHT_DAYS)
    return ff, anchor, target
