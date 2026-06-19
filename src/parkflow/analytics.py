"""Supporting analytics (PRD deliverables 5 & 6): temporal trends and
junction risk assessment. These feed the dashboard's Analytics Center and
Junction Risk tabs.
"""

from __future__ import annotations

import pandas as pd

from . import schema as S
from .config import Config
from .logging_utils import get_logger

log = get_logger("analytics")

# Columns shipped to the dashboard so it can compute filtered trends/heatmaps
# without re-running the model (read-only consumption of a precomputed table).
ANALYTICS_COLUMNS = [
    S.LATITUDE,
    S.LONGITUDE,
    S.ZONE,
    S.ZONE_KIND,
    S.POLICE_STATION,
    S.VIOLATION_TYPE,
    S.VEHICLE_TYPE,
    S.CREATED_DATETIME,
]

_DOW_ORDER = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def analytics_events(events: pd.DataFrame) -> pd.DataFrame:
    """Compact cleaned-events table for the dashboard."""
    cols = [c for c in ANALYTICS_COLUMNS if c in events.columns]
    out = events[cols].copy()
    if S.VALIDATION_STATUS in events.columns:
        out[S.VALIDATION_STATUS] = events[S.VALIDATION_STATUS].astype(str).str.lower()
    return out


def temporal_trends(events: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Hour / day-of-week / weekly / monthly violation counts."""
    ts = events[S.CREATED_DATETIME]
    by_hour = (
        events.assign(hour=ts.dt.hour).groupby("hour").size().rename("violations").reset_index()
    )
    by_dow = (
        events.assign(dow=ts.dt.dayofweek).groupby("dow").size().rename("violations").reset_index()
    )
    by_dow["day"] = by_dow["dow"].map(dict(enumerate(_DOW_ORDER)))
    by_week = (
        events.assign(week=ts.dt.to_period("W").astype(str))
        .groupby("week")
        .size()
        .rename("violations")
        .reset_index()
    )
    by_month = (
        events.assign(month=ts.dt.to_period("M").astype(str))
        .groupby("month")
        .size()
        .rename("violations")
        .reset_index()
    )
    return {"by_hour": by_hour, "by_dow": by_dow, "by_week": by_week, "by_month": by_month}


def junction_risk_table(
    events: pd.DataFrame, ranked_forecast: pd.DataFrame, cfg: Config
) -> pd.DataFrame:
    """Per-junction severity assessment (PRD deliverable 6).

    Combines historical density with the forecast/priority so each junction gets
    total + average + peak-hour + predicted-next + risk + priority in one table.
    """
    ev = events[events[S.ZONE_KIND] == "junction"].copy()
    if ev.empty:
        return pd.DataFrame()

    ts = ev[S.CREATED_DATETIME]
    ev["hour"] = ts.dt.hour
    ev["date"] = ts.dt.date

    grp = ev.groupby(S.ZONE)
    table = grp.size().rename("historical_violations").to_frame()
    table["active_days"] = grp["date"].nunique()
    table["avg_per_day"] = (table["historical_violations"] / table["active_days"]).round(2)
    table["peak_hour"] = grp["hour"].agg(lambda s: int(s.value_counts().idxmax()))
    table["dominant_vehicle"] = grp[S.VEHICLE_TYPE].agg(
        lambda s: s.astype(str).str.upper().value_counts().idxmax()
    )
    table = table.reset_index()

    # Bring in forecast-side fields if present.
    fcols = [c for c in ("predicted_violations", "risk", "priority_score") if c in ranked_forecast]
    if fcols:
        table = table.merge(ranked_forecast[[S.ZONE] + fcols], on=S.ZONE, how="left")

    sort_key = "priority_score" if "priority_score" in table else "historical_violations"
    table = table.sort_values(sort_key, ascending=False).reset_index(drop=True)
    table.insert(0, "rank", range(1, len(table) + 1))
    log.info("Built junction-risk table for %d junctions", len(table))
    return table
