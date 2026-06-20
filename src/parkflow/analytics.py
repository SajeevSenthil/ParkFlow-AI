"""Supporting analytics (PRD deliverables 5 & 6): temporal trends and
junction risk assessment. These feed the dashboard's Analytics Center and
Junction Risk tabs.
"""

from __future__ import annotations

import numpy as np
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


def repeat_offender_tables(events: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Repeat-offender analytics from the (anonymized but consistent) vehicle_number.

    Returns:
      * top_offenders  — vehicles with the most violations, and how many zones they hit.
      * zone_repeat    — per zone, the share of violations from repeat offenders.
        A high *unique*-offender share suggests an infrastructure problem (bad signage,
        no legal parking); a high *repeat* share suggests willful disregard → towing.
    """
    if S.VEHICLE_NUMBER not in events.columns:
        return pd.DataFrame(), pd.DataFrame()

    counts = events[S.VEHICLE_NUMBER].value_counts()
    repeat_ids = set(counts[counts > 1].index)

    top_offenders = (
        events[events[S.VEHICLE_NUMBER].isin(repeat_ids)]
        .groupby(S.VEHICLE_NUMBER)
        .agg(violations=(S.VEHICLE_NUMBER, "size"), zones_hit=(S.ZONE, "nunique"))
        .reset_index()
        .sort_values("violations", ascending=False)
        .head(25)
        .reset_index(drop=True)
    )

    ev = events.assign(is_repeat=events[S.VEHICLE_NUMBER].isin(repeat_ids))
    zone_repeat = (
        ev.groupby(S.ZONE)
        .agg(
            total_violations=(S.VEHICLE_NUMBER, "size"),
            unique_vehicles=(S.VEHICLE_NUMBER, "nunique"),
            repeat_violations=("is_repeat", "sum"),
        )
        .reset_index()
    )
    zone_repeat["repeat_offender_share_pct"] = (
        100 * zone_repeat["repeat_violations"] / zone_repeat["total_violations"]
    ).round(1)
    zone_repeat["signal"] = np.where(
        zone_repeat["repeat_offender_share_pct"] >= 40,
        "willful (towing)",
        "infrastructure (signage/space)",
    )
    zone_repeat = zone_repeat.sort_values("total_violations", ascending=False).reset_index(drop=True)
    log.info(
        "Repeat-offender analytics: %d repeat vehicles, %.1f%% of all violations",
        len(repeat_ids),
        100 * counts[counts > 1].sum() / max(int(counts.sum()), 1),
    )
    return top_offenders, zone_repeat
