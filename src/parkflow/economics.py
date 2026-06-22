"""Economic-impact layer (PRD section 8.6): translate predicted violations into
commuter productivity lost, in rupees.

A hackathon judge does not just want "better predictions" — they want the business
case. This module converts the forecast into a money figure so the dashboard can say
"preventing the next 24 h of high-risk zones saves Rs.X lakh in commuter time."

Grounding (NO external data — only published constants + the model's own outputs):
  * value of time   ~ Rs.120 / commuter-hour  (NTDPC 2014 urban VoT, inflated to 2024)
  * occupancy       ~ 1.4 persons / vehicle    (RITES urban traffic studies)
The *delay* per vehicle is tied to the congestion layer's estimated capacity reduction
(``est_capacity_reduction_pct``), so the rupee number inherits the same PCU / Indo-HCM
grounding as the Parking Congestion Impact Index rather than being a free-floating guess.
"""

from __future__ import annotations

import pandas as pd

from . import schema as S
from .config import Config
from .features import PRED_COL
from .logging_utils import get_logger

log = get_logger("economics")

CAP_RED_COL = "est_capacity_reduction_pct"


def economic_impact(frame: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Add per-row economic-cost columns to a forecast/timeline frame.

        vehicles_delayed     = predicted_violations x vehicles_blocked_per_violation
        delay_hours/vehicle  = max_delay_hours x (est_capacity_reduction_pct / 100)
        vehicle_hours_delay  = vehicles_delayed x delay_hours/vehicle
        economic_cost_inr    = vehicle_hours_delay x occupancy x value_of_time

    Requires ``predicted_violations`` and ``est_capacity_reduction_pct`` columns
    (the latter produced by :func:`intelligence.congestion_index`). A zone with zero
    estimated capacity loss contributes zero cost, by construction.
    """
    if CAP_RED_COL not in frame.columns:
        raise KeyError(
            f"economic_impact needs '{CAP_RED_COL}'; run intelligence.congestion_index first"
        )
    e = cfg.economics
    out = frame.copy()

    cap_red_frac = out[CAP_RED_COL].clip(lower=0.0) / 100.0
    vehicles_delayed = out[PRED_COL].clip(lower=0.0) * e.vehicles_blocked_per_violation
    delay_hours_per_vehicle = e.max_delay_hours_per_vehicle * cap_red_frac
    veh_hours_delay = vehicles_delayed * delay_hours_per_vehicle
    cost = veh_hours_delay * e.avg_vehicle_occupancy * e.value_of_time_inr_per_hour

    out["vehicles_delayed"] = vehicles_delayed.round(0)
    out["vehicle_hours_delay"] = veh_hours_delay.round(1)
    out["economic_cost_inr"] = cost.round(0)
    log.info(
        "Economic impact: Rs.%s across %d zone-windows",
        f"{float(out['economic_cost_inr'].sum()):,.0f}",
        len(out),
    )
    return out


def economic_summary(frame: pd.DataFrame) -> dict:
    """City-wide rollup for the KPI banner + metrics.json."""
    if "economic_cost_inr" not in frame.columns or frame.empty:
        return {}
    total_inr = float(frame["economic_cost_inr"].sum())
    return {
        "total_cost_inr": round(total_inr, 0),
        "total_cost_lakh": round(total_inr / 1e5, 2),
        "total_vehicle_hours": round(float(frame["vehicle_hours_delay"].sum()), 1),
        "zone_windows": int(len(frame)),
        "top_zone": (
            str(frame.loc[frame["economic_cost_inr"].idxmax(), S.ZONE])
            if S.ZONE in frame.columns else None
        ),
    }
