"""Post-prediction intelligence layers (PRD section 8):
risk banding, disruption proxy, enforcement priority, patrol allocation.

These are deterministic business logic, not ML -- kept separate so they can be
tuned/explained without retraining anything.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import schema as S
from .config import Config
from .logging_utils import get_logger

log = get_logger("intelligence")

PRED_COL = "predicted_violations"


# --- 8.1 Risk banding -------------------------------------------------------
def risk_band(values: pd.Series, cfg: Config) -> pd.Series:
    names = [b.name for b in cfg.risk_bands]
    # np.inf upper bound on the last band; bins must be monotincreasing.
    edges = [-np.inf] + [b.max for b in cfg.risk_bands]
    return pd.cut(values, bins=edges, labels=names, right=True).astype(str)


# --- 8.2 Disruption proxy (HEURISTIC, not measured congestion) --------------
def zone_vehicle_weight(events: pd.DataFrame, cfg: Config) -> pd.Series:
    """Mean road-blocking weight of vehicles seen at each zone."""
    w = events[S.VEHICLE_TYPE].astype(str).str.upper().map(cfg.disruption.vehicle_weights)
    w = w.fillna(cfg.disruption.default_vehicle_weight)
    return w.groupby(events[S.ZONE]).mean().rename("veh_weight")


def disruption_proxy(
    zone_frame: pd.DataFrame, events: pd.DataFrame, cfg: Config
) -> pd.DataFrame:
    """Add a transparent disruption proxy column to a per-zone frame.

    proxy = predicted_violations * mean_vehicle_weight * road_weight
    Clearly a heuristic ranking aid -- NOT a congestion measurement.
    """
    out = zone_frame.copy()
    veh_w = zone_vehicle_weight(events, cfg)
    out = out.merge(veh_w.reset_index(), on=S.ZONE, how="left")
    out["veh_weight"] = out["veh_weight"].fillna(cfg.disruption.default_vehicle_weight)

    road_w = np.where(
        out.get(S.ZONE_KIND, "junction") == "junction",
        cfg.disruption.junction_road_weight,
        cfg.disruption.side_street_weight,
    )
    out["disruption_proxy"] = out[PRED_COL] * out["veh_weight"] * road_w
    return out


# --- 8.4 Enforcement priority ----------------------------------------------
def _minmax(s: pd.Series) -> pd.Series:
    lo, hi = float(s.min()), float(s.max())
    if hi - lo < 1e-12:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - lo) / (hi - lo)


def enforcement_priority(zone_frame: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Priority = 0.6*pred + 0.3*historical + 0.1*junction_weight, on 0-100.

    Each component is min-max normalised so the weights are comparable.
    """
    out = zone_frame.copy()
    pred_n = _minmax(out[PRED_COL])
    hist_n = _minmax(out["zone_hist_mean"]) if "zone_hist_mean" in out else pred_n * 0
    junc_n = out["is_junction"] if "is_junction" in out else pd.Series(0.0, index=out.index)

    score = (
        cfg.priority.w_predicted * pred_n
        + cfg.priority.w_historical * hist_n
        + cfg.priority.w_junction * junc_n
    )
    out["priority_score"] = (100.0 * score).round(1)
    return out.sort_values("priority_score", ascending=False).reset_index(drop=True)


# --- 8.5 Patrol allocation (greedy + spatial spread) ------------------------
def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlmb = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2) ** 2
    return float(2 * r * np.arcsin(np.sqrt(a)))


def allocate_patrols(ranked_zones: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Assign N teams to the highest-priority zones, skipping any zone within
    ``spatial_suppress_km`` of an already-assigned one so teams spread out.
    """
    assigned: list[dict] = []
    for _, row in ranked_zones.iterrows():
        if len(assigned) >= cfg.patrol.num_teams:
            break
        too_close = any(
            _haversine_km(row[S.ZONE_LAT], row[S.ZONE_LON], a["zone_lat"], a["zone_lon"])
            < cfg.patrol.spatial_suppress_km
            for a in assigned
        )
        if too_close:
            continue
        assigned.append(
            {
                "team": f"Team {chr(ord('A') + len(assigned))}",
                S.ZONE: row[S.ZONE],
                "priority_score": row["priority_score"],
                PRED_COL: round(float(row[PRED_COL]), 1),
                "risk": row.get("risk", ""),
                "zone_lat": row[S.ZONE_LAT],
                "zone_lon": row[S.ZONE_LON],
            }
        )
    result = pd.DataFrame(assigned)
    log.info("Allocated %d patrol teams (target %d)", len(result), cfg.patrol.num_teams)
    return result
