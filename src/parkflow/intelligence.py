"""Post-prediction intelligence layers (PRD section 8):
risk banding, congestion-impact index, enforcement priority, patrol allocation.

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


# --- 8.2 Parking Congestion Impact Index ------------------------------------
# Headline = estimated % road capacity lost, grounded in PCU (Indo-HCM/IRC) + the
# HCM saturation-flow principle, and MODULATED by data-observed signals (violation
# severity, peak-hour). Uses only provided data + standard constants (no external data).
def zone_mean_pcu(events: pd.DataFrame, cfg: Config) -> pd.Series:
    """Mean Passenger-Car-Unit (PCU) value of vehicles seen at each zone."""
    pcu = events[S.VEHICLE_TYPE].astype(str).str.upper().map(cfg.congestion.pcu_weights)
    pcu = pcu.fillna(cfg.congestion.default_pcu)
    return pcu.groupby(events[S.ZONE]).mean().rename("mean_pcu")


def zone_violation_severity(events: pd.DataFrame, cfg: Config) -> pd.Series:
    """Mean carriageway-blocking severity of violation types seen at each zone."""
    sev = (
        events[S.VIOLATION_TYPE]
        .astype(str)
        .str.upper()
        .map(cfg.congestion.violation_severity_weights)
        .fillna(cfg.congestion.default_violation_severity)
    )
    return sev.groupby(events[S.ZONE]).mean().rename("viol_severity")


def congestion_index(
    zone_frame: pd.DataFrame, events: pd.DataFrame, cfg: Config
) -> pd.DataFrame:
    """Parking Congestion Impact Index — estimated % of road capacity lost.

        effective_load = predicted_violations × mean_PCU × road_factor
                         × violation_severity × peak_hour_multiplier
        est_cap_red%   = max_cap × (1 − exp(−effective_load / saturation_pcu))   [Indo-HCM-style]
        congestion_index (0–100) = est_cap_red% / max_cap × 100

    PCU values and the HCM saturation-flow principle are standard traffic-engineering
    constants; severity and peak-hour are observed from the provided data. No external data.
    """
    c = cfg.congestion
    out = zone_frame.copy()

    # Vehicle PCU + violation severity per zone.
    pcu = zone_mean_pcu(events, cfg)
    out = out.merge(pcu.reset_index(), on=S.ZONE, how="left")
    out["mean_pcu"] = out["mean_pcu"].fillna(c.default_pcu)
    sev = zone_violation_severity(events, cfg)
    out = out.merge(sev.reset_index(), on=S.ZONE, how="left")
    out["viol_severity"] = out["viol_severity"].fillna(c.default_violation_severity)

    # Road class + peak-hour modulation.
    road_factor = np.where(
        out.get(S.ZONE_KIND, "junction") == "junction",
        c.junction_road_factor,
        c.side_street_factor,
    )
    peak_hours = set(c.peak_hours_morning) | set(c.peak_hours_evening)
    if S.BIN_START in out.columns:
        hour = pd.to_datetime(out[S.BIN_START]).dt.hour
    else:
        hour = pd.Series(0, index=out.index)
    peak_mult = np.where(hour.isin(peak_hours), c.peak_hour_multiplier, 1.0)

    effective_load = (
        out[PRED_COL] * out["mean_pcu"] * road_factor * out["viol_severity"] * peak_mult
    )
    est_cap_red = c.max_capacity_reduction_pct * (1.0 - np.exp(-effective_load / c.saturation_pcu))

    out["pcu_load"] = effective_load.round(2)
    out["est_capacity_reduction_pct"] = est_cap_red.round(1)
    out["congestion_index"] = (100.0 * est_cap_red / c.max_capacity_reduction_pct).round(1)
    # Backward-compat alias so older dashboard column lists still resolve.
    out["disruption_proxy"] = out["congestion_index"]
    out = out.drop(columns=["mean_pcu", "viol_severity"], errors="ignore")
    log.info("Computed congestion impact index for %d zones", len(out))
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
