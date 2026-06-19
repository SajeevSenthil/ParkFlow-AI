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


# --- 8.2 Congestion Impact Score (CIS) --------------------------------------
# Data-driven proxy built from five observed signals. NOT measured congestion.
# Column kept as "disruption_proxy" for dashboard/pipeline backward compatibility.

def _zone_vehicle_weight(events: pd.DataFrame, cfg: Config) -> pd.Series:
    """Mean road-blocking weight of vehicles seen at each zone."""
    w = events[S.VEHICLE_TYPE].astype(str).str.upper().map(cfg.disruption.vehicle_weights)
    w = w.fillna(cfg.disruption.default_vehicle_weight)
    return w.groupby(events[S.ZONE]).mean().rename("veh_weight")


def _zone_violation_severity(events: pd.DataFrame, cfg: Config) -> pd.Series:
    """Mean carriageway-blocking severity of violation types seen at each zone."""
    sev = events[S.VIOLATION_TYPE].astype(str).str.upper().map(
        cfg.disruption.violation_severity_weights
    ).fillna(cfg.disruption.default_violation_severity)
    return sev.groupby(events[S.ZONE]).mean().rename("viol_severity")


def _zone_simultaneous_density(events: pd.DataFrame, cfg: Config) -> pd.Series:
    """Max violations recorded in any single time-bin at each zone, normalised 0-1.

    A zone where 200 vehicles were illegally parked simultaneously is far more
    disruptive than one where 200 violations occurred over several months.
    """
    if S.CREATED_DATETIME not in events.columns:
        return pd.Series(dtype=float)
    bins = events[S.CREATED_DATETIME].dt.floor(f"{cfg.temporal.bin_hours}h")
    bin_counts = events.groupby([S.ZONE, bins]).size()
    max_per_zone = bin_counts.groupby(level=0).max().rename("sim_density")
    global_max = float(max_per_zone.max()) if len(max_per_zone) else 1.0
    return (max_per_zone / max(global_max, 1.0))


def _zone_repeat_offender_factor(events: pd.DataFrame, cfg: Config) -> pd.Series:
    """log1p of the count of vehicles exceeding the repeat-offender threshold per zone."""
    if S.VEHICLE_NUMBER not in events.columns:
        return pd.Series(dtype=float)
    threshold = cfg.disruption.repeat_offender_threshold
    per_zone_veh = events.groupby([S.ZONE, S.VEHICLE_NUMBER]).size()
    repeat_count = (per_zone_veh > threshold).groupby(level=0).sum()
    return np.log1p(repeat_count).rename("repeat_factor")


def disruption_proxy(
    zone_frame: pd.DataFrame, events: pd.DataFrame, cfg: Config
) -> pd.DataFrame:
    """Compute the Congestion Impact Score (CIS) for each zone.

    CIS = simultaneous_density_norm
          × violation_severity_weight
          × vehicle_blocking_weight
          × peak_hour_multiplier
          × log1p(repeat_offender_count)

    Result is min-max scaled to 0-100 and stored as ``disruption_proxy`` so the
    rest of the pipeline and dashboard need no changes.
    Labeled 'Congestion Impact Score' in the UI; clearly NOT measured congestion.
    """
    out = zone_frame.copy()
    dc = cfg.disruption

    # --- Signal 1: simultaneous violation density (0-1) ---
    sim_dens = _zone_simultaneous_density(events, cfg)
    out = out.merge(sim_dens.reset_index(name="sim_density"), on=S.ZONE, how="left")
    out["sim_density"] = out["sim_density"].fillna(0.0)

    # --- Signal 2: violation severity weight ---
    viol_sev = _zone_violation_severity(events, cfg)
    out = out.merge(viol_sev.reset_index(), on=S.ZONE, how="left")
    out["viol_severity"] = out["viol_severity"].fillna(dc.default_violation_severity)

    # --- Signal 3: vehicle blocking weight ---
    veh_w = _zone_vehicle_weight(events, cfg)
    out = out.merge(veh_w.reset_index(), on=S.ZONE, how="left")
    out["veh_weight"] = out["veh_weight"].fillna(dc.default_vehicle_weight)

    # --- Signal 4: peak-hour multiplier (based on forecast bin's hour) ---
    peak_hours = set(dc.peak_hours_morning) | set(dc.peak_hours_evening)
    if S.BIN_START in out.columns:
        hour = pd.to_datetime(out[S.BIN_START]).dt.hour
    else:
        hour = pd.Series(0, index=out.index)
    peak_mult = np.where(hour.isin(peak_hours), dc.peak_hour_multiplier, 1.0)

    # --- Signal 5: repeat-offender density ---
    rep_factor = _zone_repeat_offender_factor(events, cfg)
    out = out.merge(rep_factor.reset_index(), on=S.ZONE, how="left")
    out["repeat_factor"] = out["repeat_factor"].fillna(0.0)

    # --- CIS: product of all five signals ---
    raw_cis = (
        out["sim_density"]
        * out["viol_severity"]
        * out["veh_weight"]
        * peak_mult
        * (1.0 + out["repeat_factor"])
    )

    # Scale to 0-100 for readability.
    cis_min, cis_max = float(raw_cis.min()), float(raw_cis.max())
    if cis_max - cis_min > 1e-12:
        out["disruption_proxy"] = ((raw_cis - cis_min) / (cis_max - cis_min) * 100).round(1)
    else:
        out["disruption_proxy"] = 0.0

    # Drop intermediate columns to keep artifact lean.
    out = out.drop(columns=["sim_density", "viol_severity", "veh_weight", "repeat_factor"],
                   errors="ignore")
    log.info("Computed CIS (disruption_proxy) for %d zones", len(out))
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
