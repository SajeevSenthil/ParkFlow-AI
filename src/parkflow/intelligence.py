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

# OR-Tools is an OPTIONAL dependency: route optimization uses it when present, and the
# pipeline gracefully falls back to greedy spatial-spread allocation when it is not, so
# `parkflow run` never hard-fails on a box without it.
try:  # pragma: no cover - import guard
    from ortools.constraint_solver import pywrapcp, routing_enums_pb2

    _HAS_ORTOOLS = True
except ImportError:  # pragma: no cover - exercised only when ortools is absent
    _HAS_ORTOOLS = False

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


def greedy_spread_zones(ranked_zones: pd.DataFrame, n: int, suppress_km: float) -> pd.DataFrame:
    """Top-``n`` priority zones with haversine spatial suppression (≥ ``suppress_km``
    apart). This is the hand-dispatcher heuristic — spread teams out — and is reused as
    the *naive* baseline the displacement comparison pits the routed layout against.
    """
    picked: list[pd.Series] = []
    for _, row in ranked_zones.iterrows():
        if len(picked) >= n:
            break
        too_close = any(
            _haversine_km(row[S.ZONE_LAT], row[S.ZONE_LON], p[S.ZONE_LAT], p[S.ZONE_LON])
            < suppress_km
            for p in picked
        )
        if not too_close:
            picked.append(row)
    return pd.DataFrame(picked).reset_index(drop=True)


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


# --- 8.5b Route-optimized patrol allocation (OR-Tools CVRP) ------------------
def _distance_matrix_m(lats: list[float], lons: list[float]) -> list[list[int]]:
    """Symmetric haversine distance matrix in integer metres (OR-Tools wants ints).

    Built purely from zone coordinates already in the data — NO external road network
    or routing API, so the project's no-external-data compliance is preserved.
    """
    n = len(lats)
    mat = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            d = int(round(_haversine_km(lats[i], lons[i], lats[j], lons[j]) * 1000))
            mat[i][j] = mat[j][i] = d
    return mat


def route_patrols(ranked_zones: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Route-optimized patrol deployment via an OR-Tools Capacitated VRP.

    Each of ``num_teams`` teams drives an *ordered* route through up to
    ``zones_per_team`` of the top-priority zones, minimizing total travel distance on a
    haversine matrix. With more candidate zones than total team capacity, node
    *disjunctions* let the solver skip zones — penalised by priority, so it preferentially
    visits the highest-priority ones. A virtual depot (the candidate centroid) models a
    shared command-centre start.

    Returns ordered stops: ``team, stop_order, zone, zone_lat, zone_lon, priority_score,
    predicted_violations, risk, route_method``. Falls back to the greedy spatial-spread
    plan (one stop per team) when OR-Tools is unavailable.
    """
    if ranked_zones.empty:
        return pd.DataFrame()

    if not _HAS_ORTOOLS:
        log.warning("OR-Tools not installed -> greedy fallback for patrol routing")
        plan = allocate_patrols(ranked_zones, cfg)
        plan["stop_order"] = 1
        plan["route_method"] = "greedy_fallback"
        return plan

    pool = (
        ranked_zones.dropna(subset=[S.ZONE_LAT, S.ZONE_LON])
        .head(cfg.patrol.route_candidate_pool)
        .reset_index(drop=True)
    )
    n = len(pool)
    if n == 0:
        return pd.DataFrame()

    # Node 0 = virtual depot (centroid of the candidate zones); nodes 1..n = zones.
    lats = [float(pool[S.ZONE_LAT].mean())] + pool[S.ZONE_LAT].astype(float).tolist()
    lons = [float(pool[S.ZONE_LON].mean())] + pool[S.ZONE_LON].astype(float).tolist()
    dist = _distance_matrix_m(lats, lons)

    n_teams = cfg.patrol.num_teams
    cap = cfg.patrol.zones_per_team
    manager = pywrapcp.RoutingIndexManager(n + 1, n_teams, 0)
    routing = pywrapcp.RoutingModel(manager)

    def _transit(i, j):
        return dist[manager.IndexToNode(i)][manager.IndexToNode(j)]

    transit_idx = routing.RegisterTransitCallback(_transit)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_idx)

    # Capacity dimension: each real zone has demand 1; teams carry at most `cap` zones.
    def _demand(i):
        return 0 if manager.IndexToNode(i) == 0 else 1

    demand_idx = routing.RegisterUnaryTransitCallback(_demand)
    routing.AddDimensionWithVehicleCapacity(demand_idx, 0, [cap] * n_teams, True, "Cap")

    # Disjunctions: skipping a zone costs a priority-scaled penalty, so the solver fills
    # team capacity with the highest-priority zones first. Penalty dominates any arc cost.
    for node in range(1, n + 1):
        priority = float(pool.loc[node - 1, "priority_score"])
        penalty = int(priority * 100_000) + 1
        routing.AddDisjunction([manager.NodeToIndex(node)], penalty)

    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    params.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    )
    params.time_limit.FromSeconds(5)

    solution = routing.SolveWithParameters(params)
    if solution is None:
        log.warning("OR-Tools found no solution -> greedy fallback")
        plan = allocate_patrols(ranked_zones, cfg)
        plan["stop_order"] = 1
        plan["route_method"] = "greedy_fallback"
        return plan

    rows: list[dict] = []
    for v in range(n_teams):
        team = f"Team {chr(ord('A') + v)}"
        idx = routing.Start(v)
        order = 1
        while not routing.IsEnd(idx):
            node = manager.IndexToNode(idx)
            if node != 0:  # skip the virtual depot
                z = pool.loc[node - 1]
                rows.append(
                    {
                        "team": team,
                        "stop_order": order,
                        S.ZONE: z[S.ZONE],
                        "zone_lat": z[S.ZONE_LAT],
                        "zone_lon": z[S.ZONE_LON],
                        "priority_score": z["priority_score"],
                        PRED_COL: round(float(z[PRED_COL]), 1),
                        "risk": z.get("risk", ""),
                        "route_method": "ortools_cvrp",
                    }
                )
                order += 1
            idx = solution.Value(routing.NextVar(idx))

    result = pd.DataFrame(rows)
    log.info(
        "Routed %d stops across %d teams (OR-Tools CVRP, cap %d/team)",
        len(result), result["team"].nunique() if len(result) else 0, cap,
    )
    return result
