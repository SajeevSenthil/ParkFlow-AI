"""Spatial displacement simulation (PRD section 8.7) — the behavioural-response layer.

Almost every team in the hackathon predicts violations and deploys patrols. What almost
none model is the *reaction*: when enforcement covers a zone, offenders do not vanish —
a share of them re-park in the nearest uncovered zone instead. Ignoring this creates
"displacement blindspots" where the problem silently moves one block over.

This module simulates that response and lets us show a concrete, novel result: a
route-optimized patrol layout (OR-Tools CVRP) leaks fewer violations into blindspots
than a naive greedy layout. It is deterministic business logic (no ML, no external data),
reusing the same haversine helper as the patrol layer.
"""

from __future__ import annotations

import pandas as pd

from . import schema as S
from .config import Config
from .intelligence import PRED_COL, _haversine_km
from .logging_utils import get_logger

log = get_logger("displacement")


def covered_zone_set(forecast: pd.DataFrame, stops: pd.DataFrame, radius_km: float) -> set:
    """Zones whose centroid lies within ``radius_km`` of any patrol stop."""
    if stops is None or stops.empty:
        return set()
    stop_pts = stops.dropna(subset=["zone_lat", "zone_lon"])[["zone_lat", "zone_lon"]].to_numpy()
    covered: set = set()
    for _, z in forecast.dropna(subset=[S.ZONE_LAT, S.ZONE_LON]).iterrows():
        for slat, slon in stop_pts:
            if _haversine_km(z[S.ZONE_LAT], z[S.ZONE_LON], slat, slon) <= radius_km:
                covered.add(z[S.ZONE])
                break
    return covered


def simulate_displacement(
    forecast: pd.DataFrame, stops: pd.DataFrame, cfg: Config
) -> tuple[pd.DataFrame, dict]:
    """Simulate offender displacement for one patrol layout.

    For each *covered* zone, ``displaced_fraction`` of its predicted violations relocate
    to the nearest *uncovered* zone within ``displacement_radius_km``. If no uncovered
    zone is in range the displaced share is treated as genuinely *suppressed* (enforcement
    worked). Conservation therefore reads: displaced_out == relocated_in + suppressed.

    Returns ``(per_zone_df, summary)``. ``per_zone_df`` carries
    ``covered, displaced_out, displaced_in, residual_predicted`` per zone.
    """
    d = cfg.displacement
    f = forecast.dropna(subset=[S.ZONE_LAT, S.ZONE_LON]).copy()
    covered = covered_zone_set(f, stops, d.coverage_radius_km)

    f["covered"] = f[S.ZONE].isin(covered)
    f["displaced_out"] = 0.0
    f["displaced_in"] = 0.0

    uncovered = f[~f["covered"]].reset_index(drop=True)
    uncov_pts = uncovered[[S.ZONE, S.ZONE_LAT, S.ZONE_LON]].to_numpy()
    in_accum: dict = {}
    suppressed = 0.0

    for i, z in f[f["covered"]].iterrows():
        out = d.displaced_fraction * float(z[PRED_COL])
        if out <= 0:
            continue
        f.at[i, "displaced_out"] = round(out, 2)
        # Nearest uncovered zone within the re-park radius.
        nearest_zone, nearest_d = None, float("inf")
        for uz, ulat, ulon in uncov_pts:
            dist = _haversine_km(z[S.ZONE_LAT], z[S.ZONE_LON], ulat, ulon)
            if dist < nearest_d:
                nearest_zone, nearest_d = uz, dist
        if nearest_zone is not None and nearest_d <= d.displacement_radius_km:
            in_accum[nearest_zone] = in_accum.get(nearest_zone, 0.0) + out
        else:
            suppressed += out

    if in_accum:
        f["displaced_in"] = f[S.ZONE].map(in_accum).fillna(0.0).round(2)

    f["residual_predicted"] = (
        f[PRED_COL] - f["displaced_out"] + f["displaced_in"]
    ).clip(lower=0.0).round(2)

    relocated = float(f["displaced_in"].sum())
    displaced_out = float(f["displaced_out"].sum())
    summary = {
        "total_predicted": round(float(f[PRED_COL].sum()), 1),
        "displaced_out": round(displaced_out, 1),
        "relocated_in": round(relocated, 1),
        "suppressed": round(suppressed, 1),
        "n_covered": int(f["covered"].sum()),
        "n_blindspots": int((f["displaced_in"] > 0).sum()),
        # Leakage = violations that simply moved into an unwatched zone despite enforcement.
        "leakage": round(relocated, 1),
    }
    log.info(
        "Displacement: %.1f displaced -> %.1f relocated into %d blindspots, %.1f suppressed",
        displaced_out, relocated, summary["n_blindspots"], suppressed,
    )
    return f, summary


def compare_layouts(
    forecast: pd.DataFrame, naive_stops: pd.DataFrame, routed_stops: pd.DataFrame, cfg: Config
) -> dict:
    """Headline comparison of two *same-size* patrol layouts: how much each leaks into
    blindspots. ``naive_stops`` = top-K zones by priority (no spatial reasoning);
    ``routed_stops`` = the OR-Tools route-optimized layout. Reported honestly — the
    reduction can be negative if the naive layout happens to leak less on this data.
    """
    _, naive = simulate_displacement(forecast, naive_stops, cfg)
    _, routed = simulate_displacement(forecast, routed_stops, cfg)
    n_leak, r_leak = naive.get("leakage", 0.0), routed.get("leakage", 0.0)
    reduction_pct = round(100.0 * (n_leak - r_leak) / n_leak, 1) if n_leak > 0 else 0.0
    return {
        "naive_leakage": n_leak,
        "routed_leakage": r_leak,
        "leakage_reduction_pct": reduction_pct,
        "naive_n_covered": naive.get("n_covered", 0),
        "routed_n_covered": routed.get("n_covered", 0),
    }
