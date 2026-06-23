from __future__ import annotations

import numpy as np
import pandas as pd

from parkflow import intelligence as I
from parkflow import schema as S
from parkflow.config import Config


def _ranked(n: int = 20) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            S.ZONE: [f"Z{i}" for i in range(n)],
            S.ZONE_LAT: 12.95 + rng.normal(0, 0.02, n),
            S.ZONE_LON: 77.59 + rng.normal(0, 0.02, n),
            "priority_score": np.linspace(100, 1, n),
            "predicted_violations": rng.uniform(0, 20, n),
            "risk": ["High"] * n,
        }
    )


def test_route_patrols_returns_valid_plan():
    cfg = Config.load()
    routes = I.route_patrols(_ranked(20), cfg)
    assert len(routes) > 0
    assert {"team", "stop_order", S.ZONE}.issubset(routes.columns)


def test_routes_respect_capacity_and_uniqueness():
    cfg = Config.load()
    routes = I.route_patrols(_ranked(20), cfg)
    method = str(routes["route_method"].iloc[0]) if "route_method" in routes else ""
    if method == "ortools_cvrp":
        per_team = routes.groupby("team").size()
        assert (per_team <= cfg.patrol.zones_per_team).all()
        assert routes[S.ZONE].is_unique  # no zone double-covered
    # Either way the plan never assigns more total stops than capacity allows.
    assert len(routes) <= cfg.patrol.num_teams * cfg.patrol.zones_per_team


def test_route_patrols_empty_input():
    cfg = Config.load()
    assert I.route_patrols(pd.DataFrame(), cfg).empty


def test_greedy_spread_respects_spacing_and_count():
    cfg = Config.load()
    picked = I.greedy_spread_zones(_ranked(20), 5, cfg.patrol.spatial_suppress_km)
    assert len(picked) <= 5
    # Every picked pair is at least the suppression distance apart.
    for i in range(len(picked)):
        for j in range(i + 1, len(picked)):
            d = I._haversine_km(
                picked.iloc[i][S.ZONE_LAT], picked.iloc[i][S.ZONE_LON],
                picked.iloc[j][S.ZONE_LAT], picked.iloc[j][S.ZONE_LON],
            )
            assert d >= cfg.patrol.spatial_suppress_km
