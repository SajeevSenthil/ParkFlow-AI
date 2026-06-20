from __future__ import annotations

import numpy as np
import pandas as pd

from parkflow import intelligence as I
from parkflow import schema as S
from parkflow.config import Config
from parkflow.evaluation import hotspot_pr_auc, top_k_hit_rate


def _events():
    # Two zones: one with heavy vehicles (buses), one with two-wheelers.
    return pd.DataFrame(
        {
            S.ZONE: ["A", "A", "B", "B"],
            S.ZONE_KIND: ["junction", "junction", "junction", "junction"],
            S.VEHICLE_TYPE: ["BUS", "BUS", "SCOOTER", "SCOOTER"],
        }
    )


def test_congestion_index_higher_for_heavier_vehicles():
    cfg = Config.load()
    zone_frame = pd.DataFrame(
        {
            S.ZONE: ["A", "B"],
            S.ZONE_KIND: ["junction", "junction"],
            I.PRED_COL: [10.0, 10.0],  # same predicted volume
        }
    )
    out = I.congestion_index(zone_frame, _events(), cfg)
    a = out.loc[out[S.ZONE] == "A", "congestion_index"].iloc[0]
    b = out.loc[out[S.ZONE] == "B", "congestion_index"].iloc[0]
    # Same count, but buses (PCU 3.5) should outrank scooters (PCU 0.5).
    assert a > b
    # Index is bounded 0-100, capacity reduction within the configured cap.
    assert 0 <= b <= a <= 100
    assert out["est_capacity_reduction_pct"].max() <= cfg.congestion.max_capacity_reduction_pct + 1e-6


def test_congestion_index_monotonic_in_volume():
    cfg = Config.load()
    zf = pd.DataFrame(
        {S.ZONE: ["A", "A2"], S.ZONE_KIND: ["junction", "junction"], I.PRED_COL: [2.0, 20.0]}
    )
    ev = pd.DataFrame({S.ZONE: ["A", "A2"], S.ZONE_KIND: ["junction"] * 2, S.VEHICLE_TYPE: ["CAR", "CAR"]})
    out = I.congestion_index(zf, ev, cfg).set_index(S.ZONE)
    assert out.loc["A2", "congestion_index"] > out.loc["A", "congestion_index"]


def test_top_k_hit_rate_perfect_and_pr_auc():
    # Perfect ranking within a single time slice -> hit rate 1.0
    frame = pd.DataFrame(
        {"t": [0] * 5, "y": [5, 4, 3, 2, 1], "pred": [50, 40, 30, 20, 10]}
    )
    assert top_k_hit_rate(frame, "t", "y", "pred", k=2) == 1.0
    # PR-AUC: perfect separation -> 1.0
    auc = hotspot_pr_auc(np.array([0, 0, 5, 5]), np.array([0.1, 0.2, 0.9, 0.8]), threshold=5)
    assert auc == 1.0
