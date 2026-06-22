from __future__ import annotations

import pandas as pd

from parkflow import economics as E
from parkflow import schema as S
from parkflow.config import Config


def _frame(preds: list[float], cap_red: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            S.ZONE: [f"Z{i}" for i in range(len(preds))],
            "predicted_violations": preds,
            "est_capacity_reduction_pct": cap_red,
        }
    )


def test_cost_monotonic_in_violations():
    cfg = Config.load()
    out = E.economic_impact(_frame([2.0, 20.0], [30.0, 30.0]), cfg).set_index(S.ZONE)
    # Same capacity reduction, more violations -> strictly higher economic cost.
    assert out.loc["Z1", "economic_cost_inr"] > out.loc["Z0", "economic_cost_inr"]


def test_zero_capacity_reduction_means_zero_cost():
    cfg = Config.load()
    out = E.economic_impact(_frame([10.0, 10.0], [0.0, 50.0]), cfg).set_index(S.ZONE)
    assert out.loc["Z0", "economic_cost_inr"] == 0.0
    assert out.loc["Z1", "economic_cost_inr"] > 0.0


def test_summary_totals_match_rows():
    cfg = Config.load()
    out = E.economic_impact(_frame([5.0, 8.0], [20.0, 40.0]), cfg)
    s = E.economic_summary(out)
    assert s["total_cost_inr"] == out["economic_cost_inr"].sum()
    assert s["zone_windows"] == 2
    # Worst zone is the one with the larger cost.
    assert s["top_zone"] == out.loc[out["economic_cost_inr"].idxmax(), S.ZONE]


def test_requires_capacity_reduction_column():
    cfg = Config.load()
    bad = pd.DataFrame({S.ZONE: ["A"], "predicted_violations": [5.0]})
    try:
        E.economic_impact(bad, cfg)
        assert False, "expected KeyError when est_capacity_reduction_pct is absent"
    except KeyError:
        pass
