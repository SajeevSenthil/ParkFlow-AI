from __future__ import annotations

import pandas as pd

from parkflow import displacement as D
from parkflow import schema as S
from parkflow.config import Config


def _line_forecast() -> pd.DataFrame:
    # Three near zones (covered cluster) + two far zones (uncovered).
    return pd.DataFrame(
        {
            S.ZONE: ["A", "B", "C", "D", "E"],
            S.ZONE_LAT: [12.950, 12.951, 12.952, 12.980, 12.990],
            S.ZONE_LON: [77.590, 77.590, 77.590, 77.590, 77.590],
            "predicted_violations": [10.0, 8.0, 6.0, 4.0, 2.0],
        }
    )


def test_conservation_out_equals_in_plus_suppressed():
    cfg = Config.load()
    stops = pd.DataFrame({"zone_lat": [12.950], "zone_lon": [77.590]})
    _, summary = D.simulate_displacement(_line_forecast(), stops, cfg)
    out = summary["displaced_out"]
    assert abs(out - (summary["relocated_in"] + summary["suppressed"])) < 1e-6


def test_covered_zone_sheds_exact_fraction():
    cfg = Config.load()
    stops = pd.DataFrame({"zone_lat": [12.950], "zone_lon": [77.590]})
    per_zone, _ = D.simulate_displacement(_line_forecast(), stops, cfg)
    a = per_zone.set_index(S.ZONE).loc["A"]
    assert bool(a["covered"]) is True
    assert abs(a["displaced_out"] - cfg.displacement.displaced_fraction * 10.0) < 1e-6


def test_displaced_out_bounded_by_prediction():
    cfg = Config.load()
    stops = pd.DataFrame({"zone_lat": [12.950], "zone_lon": [77.590]})
    per_zone, _ = D.simulate_displacement(_line_forecast(), stops, cfg)
    assert (per_zone["displaced_out"] >= 0).all()
    assert (per_zone["displaced_out"] <= per_zone["predicted_violations"] + 1e-9).all()


def test_relocation_to_near_uncovered_zone():
    cfg = Config.load()
    # Stop S; A is ~0.9 km from S (covered); B is ~1.3 km from S (uncovered) but
    # ~0.4 km from A (within the re-park radius) -> A's displaced share relocates to B.
    f = pd.DataFrame(
        {
            S.ZONE: ["A", "B"],
            S.ZONE_LAT: [12.9581, 12.9617],
            S.ZONE_LON: [77.5900, 77.5900],
            "predicted_violations": [10.0, 1.0],
        }
    )
    stops = pd.DataFrame({"zone_lat": [12.9500], "zone_lon": [77.5900]})
    per_zone, summary = D.simulate_displacement(f, stops, cfg)
    pz = per_zone.set_index(S.ZONE)
    assert bool(pz.loc["A", "covered"]) is True
    assert bool(pz.loc["B", "covered"]) is False
    assert summary["relocated_in"] > 0
    assert pz.loc["B", "displaced_in"] > 0


def test_compare_layouts_reports_both_leakages():
    cfg = Config.load()
    f = _line_forecast()
    naive = pd.DataFrame({S.ZONE: ["A"], "zone_lat": [12.950], "zone_lon": [77.590]})
    routed = pd.DataFrame({S.ZONE: ["C"], "zone_lat": [12.952], "zone_lon": [77.590]})
    cmp = D.compare_layouts(f, naive, routed, cfg)
    assert "naive_leakage" in cmp and "routed_leakage" in cmp
    assert "leakage_reduction_pct" in cmp
