"""End-to-end orchestration (PRD section 11).

raw -> clean -> zones -> features(zero-fill) -> baseline+model -> evaluate
    -> one-step forecast + rolling 24h timeline
    -> risk/congestion/economic/priority/patrol(greedy + OR-Tools route)/displacement
    -> artifacts.

Artifacts are written to ``artifacts/`` so the Streamlit app only ever *reads*
precomputed outputs (no training at dashboard load). The single exception is the
operator deployment log (``enforcement_log.db``), which the dashboard writes to.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from . import analytics as A
from . import displacement as D
from . import economics as E
from . import features as F
from . import intelligence as I
from . import schema as S
from . import spatial
from .baseline import SeasonalNaiveBaseline
from .config import Config
from .evaluation import ranking_metrics, regression_metrics
from .explain import explain_forecast
from .io import read_table, write_json, write_table
from .logging_utils import get_logger
from .model import ViolationForecaster
from .preprocessing import clean

log = get_logger("pipeline")


@dataclass
class PipelineResult:
    metrics: dict
    artifacts_dir: Path


def run(cfg: Config, horizon: int | None = None, live: bool = False) -> PipelineResult:
    cfg.paths.ensure_dirs()
    art = cfg.paths.artifacts_dir

    # 1. Load -------------------------------------------------------------
    raw_path = cfg.paths.raw_data
    if not raw_path.exists():
        raise FileNotFoundError(
            f"Raw data not found at {raw_path}. Set paths.raw_data in config/config.yaml "
            "to the violations CSV."
        )
    raw = read_table(raw_path)
    log.info("Loaded %d raw rows from %s", len(raw), raw_path.name)

    # 2. Clean ------------------------------------------------------------
    events, clean_stats = clean(raw, cfg)

    # 3. Spatial zones ----------------------------------------------------
    events = spatial.assign_zones(events, cfg)
    zmeta = spatial.zone_metadata(events)

    # 4. Features (with zero-fill) + temporal split -----------------------
    bundle, train_mask = F.build_features(events, cfg)
    dense = bundle.frame
    fcols, target = bundle.feature_cols, bundle.target_col
    train_df, test_df = dense[train_mask], dense[~train_mask]
    log.info("Temporal split: %d train / %d test rows", len(train_df), len(test_df))

    # 5. Baseline vs model ------------------------------------------------
    baseline = SeasonalNaiveBaseline().fit(train_df)
    base_test_pred = baseline.predict(test_df)

    model = ViolationForecaster.train(train_df, fcols, target, cfg)
    model_test_pred = model.predict(test_df)

    y_test = test_df[target].to_numpy()
    rank_frame = test_df[[S.ZONE, S.BIN_START]].copy()
    rank_frame["y"] = y_test
    rank_frame["model"] = model_test_pred
    rank_frame["baseline"] = base_test_pred
    thr, k = cfg.evaluation.hotspot_threshold, cfg.evaluation.top_k
    metrics = {
        "rows": {"train": int(len(train_df)), "test": int(len(test_df))},
        "baseline": regression_metrics(y_test, base_test_pred),
        "model": regression_metrics(y_test, model_test_pred),
        "ranking": {
            "baseline": ranking_metrics(rank_frame, S.BIN_START, "y", "baseline", thr, k),
            "model": ranking_metrics(rank_frame, S.BIN_START, "y", "model", thr, k),
        },
        "clean_stats": clean_stats.as_dict(),
        "config": {
            "bin_hours": cfg.temporal.bin_hours,
            "test_fraction": cfg.model.test_fraction,
            "objective": cfg.model.objective,
            "hotspot_threshold": thr,
            "top_k": k,
            "zones_per_team": cfg.patrol.zones_per_team,
        },
        "pipeline_run_at": pd.Timestamp.now().isoformat(),
        "data_date_range": {
            "from": str(events[S.CREATED_DATETIME].min()),
            "to": str(events[S.CREATED_DATETIME].max()),
        },
    }
    metrics["model_beats_baseline"] = (
        metrics["model"]["mae"] < metrics["baseline"]["mae"]
    )
    _log_scorecard(metrics)

    # 6. One-step-ahead forecast for the next window ----------------------
    fut, next_bin = F.build_future_frame(dense, fcols, cfg)
    fut[I.PRED_COL] = model.predict(fut)

    # 6b. Rolling multi-horizon forecast (the next 24h, recursive) --------
    horizon = horizon or cfg.temporal.forecast_horizon_bins
    anchor = pd.Timestamp.now() if live else None
    timeline = F.build_multi_horizon_frames(dense, fcols, model, cfg, horizon, anchor)
    timeline = timeline.merge(zmeta, on=S.ZONE, how="left")
    timeline["risk"] = I.risk_band(timeline[I.PRED_COL], cfg)
    timeline = I.congestion_index(timeline, events, cfg)
    timeline = E.economic_impact(timeline, cfg)

    # 7. Intelligence layers (decision target = next window) --------------
    fut = fut.merge(zmeta, on=S.ZONE, how="left")
    fut["risk"] = I.risk_band(fut[I.PRED_COL], cfg)
    fut = I.congestion_index(fut, events, cfg)
    fut = E.economic_impact(fut, cfg)
    ranked = I.enforcement_priority(fut, cfg)
    patrol_plan = I.allocate_patrols(ranked, cfg)      # greedy: one stop per team
    patrol_routes = I.route_patrols(ranked, cfg)       # OR-Tools CVRP: ordered routes

    # 7b. Displacement: behavioural response + naive-vs-routed leakage -----
    # The recommended (routed) layout drives the displacement artifact. The comparison
    # is kept fair by pitting the routes against a NAIVE layout of the *same* number of
    # stops (top-K purely by priority, no spatial reasoning) so any leakage difference
    # reflects the spatial arrangement, not the headcount.
    disp_zones, disp_summary = D.simulate_displacement(ranked, patrol_routes, cfg)
    k_stops = cfg.patrol.num_teams * cfg.patrol.zones_per_team
    naive_stops = I.greedy_spread_zones(ranked, k_stops, cfg.patrol.spatial_suppress_km)
    layout_cmp = D.compare_layouts(ranked, naive_stops, patrol_routes, cfg)

    # Per-zone 24h economic rollup for the Economic Impact tab.
    econ_zone = (
        timeline.groupby(S.ZONE)
        .agg(
            predicted_violations=(I.PRED_COL, "sum"),
            vehicle_hours_delay=("vehicle_hours_delay", "sum"),
            economic_cost_inr=("economic_cost_inr", "sum"),
        )
        .reset_index()
        .merge(zmeta, on=S.ZONE, how="left")
        .sort_values("economic_cost_inr", ascending=False)
        .reset_index(drop=True)
    )

    metrics["economic_summary"] = E.economic_summary(timeline)
    metrics["displacement_summary"] = {**disp_summary, **layout_cmp}
    metrics["live_mode"] = bool(live)
    metrics["forecast_horizon_bins"] = int(horizon)

    # SHAP explanations for the forecast ("why this zone?") -- PRD trust layer.
    shap_global, shap_reasons = explain_forecast(model, fut)

    # Current hotspots = historical density per zone (PRD 8.3).
    current_hotspots = (
        events.groupby(S.ZONE)
        .size()
        .rename("historical_violations")
        .reset_index()
        .merge(zmeta, on=S.ZONE, how="left")
        .sort_values("historical_violations", ascending=False)
        .reset_index(drop=True)
    )

    # 8. Persist artifacts ------------------------------------------------
    forecast_cols = [
        S.ZONE, S.ZONE_KIND, S.BIN_START, S.ZONE_LAT, S.ZONE_LON,
        I.PRED_COL, "risk", "priority_score",
        "congestion_index", "est_capacity_reduction_pct", "pcu_load",
    ]
    ranked["forecast_window_start"] = next_bin
    write_table(ranked[[c for c in forecast_cols if c in ranked] + ["forecast_window_start"]],
                art / "future_forecast.csv")
    write_table(patrol_plan, art / "patrol_plan.csv")
    write_table(current_hotspots, art / "current_hotspots.csv")

    # Multi-horizon timeline + economic + routing + displacement artifacts.
    timeline_cols = [
        S.ZONE, S.ZONE_KIND, S.BIN_START, "horizon", S.ZONE_LAT, S.ZONE_LON,
        I.PRED_COL, "risk", "congestion_index", "est_capacity_reduction_pct",
        "economic_cost_inr", "vehicle_hours_delay",
    ]
    write_table(timeline[[c for c in timeline_cols if c in timeline]], art / "forecast_timeline.csv")
    write_table(econ_zone, art / "economic_impact.csv")
    write_table(patrol_routes, art / "patrol_routes.csv")
    disp_cols = [
        S.ZONE, S.ZONE_LAT, S.ZONE_LON, I.PRED_COL, "covered",
        "displaced_out", "displaced_in", "residual_predicted",
    ]
    write_table(disp_zones[[c for c in disp_cols if c in disp_zones]], art / "displacement.csv")
    write_table(zmeta, art / "zone_metadata.csv")
    write_table(model.feature_importance(), art / "feature_importance.csv")
    model.save(art / "model.joblib")
    write_json(metrics, art / "metrics.json")

    # SHAP explanations (PRD trust layer).
    write_table(shap_global, art / "shap_global.csv")
    write_table(shap_reasons, art / "shap_reasons.csv")

    # Test-set predictions for the model diagnostic chart in the dashboard.
    test_preds = test_df[[S.ZONE, S.BIN_START, target]].copy()
    test_preds["predicted"] = model_test_pred
    test_preds["error"] = test_preds["predicted"] - test_preds[target]
    write_table(test_preds, art / "test_predictions.csv")

    # Supporting analytics (PRD deliverables 5 & 6).
    write_table(A.analytics_events(events), art / "events_analytics.parquet")
    write_table(A.junction_risk_table(events, ranked, cfg), art / "junction_risk.csv")

    # Repeat-offender analytics: canonical vehicle list (used by the dashboard tab).
    if S.VEHICLE_NUMBER in events.columns:
        rep_threshold = cfg.congestion.repeat_offender_threshold
        top_zone = events.groupby(S.VEHICLE_NUMBER)[S.ZONE].agg(
            lambda s: s.value_counts().idxmax()
        )
        repeat_offenders = (
            events.groupby(S.VEHICLE_NUMBER)
            .agg(
                violation_count=(S.ZONE, "count"),
                unique_zones=(S.ZONE, "nunique"),
                vehicle_type=(S.VEHICLE_TYPE, "first"),
                last_seen=(S.CREATED_DATETIME, "max"),
            )
            .query(f"violation_count > {rep_threshold}")
            .join(top_zone.rename("top_zone"))
            .sort_values("violation_count", ascending=False)
            .reset_index()
        )
        write_table(repeat_offenders, art / "repeat_offenders.csv")
        log.info(
            "Repeat offenders: %d vehicles with > %d violations",
            len(repeat_offenders),
            rep_threshold,
        )

    # Per-zone willful-vs-infrastructure signal (repeat-offender share per zone).
    _top_off, zone_repeat = A.repeat_offender_tables(events)
    write_table(zone_repeat, art / "zone_repeat_offenders.csv")

    log.info("Artifacts written to %s", art)

    return PipelineResult(metrics=metrics, artifacts_dir=art)


def _log_scorecard(m: dict) -> None:
    b, md = m["baseline"], m["model"]
    log.info("--- Evaluation (held-out future) ---")
    log.info("              %8s %8s", "BASELINE", "MODEL")
    for k in ("mae", "rmse", "r2", "poisson_deviance"):
        log.info("  %-18s %8.3f %8.3f", k, b[k], md[k])
    rb, rm = m["ranking"]["baseline"], m["ranking"]["model"]
    for k in rb:
        log.info("  %-18s %8.3f %8.3f", k, rb[k], rm[k])
    verdict = "PASS" if m["model_beats_baseline"] else "FAIL (model <= baseline)"
    log.info("  model beats baseline (MAE): %s", verdict)
