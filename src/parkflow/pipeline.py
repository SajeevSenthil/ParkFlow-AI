"""End-to-end orchestration (PRD section 11).

raw -> clean -> zones -> features(zero-fill) -> baseline+model -> evaluate
    -> one-step forecast -> risk/proxy/priority/patrol -> artifacts.

Artifacts are written to ``artifacts/`` so the Streamlit app only ever *reads*
precomputed outputs (no training at dashboard load).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from . import analytics as A
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


def run(cfg: Config) -> PipelineResult:
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
        },
    }
    metrics["model_beats_baseline"] = (
        metrics["model"]["mae"] < metrics["baseline"]["mae"]
    )
    _log_scorecard(metrics)

    # 6. One-step-ahead forecast for the next window ----------------------
    fut, next_bin = F.build_future_frame(dense, fcols, cfg)
    fut[I.PRED_COL] = model.predict(fut)

    # 7. Intelligence layers ---------------------------------------------
    fut = fut.merge(zmeta, on=S.ZONE, how="left")
    fut["risk"] = I.risk_band(fut[I.PRED_COL], cfg)
    fut = I.congestion_index(fut, events, cfg)
    ranked = I.enforcement_priority(fut, cfg)
    patrol_plan = I.allocate_patrols(ranked, cfg)

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
    write_table(zmeta, art / "zone_metadata.csv")
    write_table(model.feature_importance(), art / "feature_importance.csv")
    model.save(art / "model.joblib")
    write_json(metrics, art / "metrics.json")

    # SHAP explanations (PRD trust layer).
    write_table(shap_global, art / "shap_global.csv")
    write_table(shap_reasons, art / "shap_reasons.csv")

    # Supporting analytics (PRD deliverables 5 & 6).
    write_table(A.analytics_events(events), art / "events_analytics.parquet")
    write_table(A.junction_risk_table(events, ranked, cfg), art / "junction_risk.csv")
    top_offenders, zone_repeat = A.repeat_offender_tables(events)
    write_table(top_offenders, art / "top_offenders.csv")
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
