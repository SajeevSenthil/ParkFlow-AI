"""Stage 3-4 -- temporal aggregation, **complete-grid zero-fill**, and
leakage-safe lag/rolling features (PRD sections 5 step 8, and 7.3).

The zero-fill is the load-bearing correctness step: raw aggregation only yields
rows where violations happened, so a model trained on it never sees a quiet cell
and cannot predict lulls. We therefore materialise the full ``zone x bin``
Cartesian grid and fill absent counts with 0 *before* building any lag feature.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import schema as S
from .config import Config
from .logging_utils import get_logger

log = get_logger("features")

# Columns that are inputs to the model (everything else is id/target/metadata).
TIME_FEATURES = ["hour", "dayofweek", "month", "weekofyear", "is_weekend"]


@dataclass
class FeatureBundle:
    frame: pd.DataFrame          # full feature matrix incl. target + ids
    feature_cols: list[str]      # model input columns
    target_col: str = S.TARGET


def floor_to_bin(ts: pd.Series, bin_hours: int) -> pd.Series:
    """Floor a timestamp series to the start of its aggregation window."""
    return ts.dt.floor(f"{bin_hours}h")


def aggregate_counts(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Collapse events to (zone, bin_start) -> violation_count, with vehicle mix."""
    df = df.copy()
    df[S.BIN_START] = floor_to_bin(df[S.CREATED_DATETIME], cfg.temporal.bin_hours)
    counts = (
        df.groupby([S.ZONE, S.BIN_START]).size().rename(S.TARGET).reset_index()
    )
    return counts


def build_grid(counts: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Complete ``zone x bin`` grid, zero-filled. THE critical step."""
    zones = counts[S.ZONE].unique()
    bin_freq = f"{cfg.temporal.bin_hours}h"
    full_index = pd.date_range(
        counts[S.BIN_START].min(), counts[S.BIN_START].max(), freq=bin_freq
    )
    grid = pd.MultiIndex.from_product(
        [zones, full_index], names=[S.ZONE, S.BIN_START]
    ).to_frame(index=False)

    dense = grid.merge(counts, on=[S.ZONE, S.BIN_START], how="left")
    dense[S.TARGET] = dense[S.TARGET].fillna(0).astype("int64")

    fill_ratio = 1.0 - (len(counts) / len(dense)) if len(dense) else 0.0
    log.info(
        "Zero-filled grid: %d observed cells -> %d total cells (%.1f%% were implicit zeros)",
        len(counts),
        len(dense),
        100 * fill_ratio,
    )
    return dense.sort_values([S.ZONE, S.BIN_START]).reset_index(drop=True)


def add_time_features(dense: pd.DataFrame) -> pd.DataFrame:
    dense = dense.copy()
    ts = dense[S.BIN_START]
    dense["hour"] = ts.dt.hour
    dense["dayofweek"] = ts.dt.dayofweek
    dense["month"] = ts.dt.month
    dense["weekofyear"] = ts.dt.isocalendar().week.astype("int64")
    dense["is_weekend"] = (ts.dt.dayofweek >= 5).astype("int64")
    return dense


def add_lag_features(dense: pd.DataFrame, cfg: Config) -> tuple[pd.DataFrame, list[str]]:
    """Per-zone lag + rolling features. Computed on the time-ordered dense grid,
    using only past values (shift>=1), so no future information leaks in.
    """
    dense = dense.sort_values([S.ZONE, S.BIN_START]).copy()
    bpd = cfg.temporal.bins_per_day
    grp = dense.groupby(S.ZONE, sort=False)[S.TARGET]
    cols: list[str] = []

    # Previous-bin lag.
    dense["lag_1bin"] = grp.shift(1)
    cols.append("lag_1bin")

    # Same-bin previous-day / previous-week lags.
    for d in cfg.temporal.lag_days:
        name = f"lag_{d}d"
        dense[name] = grp.shift(d * bpd)
        cols.append(name)

    # Rolling means over trailing windows (shift(1) keeps them strictly past).
    past = grp.shift(1)
    for d in cfg.temporal.rolling_days:
        name = f"roll_{d}d_mean"
        window = d * bpd
        dense[name] = (
            past.groupby(dense[S.ZONE], sort=False)
            .rolling(window=window, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )
        cols.append(name)

    for c in cols:
        dense[c] = dense[c].fillna(0.0)
    return dense, cols


def add_zone_statics(
    dense: pd.DataFrame, train_mask: pd.Series, events: pd.DataFrame, cfg: Config
) -> tuple[pd.DataFrame, list[str]]:
    """Static per-zone features derived **from the training period only** to avoid
    leakage: historical frequency, frequency encoding, dominant police station,
    vehicle mix, and a junction flag.
    """
    dense = dense.copy()
    train_zone_bins = dense.loc[train_mask, [S.ZONE, S.BIN_START]]
    train_window = (train_zone_bins[S.BIN_START].min(), train_zone_bins[S.BIN_START].max())

    # Historical frequency: mean violations per bin per zone over the train window.
    hist = (
        dense.loc[train_mask]
        .groupby(S.ZONE)[S.TARGET]
        .mean()
        .rename("zone_hist_mean")
    )
    # Frequency encoding: share of all training violations at this zone.
    freq = (
        dense.loc[train_mask]
        .groupby(S.ZONE)[S.TARGET]
        .sum()
    )
    freq_enc = (freq / max(freq.sum(), 1)).rename("zone_freq_enc")

    statics = pd.concat([hist, freq_enc], axis=1).reset_index()

    # Junction flag + vehicle mix from the underlying events (train window only).
    ev = events[
        (events[S.CREATED_DATETIME] >= train_window[0])
        & (events[S.CREATED_DATETIME] <= train_window[1] + pd.Timedelta(hours=cfg.temporal.bin_hours))
    ]
    is_junction = (
        ev.assign(j=(ev[S.ZONE_KIND] == "junction").astype(int))
        .groupby(S.ZONE)["j"]
        .max()
        .rename("is_junction")
    )
    heavy = {
        "BUS", "BUS (BMTC/KSRTC)", "PRIVATE BUS", "LGV", "GOODS AUTO",
        "MAXI-CAB", "TEMPO", "VAN",
    }
    veh_share = (
        ev.assign(heavy=ev[S.VEHICLE_TYPE].astype(str).str.upper().isin(heavy).astype(int))
        .groupby(S.ZONE)["heavy"]
        .mean()
        .rename("zone_heavy_veh_share")
    )
    statics = (
        statics.merge(is_junction.reset_index(), on=S.ZONE, how="left")
        .merge(veh_share.reset_index(), on=S.ZONE, how="left")
    )

    dense = dense.merge(statics, on=S.ZONE, how="left")
    static_cols = ["zone_hist_mean", "zone_freq_enc", "is_junction", "zone_heavy_veh_share"]
    for c in static_cols:
        dense[c] = dense[c].fillna(0.0)
    return dense, static_cols


def temporal_split_mask(dense: pd.DataFrame, test_fraction: float) -> pd.Series:
    """Boolean mask: True = train. Split by time so test is strictly the future."""
    cutoff = dense[S.BIN_START].quantile(1.0 - test_fraction)
    return dense[S.BIN_START] <= cutoff


def build_features(events_with_zones: pd.DataFrame, cfg: Config) -> tuple[FeatureBundle, pd.Series]:
    """Full feature pipeline. Returns the bundle and the train mask."""
    counts = aggregate_counts(events_with_zones, cfg)
    dense = build_grid(counts, cfg)
    dense = add_time_features(dense)
    dense, lag_cols = add_lag_features(dense, cfg)

    train_mask = temporal_split_mask(dense, cfg.model.test_fraction)
    dense, static_cols = add_zone_statics(dense, train_mask, events_with_zones, cfg)

    feature_cols = TIME_FEATURES + lag_cols + static_cols
    log.info("Built %d features over %d grid rows", len(feature_cols), len(dense))
    return FeatureBundle(frame=dense, feature_cols=feature_cols), train_mask


def _split_feature_cols(feature_cols: list[str]) -> tuple[list[str], list[str]]:
    lag_cols = [c for c in feature_cols if c.startswith(("lag_", "roll_"))]
    static_cols = [c for c in feature_cols if c not in TIME_FEATURES and c not in lag_cols]
    return lag_cols, static_cols


def build_future_frame(
    dense: pd.DataFrame, feature_cols: list[str], cfg: Config
) -> tuple[pd.DataFrame, pd.Timestamp]:
    """One-step-ahead frame: the next bin for every zone, with lag/rolling/static
    features computed from observed history (no recursion, no leakage of future).
    """
    lag_cols, static_cols = _split_feature_cols(feature_cols)
    bpd = cfg.temporal.bins_per_day
    last_bin = dense[S.BIN_START].max()
    next_bin = last_bin + pd.Timedelta(hours=cfg.temporal.bin_hours)
    zones = dense[S.ZONE].unique()

    fut = pd.DataFrame({S.ZONE: zones, S.BIN_START: next_bin})
    fut = add_time_features(fut)

    target_lookup = dense.set_index([S.ZONE, S.BIN_START])[S.TARGET]

    def lookup(ts: pd.Timestamp) -> list[float]:
        idx = pd.MultiIndex.from_arrays([zones, [ts] * len(zones)])
        return target_lookup.reindex(idx).fillna(0.0).to_numpy(dtype=float)

    if "lag_1bin" in lag_cols:
        fut["lag_1bin"] = lookup(last_bin)
    for d in cfg.temporal.lag_days:
        name = f"lag_{d}d"
        if name in lag_cols:
            fut[name] = lookup(next_bin - pd.Timedelta(days=d))
    for d in cfg.temporal.rolling_days:
        name = f"roll_{d}d_mean"
        if name in lag_cols:
            window = d * bpd
            roll = dense.groupby(S.ZONE)[S.TARGET].apply(lambda s: s.tail(window).mean())
            fut[name] = fut[S.ZONE].map(roll)

    statics = dense.groupby(S.ZONE)[static_cols].first()
    fut = fut.merge(statics, on=S.ZONE, how="left")
    fut[feature_cols] = fut[feature_cols].fillna(0.0)
    return fut, next_bin
