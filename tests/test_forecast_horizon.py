from __future__ import annotations

import pandas as pd

from parkflow import features as F
from parkflow import schema as S
from parkflow import spatial
from parkflow.config import Config
from parkflow.model import ViolationForecaster
from parkflow.preprocessing import clean

from conftest import make_raw


def _trained():
    cfg = Config.load()
    raw = make_raw(days=45, seed=3)
    events, _ = clean(raw, cfg)
    events = spatial.assign_zones(events, cfg)
    bundle, train_mask = F.build_features(events, cfg)
    model = ViolationForecaster.train(
        bundle.frame[train_mask], bundle.feature_cols, bundle.target_col, cfg
    )
    return cfg, bundle, model


def test_multi_horizon_shape_and_nonnegative():
    cfg, bundle, model = _trained()
    horizon = 6
    tl = F.build_multi_horizon_frames(bundle.frame, bundle.feature_cols, model, cfg, horizon=horizon)
    n_zones = bundle.frame[S.ZONE].nunique()
    # One row per (zone, horizon).
    assert len(tl) == horizon * n_zones
    assert set(tl["horizon"]) == set(range(1, horizon + 1))
    # Counts are clipped at zero.
    assert (tl["predicted_violations"] >= 0).all()


def test_horizon_bins_strictly_increase():
    cfg, bundle, model = _trained()
    tl = F.build_multi_horizon_frames(bundle.frame, bundle.feature_cols, model, cfg, horizon=4)
    bins = sorted(pd.to_datetime(tl[S.BIN_START]).unique())
    assert len(bins) == 4
    for earlier, later in zip(bins, bins[1:]):
        assert later > earlier


def test_live_anchor_relabels_timeline():
    cfg, bundle, model = _trained()
    anchor = pd.Timestamp("2030-01-01 09:00")
    tl = F.build_multi_horizon_frames(
        bundle.frame, bundle.feature_cols, model, cfg, horizon=3, anchor=anchor
    )
    first_bin = sorted(pd.to_datetime(tl[S.BIN_START]).unique())[0]
    # Bin 1 is relabelled to the anchor (floored to the bin grid).
    assert pd.Timestamp(first_bin) == anchor.floor(f"{cfg.temporal.bin_hours}h")
