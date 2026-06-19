from __future__ import annotations

import numpy as np
import pandas as pd

from parkflow import features as F
from parkflow import schema as S
from parkflow import spatial
from parkflow.config import Config
from parkflow.preprocessing import clean

from conftest import make_raw


def _prep():
    cfg = Config.load()
    raw = make_raw(days=30, seed=7)
    events, _ = clean(raw, cfg)
    events = spatial.assign_zones(events, cfg)
    return cfg, events


def test_grid_is_complete_and_zero_filled():
    cfg, events = _prep()
    counts = F.aggregate_counts(events, cfg)
    dense = F.build_grid(counts, cfg)

    n_zones = dense[S.ZONE].nunique()
    n_bins = dense[S.BIN_START].nunique()
    # Complete cartesian product: every zone has every bin.
    assert len(dense) == n_zones * n_bins
    # Zero-fill introduced explicit zeros that raw aggregation lacked.
    assert (dense[S.TARGET] == 0).any()
    assert len(dense) > len(counts)
    assert dense[S.TARGET].isna().sum() == 0


def test_lag_features_have_no_future_leakage():
    cfg, events = _prep()
    bundle, train_mask = F.build_features(events, cfg)
    d = bundle.frame.sort_values([S.ZONE, S.BIN_START]).reset_index(drop=True)

    # For each zone, lag_1bin at row t must equal target at row t-1.
    for _, g in d.groupby(S.ZONE):
        g = g.reset_index(drop=True)
        if len(g) < 3:
            continue
        expected = g[S.TARGET].shift(1).fillna(0.0).to_numpy()
        assert np.allclose(g["lag_1bin"].to_numpy(), expected)
        break  # one zone is enough to assert the contract


def test_future_frame_one_row_per_zone():
    cfg, events = _prep()
    bundle, _ = F.build_features(events, cfg)
    fut, next_bin = F.build_future_frame(bundle.frame, bundle.feature_cols, cfg)

    assert len(fut) == bundle.frame[S.ZONE].nunique()
    assert (fut[S.BIN_START] == next_bin).all()
    # All model features present and finite.
    assert not fut[bundle.feature_cols].isna().any().any()
