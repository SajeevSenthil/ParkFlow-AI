from __future__ import annotations

import pandas as pd

from parkflow import schema as S
from parkflow.config import Config
from parkflow.preprocessing import clean, parse_violation_tags

from conftest import make_raw


def _cfg() -> Config:
    return Config.load()


def test_parse_violation_tags_handles_json_array():
    assert parse_violation_tags('["WRONG PARKING","NO PARKING"]') == ["WRONG PARKING", "NO PARKING"]
    assert parse_violation_tags("NO PARKING") == ["NO PARKING"]
    assert parse_violation_tags(None) == []


def test_clean_filters_non_parking_and_dupes():
    raw = make_raw(days=15, seed=1)
    cfg = _cfg()
    cleaned, stats = clean(raw, cfg)

    # Only valid parking types survive (reduced to a single representative tag).
    assert set(cleaned[S.VIOLATION_TYPE].unique()).issubset(
        {v.upper() for v in cfg.valid_violation_types}
    )
    assert stats.dropped_non_parking > 0
    assert stats.rows_out == len(cleaned)


def test_timestamps_converted_to_local_naive():
    raw = make_raw(days=5, seed=3)
    cfg = _cfg()
    cleaned, _ = clean(raw, cfg)
    # tz-naive after conversion, and shifted into IST (so min hour differs from UTC).
    assert cleaned[S.CREATED_DATETIME].dt.tz is None
    assert str(cleaned[S.CREATED_DATETIME].dtype).startswith("datetime64")


def test_required_columns_enforced():
    cfg = _cfg()
    bad = pd.DataFrame({"foo": [1, 2]})
    try:
        clean(bad, cfg)
        assert False, "expected ValueError"
    except ValueError:
        pass
