"""Stage 1 -- cleaning (PRD section 5, steps 1-5 / data_preprocessing.md steps 1-4).

Each step is a pure function returning a new frame plus a small stats dict, so the
pipeline can log exactly what was removed and why.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field

import pandas as pd

from . import schema as S
from .config import Config
from .logging_utils import get_logger

log = get_logger("preprocessing")


def parse_violation_tags(value: object) -> list[str]:
    """Parse a raw ``violation_type`` cell into a list of upper-cased tags.

    Real data stores this as a JSON-array string, e.g.
    ``'["WRONG PARKING","PARKING NEAR ROAD CROSSING"]'``. Falls back to treating
    the value as a single tag if it is not a parseable list.
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    if isinstance(value, list):
        items = value
    else:
        s = str(value).strip()
        if s.startswith("[") and s.endswith("]"):
            try:
                items = ast.literal_eval(s)
            except (ValueError, SyntaxError):
                items = [s]
        else:
            items = [s]
    return [str(t).strip().upper() for t in items if str(t).strip()]


@dataclass
class CleanStats:
    rows_in: int = 0
    rows_out: int = 0
    dropped_non_parking: int = 0
    dropped_duplicates: int = 0
    dropped_no_timestamp: int = 0
    filled_junction: int = 0
    filled_station: int = 0
    details: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


def clean(df: pd.DataFrame, cfg: Config) -> tuple[pd.DataFrame, CleanStats]:
    """Run the full cleaning sequence and return (clean_df, stats)."""
    stats = CleanStats(rows_in=len(df))
    df = df.copy()

    _require_columns(df)

    # Step 1: drop admin columns that exist (kept-for-analytics handled by caller).
    drop_cols = [c for c in S.ADMIN_COLUMNS if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)
        stats.details["dropped_admin_columns"] = drop_cols

    # Parse timestamp early (raw is UTC, tz-aware). Convert to local time so that
    # hour-of-day features reflect local commute patterns, then drop tz info.
    ts = pd.to_datetime(df[S.CREATED_DATETIME], errors="coerce", utc=True)
    ts = ts.dt.tz_convert(cfg.temporal.timezone).dt.tz_localize(None)
    df[S.CREATED_DATETIME] = ts
    before = len(df)
    df = df.dropna(subset=[S.CREATED_DATETIME])
    stats.dropped_no_timestamp = before - len(df)

    # Step 2: impute key categoricals.
    if S.JUNCTION_NAME in df.columns:
        missing = df[S.JUNCTION_NAME].isna() | (df[S.JUNCTION_NAME].astype(str).str.strip() == "")
        stats.filled_junction = int(missing.sum())
        df.loc[missing, S.JUNCTION_NAME] = cfg.spatial.missing_junction_label
    if S.POLICE_STATION in df.columns:
        missing = df[S.POLICE_STATION].isna() | (df[S.POLICE_STATION].astype(str).str.strip() == "")
        stats.filled_station = int(missing.sum())
        df.loc[missing, S.POLICE_STATION] = cfg.spatial.missing_station_label

    # Step 3: keep only genuine parking violations. violation_type is a JSON array;
    # keep a record if ANY tag is a parking type, and reduce it to a single
    # representative `violation_type` (the first matching parking tag) for analytics.
    valid = {v.upper() for v in cfg.valid_violation_types}
    tags = df[S.VIOLATION_TYPE].map(parse_violation_tags)
    primary = tags.map(lambda ts_: next((t for t in ts_ if t in valid), None))
    before = len(df)
    keep = primary.notna()
    df = df[keep]
    df[S.VIOLATION_TYPE] = primary[keep]
    stats.dropped_non_parking = before - len(df)

    # Step 3b: filter by validation status when configured.
    # "approved_only" keeps only BTP-confirmed violations to remove false positives.
    # By default, only hardcoded bad statuses (rejected / duplicate) are excluded.
    if S.VALIDATION_STATUS in df.columns:
        before = len(df)
        if cfg.preprocessing.approved_only:
            df = df[df[S.VALIDATION_STATUS].astype(str).str.lower() == "approved"]
        else:
            excluded = {s.lower() for s in cfg.preprocessing.exclude_statuses}
            df = df[~df[S.VALIDATION_STATUS].astype(str).str.lower().isin(excluded)]
        stats.details["dropped_by_validation_status"] = before - len(df)
        log.info(
            "Validation filter: kept %d / %d rows (approved_only=%s)",
            len(df),
            before,
            cfg.preprocessing.approved_only,
        )

    # Step 4: drop duplicate captures of the same event.
    dedupe_keys = [
        c
        for c in (S.VEHICLE_NUMBER, S.LATITUDE, S.LONGITUDE, S.CREATED_DATETIME)
        if c in df.columns
    ]
    before = len(df)
    df = df.drop_duplicates(subset=dedupe_keys, keep="first")
    stats.dropped_duplicates = before - len(df)

    df = df.reset_index(drop=True)
    stats.rows_out = len(df)
    log.info(
        "Cleaned %d -> %d rows (non-parking -%d, dupes -%d, no-ts -%d)",
        stats.rows_in,
        stats.rows_out,
        stats.dropped_non_parking,
        stats.dropped_duplicates,
        stats.dropped_no_timestamp,
    )
    if stats.rows_out == 0:
        raise ValueError("Cleaning removed all rows; check violation-type filter and schema.")
    return df, stats


def _require_columns(df: pd.DataFrame) -> None:
    missing = [c for c in S.REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Input is missing required columns {missing}. Present: {list(df.columns)}"
        )
