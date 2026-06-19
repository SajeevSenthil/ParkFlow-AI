"""Lightweight IO helpers (CSV/Parquet with a CSV fallback, JSON metrics)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .logging_utils import get_logger

log = get_logger("io")


def read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def write_table(df: pd.DataFrame, path: str | Path) -> Path:
    """Write parquet if the engine is available, else fall back to CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".parquet":
        try:
            df.to_parquet(path, index=False)
            return path
        except (ImportError, ValueError):
            path = path.with_suffix(".csv")
            log.warning("Parquet engine unavailable; writing CSV to %s", path.name)
    df.to_csv(path, index=False)
    return path


def write_json(obj: dict[str, Any], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, default=str)
    return path
