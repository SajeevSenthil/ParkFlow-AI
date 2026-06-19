"""Seasonal-naive baseline (PRD 7.7).

Predicts each (zone, bin-of-week) cell as its mean over the training period.
This captures the dominant diurnal + weekly structure with zero ML, and is the
bar the XGBoost model must clear to justify its existence.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import schema as S
from .logging_utils import get_logger

log = get_logger("baseline")


class SeasonalNaiveBaseline:
    def __init__(self) -> None:
        self._table: pd.Series | None = None
        self._global_mean: float = 0.0

    @staticmethod
    def _bin_of_week(frame: pd.DataFrame) -> pd.Series:
        # Unique slot per (day-of-week, hour) = the seasonal key.
        return frame[S.BIN_START].dt.dayofweek * 24 + frame[S.BIN_START].dt.hour

    def fit(self, train: pd.DataFrame) -> "SeasonalNaiveBaseline":
        key = self._bin_of_week(train)
        self._table = train.groupby([train[S.ZONE], key])[S.TARGET].mean()
        self._global_mean = float(train[S.TARGET].mean())
        log.info("Baseline fitted on %d rows (global mean %.3f)", len(train), self._global_mean)
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        if self._table is None:
            raise RuntimeError("Baseline not fitted")
        key = self._bin_of_week(frame)
        idx = pd.MultiIndex.from_arrays([frame[S.ZONE].to_numpy(), key.to_numpy()])
        preds = self._table.reindex(idx).to_numpy(dtype=float)
        return np.where(np.isnan(preds), self._global_mean, preds)
