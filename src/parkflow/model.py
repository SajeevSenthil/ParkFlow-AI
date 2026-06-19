"""XGBoost violation-count forecaster (PRD 7.5).

Thin wrapper around ``XGBRegressor`` with a Poisson objective, count-appropriate
prediction clipping, and joblib persistence. Keeps the feature list bound to the
model so inference can't silently use the wrong columns.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from .config import Config
from .logging_utils import get_logger

log = get_logger("model")


@dataclass
class ViolationForecaster:
    feature_cols: list[str]
    model: XGBRegressor

    @classmethod
    def train(
        cls,
        train_df: pd.DataFrame,
        feature_cols: list[str],
        target_col: str,
        cfg: Config,
    ) -> "ViolationForecaster":
        model = XGBRegressor(**cfg.model.xgb_params())
        model.fit(train_df[feature_cols], train_df[target_col])
        log.info("Trained XGBoost (%s) on %d rows", cfg.model.objective, len(train_df))
        return cls(feature_cols=feature_cols, model=model)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        preds = self.model.predict(df[self.feature_cols])
        return np.clip(preds, 0.0, None)

    def feature_importance(self, top: int = 15) -> pd.DataFrame:
        imp = pd.DataFrame(
            {"feature": self.feature_cols, "importance": self.model.feature_importances_}
        )
        return imp.sort_values("importance", ascending=False).head(top).reset_index(drop=True)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"feature_cols": self.feature_cols, "model": self.model}, path)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "ViolationForecaster":
        blob = joblib.load(path)
        return cls(feature_cols=blob["feature_cols"], model=blob["model"])
