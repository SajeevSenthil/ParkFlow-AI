"""Regression metrics for count targets (PRD 7.7)."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def poisson_deviance(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Poisson deviance; appropriate for non-negative count targets.

    Guards the log against zeros. Lower is better.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.clip(np.asarray(y_pred, dtype=float), 1e-9, None)
    # Only take the log where y_true > 0 (term -> 0 as y_true -> 0); compute the
    # ratio only on the positive mask so log(0) is never evaluated.
    pos = y_true > 0
    term = np.zeros_like(y_true)
    term[pos] = y_true[pos] * np.log(y_true[pos] / y_pred[pos])
    return float(2.0 * np.mean(term - (y_true - y_pred)))


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.clip(np.asarray(y_pred, dtype=float), 0.0, None)
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "poisson_deviance": poisson_deviance(y_true, y_pred),
    }
