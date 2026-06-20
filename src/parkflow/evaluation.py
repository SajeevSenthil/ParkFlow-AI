"""Regression + ranking metrics for sparse count targets (PRD 7.7).

For sparse, zero-heavy data, MAE/RMSE alone are misleading. We also report ranking
metrics that match the operational question ("did we flag the right hotspots?"):
Top-K hit-rate per time slice and Precision-Recall AUC for hotspot classification.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


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


def hotspot_pr_auc(y_true: np.ndarray, y_pred: np.ndarray, threshold: float) -> float:
    """PR-AUC (average precision) for the binary task 'is this cell a hotspot?'
    (actual count >= threshold), scored by the predicted count. Robust to the heavy
    class imbalance that makes plain accuracy meaningless here.
    """
    labels = (np.asarray(y_true, dtype=float) >= threshold).astype(int)
    if labels.sum() == 0 or labels.sum() == len(labels):
        return float("nan")
    return float(average_precision_score(labels, np.asarray(y_pred, dtype=float)))


def top_k_hit_rate(
    frame: pd.DataFrame, time_col: str, y_true_col: str, y_pred_col: str, k: int
) -> float:
    """Mean overlap between the top-K predicted and top-K actual zones, per time slice.
    1.0 = every period's K worst zones were correctly identified.
    """
    hits: list[float] = []
    for _, g in frame.groupby(time_col):
        if len(g) < k:
            continue
        top_pred = set(g.nlargest(k, y_pred_col).index)
        top_true = set(g.nlargest(k, y_true_col).index)
        hits.append(len(top_pred & top_true) / k)
    return float(np.mean(hits)) if hits else float("nan")


def ranking_metrics(
    frame: pd.DataFrame,
    time_col: str,
    y_true_col: str,
    y_pred_col: str,
    threshold: float,
    k: int,
) -> dict[str, float]:
    return {
        f"top_{k}_hit_rate": top_k_hit_rate(frame, time_col, y_true_col, y_pred_col, k),
        "hotspot_pr_auc": hotspot_pr_auc(
            frame[y_true_col].to_numpy(), frame[y_pred_col].to_numpy(), threshold
        ),
    }
