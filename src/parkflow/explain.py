"""SHAP-based explanations (PRD: trust / interpretability).

Uses XGBoost's native exact TreeSHAP (``pred_contribs``) — no external dependency.
Produces two artifacts:
  * global feature importance  = mean |SHAP| across the forecast rows
  * per-zone "why this zone?"   = the top signed feature contributions for each zone
so a dispatcher can see *why* a junction is flagged, not just that it is.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import schema as S
from .logging_utils import get_logger
from .model import ViolationForecaster

log = get_logger("explain")


def explain_forecast(
    model: ViolationForecaster, future_frame: pd.DataFrame, top_features: int = 4
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (global_importance, per_zone_reasons) for the forecast rows."""
    contribs, _base = model.shap_contributions(future_frame)
    feats = model.feature_cols

    # Global importance = mean absolute SHAP per feature.
    global_imp = (
        pd.DataFrame({"feature": feats, "mean_abs_shap": np.abs(contribs).mean(axis=0)})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )

    # Per-zone: the features that pushed this zone's prediction up/down the most.
    zones = future_frame[S.ZONE].to_numpy()
    rows: list[dict] = []
    for i, zone in enumerate(zones):
        order = np.argsort(-np.abs(contribs[i]))[:top_features]
        for rank, j in enumerate(order, start=1):
            rows.append(
                {
                    S.ZONE: zone,
                    "rank": rank,
                    "feature": feats[j],
                    "feature_value": float(future_frame.iloc[i][feats[j]]),
                    "shap": round(float(contribs[i, j]), 4),
                    "direction": "increases" if contribs[i, j] >= 0 else "decreases",
                }
            )
    per_zone = pd.DataFrame(rows)
    log.info("Computed SHAP explanations for %d zones", len(zones))
    return global_imp, per_zone
