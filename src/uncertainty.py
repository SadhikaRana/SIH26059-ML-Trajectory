"""
uncertainty.py
==============

Uncertainty exposed to Person 4 is ALWAYS an empirical figure derived from
real held-out validation residuals for that specific horizon - never an
invented confidence value. If no validation has been run yet for a
horizon, uncertainty is None, with an explicit reason attached.
"""

from __future__ import annotations

from typing import Optional

NOT_AVAILABLE_REASON = "Not available - no real validation metrics have been computed for this horizon yet."


def estimate_uncertainty_km(
    horizon_hours: int,
    validation_metrics_by_horizon: Optional[dict],
    method: str = "rmse",
):
    if not validation_metrics_by_horizon:
        return None, NOT_AVAILABLE_REASON

    metrics = validation_metrics_by_horizon.get(horizon_hours)
    if metrics is None or metrics.get("status") != "ok":
        return None, NOT_AVAILABLE_REASON

    key = f"{method}_km"
    if key not in metrics:
        return None, NOT_AVAILABLE_REASON

    value = metrics[key]
    reason = (
        f"Empirical uncertainty ({method.upper()}) derived from real held-out "
        f"validation at the {horizon_hours}h horizon, n={metrics.get('n_samples')} samples."
    )
    return value, reason
