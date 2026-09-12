"""
evaluate.py
===========

Computes real, held-out geographic error metrics per horizon from actual
(predicted, target) position pairs. A horizon with zero real ground-truth
rows is explicitly marked "unavailable" - never silently skipped, never
given a fabricated metric.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.geo_utils import haversine_distance_km

NOT_AVAILABLE = "Not available - no real ground-truth predictions exist for this horizon."

# Distance thresholds (km) used for the presentation-facing "accuracy /
# success ratio" metric. This is a THRESHOLD-BASED HIT RATE, distinct from
# statistical model accuracy - see accuracy_within_thresholds_km docstring
# below and MODULE_README.md for the exact definition shown to users.
DEFAULT_THRESHOLDS_KM = [5, 10, 25]


def _accuracy_within_thresholds_km(errors_km: pd.Series, thresholds=DEFAULT_THRESHOLDS_KM) -> dict:
    """
    For each threshold T, the fraction of held-out real predictions whose
    haversine error was <= T km. This is a defensible, literal statement
    ("X% of predictions were within T km") - NOT the same thing as
    "the model is X% accurate" and must never be labeled as such.
    """
    n = len(errors_km)
    return {
        f"within_{t}km": float((errors_km <= t).sum()) / n if n > 0 else None
        for t in thresholds
    }


def evaluate_by_horizon(predictions: pd.DataFrame, horizons: list[int]) -> dict:
    results = {}
    for h in horizons:
        subset = predictions[predictions["horizon_hours"] == h] if not predictions.empty else predictions
        subset = subset.dropna(subset=["predicted_latitude", "predicted_longitude",
                                        "target_latitude", "target_longitude"]) if not subset.empty else subset

        if subset is None or len(subset) == 0:
            results[h] = {"status": "unavailable", "reason": NOT_AVAILABLE}
            continue

        errors_km = subset.apply(
            lambda r: haversine_distance_km(
                r["predicted_latitude"], r["predicted_longitude"],
                r["target_latitude"], r["target_longitude"],
            ),
            axis=1,
        )

        results[h] = {
            "status": "ok",
            "n_samples": int(len(subset)),
            "mae_km": float(errors_km.mean()),
            "rmse_km": float(np.sqrt((errors_km ** 2).mean())),
            "median_km": float(errors_km.median()),
            "p95_km": float(errors_km.quantile(0.95)),
            "accuracy_within_thresholds_km": _accuracy_within_thresholds_km(errors_km),
        }
    return results
