"""
predict.py
==========

End-to-end single-(iceberg, horizon) prediction: prefers a trained ML
model when available and its required features are present for this
iceberg's latest real observation; falls back to the dead-reckoning
baseline when real movement history exists but no model does; and
reports "unsupported" (never a guessed position) otherwise.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from src.baseline import predict_dead_reckoning
from src.uncertainty import estimate_uncertainty_km


def predict_for_iceberg_horizon(
    row: pd.Series,
    horizon_hours: int,
    trained_model_bundle: Optional[dict],
    validation_metrics_by_horizon: Optional[dict],
) -> dict:
    predicted_latitude = None
    predicted_longitude = None
    model_source = "unsupported"
    reason = "No trained model and insufficient real movement history for this iceberg."

    if trained_model_bundle is not None:
        feature_columns = trained_model_bundle["feature_columns"]
        has_all_features = all(
            (col in row.index) and pd.notna(row[col]) for col in feature_columns
        )
        if has_all_features:
            x = pd.DataFrame([row[feature_columns].values], columns=feature_columns)
            delta_lat = trained_model_bundle["model_lat"].predict(x)[0]
            delta_lon = trained_model_bundle["model_lon"].predict(x)[0]
            predicted_latitude = float(row["latitude"] + delta_lat)
            predicted_longitude = float(row["longitude"] + delta_lon)
            model_source = "model"
            reason = "Trained model prediction."

    if model_source == "unsupported":
        speed = row.get("speed_km_per_hour")
        bearing = row.get("bearing_deg")
        speed = None if pd.isna(speed) else speed
        bearing = None if pd.isna(bearing) else bearing
        baseline_result = predict_dead_reckoning(
            last_latitude=row["latitude"], last_longitude=row["longitude"],
            speed_km_per_hour=speed, bearing_degrees=bearing,
            horizon_hours=horizon_hours,
        )
        if baseline_result.supported:
            predicted_latitude = baseline_result.predicted_latitude
            predicted_longitude = baseline_result.predicted_longitude
            model_source = "baseline"
            reason = baseline_result.reason

    uncertainty_km, _uncertainty_reason = estimate_uncertainty_km(
        horizon_hours, validation_metrics_by_horizon
    )
    if model_source == "unsupported":
        uncertainty_km = None

    return {
        "iceberg_id": row["iceberg_id"],
        "prediction_timestamp": row["timestamp"],
        "horizon_hours": horizon_hours,
        "predicted_latitude": predicted_latitude,
        "predicted_longitude": predicted_longitude,
        "uncertainty_km": uncertainty_km,
        "model_source": model_source,
        "reason": reason,
    }
