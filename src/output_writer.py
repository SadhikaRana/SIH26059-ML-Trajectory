"""
output_writer.py
=================

Writes the canonical prediction output (CSV + nested JSON) consumed by
Person 4. Validates the schema before writing so a malformed dataframe
never silently reaches disk.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

CANONICAL_CSV_COLUMNS = [
    "iceberg_id", "prediction_timestamp", "horizon_hours",
    "predicted_latitude", "predicted_longitude", "uncertainty_km",
]


class OutputSchemaError(Exception):
    """Raised when a predictions dataframe does not match the canonical schema."""


def validate_predictions_schema(df: pd.DataFrame) -> None:
    missing = [c for c in CANONICAL_CSV_COLUMNS if c not in df.columns]
    if missing:
        raise OutputSchemaError(
            f"Predictions dataframe is missing required canonical column(s): {missing}"
        )


def write_predictions_csv(df: pd.DataFrame, path) -> None:
    validate_predictions_schema(df)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df[CANONICAL_CSV_COLUMNS].to_csv(path, index=False)


def _to_native(value):
    if value is None:
        return None
    if isinstance(value, float) and np.isnan(value):
        return None
    if pd.isna(value) if not isinstance(value, (list, dict)) else False:
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def build_json_structure(df: pd.DataFrame) -> list[dict]:
    validate_predictions_schema(df)
    structure = []
    for (iceberg_id, prediction_timestamp), group in df.groupby(
        ["iceberg_id", "prediction_timestamp"], sort=False, dropna=False
    ):
        predictions = []
        for _, row in group.iterrows():
            predictions.append({
                "horizon_hours": _to_native(row["horizon_hours"]),
                "predicted_latitude": _to_native(row["predicted_latitude"]),
                "predicted_longitude": _to_native(row["predicted_longitude"]),
                "uncertainty_km": _to_native(row["uncertainty_km"]),
            })
        structure.append({
            "iceberg_id": iceberg_id,
            "prediction_timestamp": _to_native(prediction_timestamp),
            "predictions": predictions,
        })
    return structure


def write_predictions_json(df: pd.DataFrame, path) -> None:
    structure = build_json_structure(df)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(structure, f, indent=2)
