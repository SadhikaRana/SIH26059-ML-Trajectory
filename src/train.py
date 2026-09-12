"""
train.py
========

Trains a model per horizon on REAL supervised pairs only (base observation
-> genuine future observation, matched by trajectory_builder). Refuses to
train (raises TrainingError) when there are fewer real samples than
config.training.min_samples_to_train - never fits and reports a model on
too little real data.

Chronological (time-aware) train/validation split: the split point is a
timestamp quantile of base_timestamp, so validation pairs are always
LATER in time than training pairs, for every iceberg pooled together.
This prevents leaking future information into training.

Model target is the (delta_latitude, delta_longitude) displacement from
base -> target, not the absolute target position - this keeps the
regression problem stable across icebergs starting at very different
latitudes/longitudes, and the base position is added back afterward to
produce an absolute predicted_latitude/predicted_longitude.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("sih26059.train")

try:
    from xgboost import XGBRegressor
    _MODEL_BACKEND = "xgboost"
except ImportError:
    from sklearn.ensemble import GradientBoostingRegressor as _SkGBR
    _MODEL_BACKEND = "sklearn_gradient_boosting (xgboost not installed - documented stand-in)"


class TrainingError(Exception):
    """Raised when training must refuse (insufficient real data, etc.)."""


def _make_regressor(random_state: int):
    if _MODEL_BACKEND == "xgboost":
        return XGBRegressor(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, random_state=random_state,
            n_jobs=-1,
        )
    from sklearn.ensemble import GradientBoostingRegressor
    return GradientBoostingRegressor(random_state=random_state)


def prepare_training_table(
    pairs: pd.DataFrame, features: pd.DataFrame, feature_columns: list[str], horizon_hours: int,
) -> pd.DataFrame:
    matched = pairs[pairs["matched_horizon_hours"] == horizon_hours]
    if matched.empty:
        return pd.DataFrame(columns=feature_columns + [
            "iceberg_id", "base_timestamp", "target_timestamp",
            "base_latitude", "base_longitude", "target_latitude", "target_longitude",
        ])

    base = features.loc[matched["base_index"]].reset_index(drop=True)
    target = features.loc[matched["target_index"], ["latitude", "longitude"]].reset_index(drop=True)
    target.columns = ["target_latitude", "target_longitude"]

    table = pd.concat([
        base[["iceberg_id"] + feature_columns].reset_index(drop=True),
        base[["latitude", "longitude"]].rename(
            columns={"latitude": "base_latitude", "longitude": "base_longitude"}
        ).reset_index(drop=True),
        matched[["base_timestamp", "target_timestamp"]].reset_index(drop=True),
        target,
    ], axis=1)

    # Drop rows with any missing feature value - a model cannot train on NaNs,
    # and we never impute/fabricate a feature value.
    table = table.dropna(subset=feature_columns)
    return table


def train_horizon_models(
    training_table: pd.DataFrame, feature_columns: list[str], config: dict, horizon_hours: int,
):
    min_samples = config["training"]["min_samples_to_train"]
    if len(training_table) < min_samples:
        raise TrainingError(
            f"Not available - only {len(training_table)} real supervised samples exist for "
            f"the {horizon_hours}h horizon, below the configured minimum of {min_samples}. "
            f"Refusing to train rather than fit a model on too little real data."
        )

    val_fraction = config["training"]["validation_fraction_by_time"]
    table = training_table.sort_values("base_timestamp").reset_index(drop=True)
    split_idx = int(len(table) * (1 - val_fraction))
    train_df = table.iloc[:split_idx]
    val_df = table.iloc[split_idx:]

    train_df = train_df.assign(
        delta_lat=train_df["target_latitude"] - train_df["base_latitude"],
        delta_lon=train_df["target_longitude"] - train_df["base_longitude"],
    )

    random_state = config.get("model", {}).get("random_state", 42)
    model_lat = _make_regressor(random_state)
    model_lon = _make_regressor(random_state)
    model_lat.fit(train_df[feature_columns], train_df["delta_lat"])
    model_lon.fit(train_df[feature_columns], train_df["delta_lon"])

    val_predictions = None
    if len(val_df) > 0:
        pred_delta_lat = model_lat.predict(val_df[feature_columns])
        pred_delta_lon = model_lon.predict(val_df[feature_columns])
        val_predictions = pd.DataFrame({
            "horizon_hours": horizon_hours,
            "predicted_latitude": val_df["base_latitude"].values + pred_delta_lat,
            "predicted_longitude": val_df["base_longitude"].values + pred_delta_lon,
            "target_latitude": val_df["target_latitude"].values,
            "target_longitude": val_df["target_longitude"].values,
        })

    return {
        "model_lat": model_lat,
        "model_lon": model_lon,
        "feature_columns": feature_columns,
        "horizon_hours": horizon_hours,
        "n_train": len(train_df),
        "n_val": len(val_df),
        "model_backend": _MODEL_BACKEND,
        "val_predictions": val_predictions,
    }
