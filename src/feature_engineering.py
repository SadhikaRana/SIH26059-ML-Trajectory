"""
feature_engineering.py
=======================

Derives movement features (displacement, speed, bearing, prior-step
movement) from lag columns produced by trajectory_builder.add_lag_features,
and passes through optional environmental columns (wind/current/sea-ice)
ONLY when genuinely present - never assumed, never fabricated.

All features here use only information available AT the base observation
time (lat/lon/timestamp of the current and past real observations, plus
environmental readings at that same time) - never the future target
position. This is what prevents temporal leakage in train.py.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from src.geo_utils import haversine_distance_km, bearing_deg

ENV_COLUMNS = ["wind_u", "wind_v", "current_u", "current_v", "sea_ice_concentration"]


EARTH_RADIUS_KM = 6371.0088


def _vec_haversine(lat1, lon1, lat2, lon2):
    """Vectorized haversine over pandas/numpy arrays - NaN-propagating,
    never fabricates a distance for missing input."""
    lat1r, lat2r = np.radians(lat1), np.radians(lat2)
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return EARTH_RADIUS_KM * c


def _vec_bearing(lat1, lon1, lat2, lon2):
    """Vectorized initial bearing (degrees, 0-360) - NaN-propagating."""
    lat1r, lat2r = np.radians(lat1), np.radians(lat2)
    dlon = np.radians(lon2 - lon1)
    x = np.sin(dlon) * np.cos(lat2r)
    y = np.cos(lat1r) * np.sin(lat2r) - np.sin(lat1r) * np.cos(lat2r) * np.cos(dlon)
    theta = np.arctan2(x, y)
    return (np.degrees(theta) + 360) % 360


def engineer_features(with_lags: pd.DataFrame, n_lags: int = 3):
    features = with_lags.copy()
    availability = {
        "displacement": False, "speed": False, "bearing": False,
        "prior_step_speed_and_bearing": False,
        "wind_u": False, "wind_v": False, "current_u": False,
        "current_v": False, "sea_ice_concentration": False,
    }

    has_lag1 = "lag1_latitude" in features.columns

    if has_lag1:
        features["displacement_km"] = _vec_haversine(
            features["lag1_latitude"], features["lag1_longitude"],
            features["latitude"], features["longitude"],
        )
        features["speed_km_per_hour"] = features["displacement_km"] / features["lag1_delta_hours"]
        features["bearing_deg"] = _vec_bearing(
            features["lag1_latitude"], features["lag1_longitude"],
            features["latitude"], features["longitude"],
        )
        availability["displacement"] = True
        availability["speed"] = True
        availability["bearing"] = True
    else:
        features["displacement_km"] = np.nan
        features["speed_km_per_hour"] = np.nan
        features["bearing_deg"] = np.nan

    has_lag2 = n_lags >= 2 and "lag2_latitude" in features.columns
    if has_lag2:
        features["prior_displacement_km"] = _vec_haversine(
            features["lag2_latitude"], features["lag2_longitude"],
            features["lag1_latitude"], features["lag1_longitude"],
        )
        features["prior_bearing_deg"] = _vec_bearing(
            features["lag2_latitude"], features["lag2_longitude"],
            features["lag1_latitude"], features["lag1_longitude"],
        )
        availability["prior_step_speed_and_bearing"] = True
    else:
        features["prior_displacement_km"] = np.nan
        features["prior_bearing_deg"] = np.nan

    for col in ENV_COLUMNS:
        if col in with_lags.columns and with_lags[col].notna().any():
            availability[col] = True
            # already present in `features` via the initial copy()

    return features, availability


def get_model_feature_columns(availability: dict, n_lags: int = 3) -> list[str]:
    cols: list[str] = []
    if availability.get("displacement"):
        cols.append("displacement_km")
    if availability.get("speed"):
        cols.append("speed_km_per_hour")
    if availability.get("bearing"):
        cols.append("bearing_deg")
    if availability.get("prior_step_speed_and_bearing"):
        cols.append("prior_displacement_km")
        cols.append("prior_bearing_deg")
    for col in ENV_COLUMNS:
        if availability.get(col):
            cols.append(col)
    return cols
