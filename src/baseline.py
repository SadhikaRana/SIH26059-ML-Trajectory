"""
baseline.py
===========

Dead-reckoning (persistence-of-velocity) baseline. Given a real observed
speed and bearing, project the position forward. Refuses to guess when
speed/bearing are unavailable (e.g. an iceberg with only one real
observation) - it never fabricates a velocity it has no evidence for.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from src.geo_utils import destination_point


@dataclass
class DeadReckoningResult:
    supported: bool
    predicted_latitude: Optional[float]
    predicted_longitude: Optional[float]
    reason: str


def predict_dead_reckoning(
    last_latitude: float,
    last_longitude: float,
    speed_km_per_hour: Optional[float],
    bearing_degrees: Optional[float],
    horizon_hours: float,
) -> DeadReckoningResult:
    if speed_km_per_hour is None or bearing_degrees is None:
        return DeadReckoningResult(
            supported=False,
            predicted_latitude=None,
            predicted_longitude=None,
            reason=(
                "Insufficient movement history: real speed/bearing are not "
                "available for this iceberg's latest observation, so no "
                "dead-reckoning projection can be made without fabricating "
                "a velocity."
            ),
        )

    distance_km = speed_km_per_hour * horizon_hours
    dest_lat, dest_lon = destination_point(
        last_latitude, last_longitude, bearing_degrees, distance_km
    )
    return DeadReckoningResult(
        supported=True,
        predicted_latitude=dest_lat,
        predicted_longitude=dest_lon,
        reason="Dead-reckoning projection from real observed speed/bearing.",
    )
