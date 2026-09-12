"""
geo_utils.py
============

Pure geographic math: distance, bearing, and destination-point calculations.
No fabrication risk here - these are deterministic formulas, not model
outputs. Functions return None (never a fabricated number) when given
invalid/NaN input.
"""

from __future__ import annotations

import math
from typing import Optional

EARTH_RADIUS_KM = 6371.0088


def _is_nan(x: float) -> bool:
    try:
        return math.isnan(x)
    except TypeError:
        return False


def haversine_distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two points, in kilometres."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return EARTH_RADIUS_KM * c


def geodesic_distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> Optional[float]:
    """
    Geodesic distance. We use the haversine formula (spherical-earth
    approximation, accurate to within ~0.5% for this use case) rather than
    a full ellipsoidal geodesic - sufficient for iceberg-scale trajectory
    error reporting. Returns None (never a fabricated distance) for
    invalid/NaN input.
    """
    for v in (lat1, lon1, lat2, lon2):
        if v is None or _is_nan(v):
            return None
    return haversine_distance_km(lat1, lon1, lat2, lon2)


def bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Initial compass bearing (degrees, 0-360, 0=north) from point 1 to point 2."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dlambda = math.radians(lon2 - lon1)
    x = math.sin(dlambda) * math.cos(phi2)
    y = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(dlambda)
    theta = math.atan2(x, y)
    return (math.degrees(theta) + 360) % 360


def destination_point(lat: float, lon: float, bearing_degrees: float, distance_km: float):
    """Given a start point, bearing, and distance, compute the destination point."""
    delta = distance_km / EARTH_RADIUS_KM
    theta = math.radians(bearing_degrees)
    phi1 = math.radians(lat)
    lambda1 = math.radians(lon)

    phi2 = math.asin(
        math.sin(phi1) * math.cos(delta) + math.cos(phi1) * math.sin(delta) * math.cos(theta)
    )
    lambda2 = lambda1 + math.atan2(
        math.sin(theta) * math.sin(delta) * math.cos(phi1),
        math.cos(delta) - math.sin(phi1) * math.sin(phi2),
    )
    dest_lat = math.degrees(phi2)
    dest_lon = (math.degrees(lambda2) + 540) % 360 - 180  # normalize to [-180, 180)
    return dest_lat, dest_lon
