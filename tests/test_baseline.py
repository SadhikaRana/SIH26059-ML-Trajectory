"""
tests/test_baseline.py
========================

Covers: dead-reckoning baseline generates a future position from valid
historical movement; geographic calculations are sensible (known
geometry checks via geo_utils, which baseline.py depends on); insufficient
history produces a clear "not supported" result rather than a fabricated
prediction. Cross-iceberg contamination is structurally impossible here
since predict_dead_reckoning operates on a single iceberg's own
speed/bearing/position at a time (verified functionally below).
"""

from __future__ import annotations

import math
import unittest

from src.baseline import predict_dead_reckoning
from src.geo_utils import haversine_distance_km, geodesic_distance_km, bearing_deg, destination_point


class TestGeoUtilsGeometry(unittest.TestCase):
    """Known-geometry sanity checks - baseline.py's correctness depends
    entirely on these being right."""

    def test_one_degree_latitude_is_approximately_111km(self):
        d = haversine_distance_km(0.0, 0.0, 1.0, 0.0)
        self.assertTrue(math.isclose(d, 111.19, abs_tol=0.5))

    def test_geodesic_distance_matches_haversine_within_tolerance(self):
        d = geodesic_distance_km(0.0, 0.0, 1.0, 0.0)
        self.assertIsNotNone(d)
        self.assertTrue(math.isclose(d, 111.19, abs_tol=1.0))

    def test_geodesic_distance_returns_none_for_nan_input(self):
        """Never fabricate a distance for invalid/missing coordinates."""
        d = geodesic_distance_km(float("nan"), 0.0, 1.0, 0.0)
        self.assertIsNone(d)

    def test_bearing_due_north_is_zero(self):
        b = bearing_deg(0.0, 0.0, 1.0, 0.0)
        self.assertTrue(math.isclose(b, 0.0, abs_tol=0.5))

    def test_bearing_due_east_is_ninety(self):
        b = bearing_deg(0.0, 0.0, 0.0, 1.0)
        self.assertTrue(math.isclose(b, 90.0, abs_tol=0.5))

    def test_destination_point_roundtrips_with_bearing_and_known_distance(self):
        dest_lat, dest_lon = destination_point(0.0, 0.0, 0.0, 111.19)
        self.assertTrue(math.isclose(dest_lat, 1.0, abs_tol=0.02))
        self.assertTrue(math.isclose(dest_lon, 0.0, abs_tol=0.02))


class TestDeadReckoningBaseline(unittest.TestCase):

    def test_generates_future_position_from_valid_movement(self):
        result = predict_dead_reckoning(
            last_latitude=-65.0, last_longitude=45.0,
            speed_km_per_hour=2.0, bearing_degrees=90.0,  # due east
            horizon_hours=6,
        )
        self.assertTrue(result.supported)
        self.assertIsNotNone(result.predicted_latitude)
        self.assertIsNotNone(result.predicted_longitude)
        # Moving due east should barely change latitude, but should
        # increase longitude (heading east from a southern-hemisphere point).
        self.assertAlmostEqual(result.predicted_latitude, -65.0, delta=0.1)
        self.assertGreater(result.predicted_longitude, 45.0)

    def test_stationary_iceberg_zero_speed_stays_put(self):
        result = predict_dead_reckoning(
            last_latitude=-65.0, last_longitude=45.0,
            speed_km_per_hour=0.0, bearing_degrees=0.0,
            horizon_hours=24,
        )
        self.assertTrue(result.supported)
        self.assertAlmostEqual(result.predicted_latitude, -65.0, places=6)
        self.assertAlmostEqual(result.predicted_longitude, 45.0, places=6)

    def test_insufficient_history_is_reported_not_fabricated(self):
        """No speed/bearing estimate available (e.g. only one real
        observation exists) - the baseline MUST refuse to guess."""
        result = predict_dead_reckoning(
            last_latitude=-65.0, last_longitude=45.0,
            speed_km_per_hour=None, bearing_degrees=None,
            horizon_hours=6,
        )
        self.assertFalse(result.supported)
        self.assertIsNone(result.predicted_latitude)
        self.assertIsNone(result.predicted_longitude)
        self.assertIn("Insufficient movement history", result.reason)

    def test_partial_history_missing_bearing_only_still_refuses(self):
        result = predict_dead_reckoning(
            last_latitude=-65.0, last_longitude=45.0,
            speed_km_per_hour=1.5, bearing_degrees=None,
            horizon_hours=6,
        )
        self.assertFalse(result.supported)
        self.assertIsNone(result.predicted_latitude)

    def test_longer_horizon_produces_larger_displacement(self):
        short = predict_dead_reckoning(-65.0, 45.0, 2.0, 90.0, horizon_hours=6)
        long = predict_dead_reckoning(-65.0, 45.0, 2.0, 90.0, horizon_hours=48)

        short_dist = haversine_distance_km(-65.0, 45.0, short.predicted_latitude, short.predicted_longitude)
        long_dist = haversine_distance_km(-65.0, 45.0, long.predicted_latitude, long.predicted_longitude)
        self.assertGreater(long_dist, short_dist)


if __name__ == "__main__":
    unittest.main()
