"""
tests/test_feature_engineering.py
====================================

Covers: displacement calculation, movement speed, bearing calculation,
lagged movement features, optional environmental variables being used
when present and NOT required when absent.
"""

from __future__ import annotations

import math
import unittest

import pandas as pd

from tests.fixtures import two_iceberg_clean_fixture, two_iceberg_fixture_with_environmental

from src.preprocessing import sort_chronologically
from src.trajectory_builder import add_lag_features
from src.feature_engineering import engineer_features, get_model_feature_columns


def _lagged(df: pd.DataFrame, n_lags: int = 2) -> pd.DataFrame:
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = sort_chronologically(df, "iceberg_id", "timestamp")
    return add_lag_features(df, n_lags=n_lags)


class TestFeatureEngineering(unittest.TestCase):

    def test_displacement_and_speed_computed_when_lag_available(self):
        with_lags = _lagged(two_iceberg_clean_fixture())
        features, availability = engineer_features(with_lags, n_lags=2)

        self.assertTrue(availability["displacement"])
        self.assertTrue(availability["speed"])

        row = features[features["iceberg_id"] == "TESTA"].iloc[1]  # has lag1
        self.assertFalse(math.isnan(row["displacement_km"]))
        self.assertGreater(row["displacement_km"], 0)
        self.assertFalse(math.isnan(row["speed_km_per_hour"]))

    def test_bearing_is_computed_and_within_valid_range(self):
        with_lags = _lagged(two_iceberg_clean_fixture())
        features, availability = engineer_features(with_lags, n_lags=2)

        self.assertTrue(availability["bearing"])
        bearings = features["bearing_deg"].dropna()
        self.assertTrue(((bearings >= 0) & (bearings < 360)).all())

    def test_lagged_movement_features_from_prior_step(self):
        with_lags = _lagged(two_iceberg_clean_fixture(), n_lags=2)
        features, availability = engineer_features(with_lags, n_lags=2)

        self.assertTrue(availability["prior_step_speed_and_bearing"])
        self.assertIn("prior_displacement_km", features.columns)
        self.assertIn("prior_bearing_deg", features.columns)

        # A row with full 2-lag history should have a non-NaN prior_displacement_km.
        third_testa_row = features[features["iceberg_id"] == "TESTA"].iloc[2]
        self.assertFalse(math.isnan(third_testa_row["prior_displacement_km"]))

    def test_first_observation_of_iceberg_has_no_movement_features(self):
        """The very first observation for an iceberg has no lag1 - it must
        get NaN movement features, never a fabricated 0 or guessed value."""
        with_lags = _lagged(two_iceberg_clean_fixture())
        features, _ = engineer_features(with_lags, n_lags=2)

        first_testa_row = features[features["iceberg_id"] == "TESTA"].iloc[0]
        self.assertTrue(math.isnan(first_testa_row["displacement_km"]))
        self.assertTrue(math.isnan(first_testa_row["speed_km_per_hour"]))

    def test_environmental_columns_used_when_present(self):
        with_lags = _lagged(two_iceberg_fixture_with_environmental())
        features, availability = engineer_features(with_lags, n_lags=2)

        self.assertTrue(availability["wind_u"])
        self.assertTrue(availability["wind_v"])
        self.assertIn("wind_u", features.columns)

    def test_environmental_columns_not_required_when_absent(self):
        """Core feature engineering must succeed even with zero
        environmental columns - they are optional, never assumed."""
        with_lags = _lagged(two_iceberg_clean_fixture())  # no wind/current cols
        features, availability = engineer_features(with_lags, n_lags=2)

        self.assertFalse(availability["wind_u"])
        self.assertFalse(availability["current_u"])
        # But movement features (which don't depend on environmental data)
        # must still be computed successfully.
        self.assertTrue(availability["displacement"])

    def test_get_model_feature_columns_reflects_availability(self):
        with_lags = _lagged(two_iceberg_clean_fixture())
        _, availability = engineer_features(with_lags, n_lags=2)
        cols = get_model_feature_columns(availability, n_lags=2)

        self.assertIn("displacement_km", cols)
        self.assertNotIn("wind_u", cols)  # not available -> must not be listed

    def test_get_model_feature_columns_includes_environmental_when_available(self):
        with_lags = _lagged(two_iceberg_fixture_with_environmental())
        _, availability = engineer_features(with_lags, n_lags=2)
        cols = get_model_feature_columns(availability, n_lags=2)

        self.assertIn("wind_u", cols)
        self.assertIn("wind_v", cols)


if __name__ == "__main__":
    unittest.main()
