"""
tests/test_trajectory_builder.py
===================================

Covers: observations grouped by iceberg, lag features never crossing
iceberg boundaries, correct time-difference calculation, no future-
information leakage into "past" lag features or into supervised pairs,
and clean handling of icebergs with insufficient observations.
"""

from __future__ import annotations

import unittest

import pandas as pd

from tests.fixtures import two_iceberg_clean_fixture, single_observation_fixture, DEFAULT_CONFIG

from src.preprocessing import sort_chronologically
from src.trajectory_builder import (
    add_lag_features, build_supervised_pairs, TrajectoryBuilderError,
)


def _prepped(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return sort_chronologically(df, "iceberg_id", "timestamp")


class TestTrajectoryBuilder(unittest.TestCase):

    def test_lag_features_grouped_by_iceberg(self):
        df = _prepped(two_iceberg_clean_fixture())
        with_lags = add_lag_features(df, n_lags=2)

        # Every row should have exactly its own iceberg_id, unaffected by lag columns.
        self.assertEqual(set(with_lags["iceberg_id"]), {"TESTA", "TESTB"})
        self.assertIn("lag1_latitude", with_lags.columns)
        self.assertIn("lag2_latitude", with_lags.columns)

    def test_lag_features_do_not_cross_iceberg_boundaries(self):
        df = _prepped(two_iceberg_clean_fixture())
        with_lags = add_lag_features(df, n_lags=1)

        first_testb_row = with_lags[with_lags["iceberg_id"] == "TESTB"].iloc[0]
        # TESTB's first observation must have NO lag1, even though TESTA's
        # last observation sits immediately before it in the sorted frame.
        self.assertTrue(pd.isna(first_testb_row["lag1_latitude"]))
        self.assertTrue(pd.isna(first_testb_row["lag1_longitude"]))
        self.assertTrue(pd.isna(first_testb_row["lag1_timestamp"]))

    def test_lag_time_differences_calculated_correctly(self):
        df = _prepped(two_iceberg_clean_fixture())  # 6h cadence
        with_lags = add_lag_features(df, n_lags=1)

        second_testa_row = with_lags[with_lags["iceberg_id"] == "TESTA"].iloc[1]
        self.assertAlmostEqual(second_testa_row["lag1_delta_hours"], 6.0, places=6)

    def test_lag_features_raise_on_unsorted_input(self):
        df = two_iceberg_clean_fixture()
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        shuffled = df.sample(frac=1, random_state=3).reset_index(drop=True)  # NOT sorted

        with self.assertRaises(TrajectoryBuilderError):
            add_lag_features(shuffled, n_lags=1)

    def test_insufficient_observations_handled_cleanly(self):
        """A single-observation iceberg must not crash lag-feature
        construction - it should just get all-NaN lags."""
        df = _prepped(single_observation_fixture())
        with_lags = add_lag_features(df, n_lags=2)

        self.assertEqual(len(with_lags), 1)
        self.assertTrue(pd.isna(with_lags.iloc[0]["lag1_latitude"]))

    def test_supervised_pairs_only_use_real_future_observations(self):
        df = _prepped(two_iceberg_clean_fixture())
        pairs = build_supervised_pairs(df, DEFAULT_CONFIG)

        self.assertFalse(pairs.empty)
        # Every pair's target must be strictly after its base - no backward
        # or same-timestamp pairing (which would indicate leakage).
        self.assertTrue((pairs["target_timestamp"] > pairs["base_timestamp"]).all())

    def test_supervised_pairs_never_mix_icebergs(self):
        df = _prepped(two_iceberg_clean_fixture())
        pairs = build_supervised_pairs(df, DEFAULT_CONFIG)

        # base and target rows in a pair always originate from a groupby
        # over a single iceberg_id, but we double check by re-deriving
        # iceberg_id for the target_index and comparing.
        df_indexed = df.reset_index()  # 'index' column matches base_index/target_index
        target_iceberg = df_indexed.set_index("index").loc[pairs["target_index"], "iceberg_id"].values
        self.assertTrue((pairs["iceberg_id"].values == target_iceberg).all())

    def test_supervised_pairs_tag_matching_horizon_correctly(self):
        df = _prepped(two_iceberg_clean_fixture())  # 6h cadence
        config = {
            "prediction": {"horizons_hours": [6, 12], "horizon_feasibility_tolerance_hours": 1.0},
            "trajectory_builder": {"max_lookahead_multiplier": 1.5},
        }
        pairs = build_supervised_pairs(df, config)

        six_h_pairs = pairs[pairs["matched_horizon_hours"] == 6]
        self.assertTrue((six_h_pairs["delta_hours"].round(1) == 6.0).all())

    def test_insufficient_observations_yield_no_pairs_not_a_crash(self):
        df = _prepped(single_observation_fixture())
        pairs = build_supervised_pairs(df, DEFAULT_CONFIG)
        self.assertTrue(pairs.empty)  # explicitly empty, not fabricated, not a crash


if __name__ == "__main__":
    unittest.main()
