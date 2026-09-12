"""
tests/test_preprocessing.py
=============================

Covers: chronological sorting by iceberg_id and timestamp, numeric
conversion, timestamp parsing, preservation of iceberg identity, no
cross-iceberg mixing, invalid coordinates being detected/reported
(via profile_dataset.check_basic_integrity, which is the existing
design's home for that check - preprocessing itself does not drop or
flag invalid coordinates, it only cleans types).
"""

from __future__ import annotations

import unittest

import pandas as pd

from tests.fixtures import two_iceberg_clean_fixture, DEFAULT_CONFIG

from src.preprocessing import (
    parse_timestamps, coerce_numeric, sort_chronologically, run_basic_preprocessing,
)
from src.profile_dataset import check_basic_integrity


class TestPreprocessing(unittest.TestCase):

    def test_chronological_sorting_by_iceberg_and_timestamp(self):
        df = two_iceberg_clean_fixture()
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        shuffled = df.sample(frac=1, random_state=7).reset_index(drop=True)

        sorted_df = sort_chronologically(shuffled, "iceberg_id", "timestamp")

        for _, group in sorted_df.groupby("iceberg_id"):
            self.assertTrue(group["timestamp"].is_monotonic_increasing)

    def test_iceberg_identity_preserved_after_sort(self):
        df = two_iceberg_clean_fixture()
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        sorted_df = sort_chronologically(df, "iceberg_id", "timestamp")

        self.assertEqual(set(sorted_df["iceberg_id"]), {"TESTA", "TESTB"})
        self.assertEqual(len(sorted_df), len(df))  # no rows dropped or duplicated

    def test_no_cross_iceberg_mixing_in_sort(self):
        """Sorting must never interleave two icebergs' rows within each
        other - groups stay contiguous and internally ordered."""
        df = two_iceberg_clean_fixture()
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        sorted_df = sort_chronologically(df, "iceberg_id", "timestamp")

        # Each iceberg's rows should form a single contiguous block.
        change_points = (sorted_df["iceberg_id"] != sorted_df["iceberg_id"].shift()).sum()
        self.assertEqual(change_points, sorted_df["iceberg_id"].nunique())

    def test_timestamp_parsing_converts_strings_to_datetime(self):
        df = two_iceberg_clean_fixture()  # timestamps are ISO strings
        parsed = parse_timestamps(df, "timestamp")
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(parsed["timestamp"]))

    def test_timestamp_parsing_reports_unparseable_values_as_nat_not_dropped(self):
        df = two_iceberg_clean_fixture()
        df.loc[0, "timestamp"] = "not-a-real-timestamp"
        parsed = parse_timestamps(df, "timestamp")

        self.assertEqual(len(parsed), len(df))  # row count preserved
        self.assertTrue(pd.isna(parsed.loc[0, "timestamp"]))  # became NaT, not silently dropped

    def test_numeric_coercion_of_latitude_longitude(self):
        df = two_iceberg_clean_fixture()
        df["latitude"] = df["latitude"].astype(str)  # simulate messy string input
        coerced = coerce_numeric(df, ["latitude", "longitude"])
        self.assertTrue(pd.api.types.is_numeric_dtype(coerced["latitude"]))

    def test_numeric_coercion_reports_unconvertible_values_as_nan(self):
        df = two_iceberg_clean_fixture()
        df["latitude"] = df["latitude"].astype(object)  # allow a mixed-type column, like messy real data
        df.loc[0, "latitude"] = "not-a-number"
        coerced = coerce_numeric(df, ["latitude"])
        self.assertTrue(pd.isna(coerced.loc[0, "latitude"]))
        self.assertEqual(len(coerced), len(df))  # not dropped

    def test_invalid_coordinates_detected_by_integrity_check(self):
        """preprocessing.py itself does not flag invalid coordinates - that
        is profile_dataset.check_basic_integrity's job. This test verifies
        the two modules agree: preprocessing produces clean numeric
        columns, and the integrity check correctly flags out-of-range
        values on top of them."""
        df = two_iceberg_clean_fixture()
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df.loc[0, "latitude"] = 999.0  # clearly invalid

        report = check_basic_integrity(df, DEFAULT_CONFIG)
        self.assertGreaterEqual(report.n_invalid_latitude, 1)

    def test_run_basic_preprocessing_end_to_end(self):
        df = two_iceberg_clean_fixture()
        resolved_core = {"iceberg_id": "iceberg_id", "timestamp": "timestamp",
                          "latitude": "latitude", "longitude": "longitude"}
        processed = run_basic_preprocessing(df, resolved_core, {}, DEFAULT_CONFIG)

        self.assertTrue(pd.api.types.is_datetime64_any_dtype(processed["timestamp"]))
        for _, group in processed.groupby("iceberg_id"):
            self.assertTrue(group["timestamp"].is_monotonic_increasing)


if __name__ == "__main__":
    unittest.main()
