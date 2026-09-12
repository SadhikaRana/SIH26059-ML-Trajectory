"""
tests/test_data_loader.py
==========================

Covers: supported CSV loading, missing dataset path, unsupported
extension, missing required columns, config-based column mapping.

All fixtures are written to tempfile.TemporaryDirectory() - never to the
project's real data/raw/.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tests.fixtures import two_iceberg_clean_fixture, DEFAULT_CONFIG

from src.data_loader import (
    load_raw_dataset, resolve_column_mapping, DataLoadError,
)


class TestDataLoader(unittest.TestCase):

    def test_loads_supported_csv(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            raw_dir = Path(tmpdir) / "raw"
            raw_dir.mkdir()
            two_iceberg_clean_fixture().to_csv(raw_dir / "fixture.csv", index=False)

            config = {"data": {"raw_dir": str(raw_dir), "supported_extensions": [".csv"]}}
            df = load_raw_dataset(config)

            self.assertEqual(len(df), 12)
            self.assertIn("iceberg_id", df.columns)

    def test_missing_dataset_path_raises_clear_error(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            raw_dir = Path(tmpdir) / "raw"
            raw_dir.mkdir()  # exists but empty - no files at all
            config = {"data": {"raw_dir": str(raw_dir), "supported_extensions": [".csv"]}}

            with self.assertRaises(DataLoadError) as ctx:
                load_raw_dataset(config)
            self.assertIn("No supported data files found", str(ctx.exception))

    def test_nonexistent_raw_dir_raises_clear_error(self):
        config = {"data": {"raw_dir": "/this/path/does/not/exist/anywhere",
                            "supported_extensions": [".csv"]}}
        with self.assertRaises(DataLoadError) as ctx:
            load_raw_dataset(config)
        self.assertIn("does not exist", str(ctx.exception))

    def test_unsupported_extension_is_not_picked_up(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            raw_dir = Path(tmpdir) / "raw"
            raw_dir.mkdir()
            (raw_dir / "notes.txt").write_text("not a dataset")

            config = {"data": {"raw_dir": str(raw_dir), "supported_extensions": [".csv"]}}
            with self.assertRaises(DataLoadError):
                load_raw_dataset(config)

    def test_explicit_filename_used_when_multiple_candidates_present(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            raw_dir = Path(tmpdir) / "raw"
            raw_dir.mkdir()
            two_iceberg_clean_fixture().to_csv(raw_dir / "a.csv", index=False)
            two_iceberg_clean_fixture().to_csv(raw_dir / "b.csv", index=False)

            # Without an explicit filename, ambiguity must be reported, not guessed.
            ambiguous_config = {"data": {"raw_dir": str(raw_dir), "supported_extensions": [".csv"]}}
            with self.assertRaises(DataLoadError) as ctx:
                load_raw_dataset(ambiguous_config)
            self.assertIn("Multiple candidate data files", str(ctx.exception))

            # With an explicit filename, it should load without ambiguity.
            explicit_config = {"data": {"raw_dir": str(raw_dir), "input_filename": "a.csv",
                                          "supported_extensions": [".csv"]}}
            df = load_raw_dataset(explicit_config)
            self.assertEqual(len(df), 12)

    def test_missing_required_column_is_reported_not_silently_ignored(self):
        df = two_iceberg_clean_fixture().drop(columns=["latitude"])
        resolved_core, resolved_optional, missing_core = resolve_column_mapping(df, DEFAULT_CONFIG)

        self.assertIn("latitude", missing_core)
        self.assertNotIn("latitude", resolved_core)
        # The other three core columns should still resolve fine.
        self.assertIn("iceberg_id", resolved_core)
        self.assertIn("timestamp", resolved_core)
        self.assertIn("longitude", resolved_core)

    def test_config_based_column_mapping_resolves_renamed_columns(self):
        """The loader must not assume the real dataset's column names - it
        should resolve whatever name config.yaml points at."""
        df = two_iceberg_clean_fixture().rename(columns={
            "latitude": "Lat_deg", "longitude": "Lon_deg",
        })
        config = {
            "columns": {"iceberg_id": "iceberg_id", "timestamp": "timestamp",
                         "latitude": "Lat_deg", "longitude": "Lon_deg"},
            "environmental": {}, "iceberg_characteristics": {},
        }
        resolved_core, resolved_optional, missing_core = resolve_column_mapping(df, config)

        self.assertEqual(missing_core, [])
        self.assertEqual(resolved_core["latitude"], "Lat_deg")
        self.assertEqual(resolved_core["longitude"], "Lon_deg")

    def test_optional_environmental_columns_resolved_only_when_present(self):
        df = two_iceberg_clean_fixture()  # no wind/current columns
        resolved_core, resolved_optional, missing_core = resolve_column_mapping(df, DEFAULT_CONFIG)

        self.assertEqual(resolved_optional, {})  # none present -> none resolved, not fabricated


if __name__ == "__main__":
    unittest.main()
