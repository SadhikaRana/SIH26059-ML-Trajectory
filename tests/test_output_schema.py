"""
tests/test_output_schema.py
==============================

Covers: prediction output conforms to the canonical schema
(iceberg_id, prediction_timestamp, horizon_hours, predicted_latitude,
predicted_longitude, uncertainty_km), latitude/longitude bounds,
horizon_hours validity, and uncertainty_km being either a legitimate
number or null/empty when unavailable.

All CSV/JSON artifacts here are written to a temporary directory, never
to the project's real outputs/predictions/.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.output_writer import (
    validate_predictions_schema, write_predictions_csv, write_predictions_json,
    build_json_structure, OutputSchemaError, CANONICAL_CSV_COLUMNS,
)

VALID_HORIZONS = {6, 12, 24, 48}


def _make_predictions_df(uncertainty_km=1.5) -> pd.DataFrame:
    return pd.DataFrame([
        {"iceberg_id": "TESTA", "prediction_timestamp": pd.Timestamp("2026-01-01T12:00:00"),
         "horizon_hours": 6, "predicted_latitude": -65.5, "predicted_longitude": 45.2,
         "uncertainty_km": uncertainty_km},
        {"iceberg_id": "TESTA", "prediction_timestamp": pd.Timestamp("2026-01-01T12:00:00"),
         "horizon_hours": 12, "predicted_latitude": -65.7, "predicted_longitude": 45.4,
         "uncertainty_km": uncertainty_km},
    ])


class TestOutputSchema(unittest.TestCase):

    def test_valid_dataframe_passes_schema_validation(self):
        df = _make_predictions_df()
        try:
            validate_predictions_schema(df)
        except OutputSchemaError:
            self.fail("validate_predictions_schema raised unexpectedly on a valid dataframe")

    def test_missing_canonical_column_is_rejected(self):
        df = _make_predictions_df().drop(columns=["uncertainty_km"])
        with self.assertRaises(OutputSchemaError) as ctx:
            validate_predictions_schema(df)
        self.assertIn("uncertainty_km", str(ctx.exception))

    def test_csv_has_exact_canonical_column_order(self):
        df = _make_predictions_df()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "preds.csv"
            write_predictions_csv(df, path)
            written = pd.read_csv(path)
            self.assertEqual(list(written.columns), CANONICAL_CSV_COLUMNS)

    def test_csv_predictions_have_valid_geographic_bounds(self):
        df = _make_predictions_df()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "preds.csv"
            write_predictions_csv(df, path)
            written = pd.read_csv(path)

            self.assertTrue((written["predicted_latitude"] >= -90).all())
            self.assertTrue((written["predicted_latitude"] <= 90).all())
            self.assertTrue((written["predicted_longitude"] >= -180).all())
            self.assertTrue((written["predicted_longitude"] <= 180).all())

    def test_horizon_hours_is_numeric_and_valid(self):
        df = _make_predictions_df()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "preds.csv"
            write_predictions_csv(df, path)
            written = pd.read_csv(path)

            self.assertTrue(pd.api.types.is_numeric_dtype(written["horizon_hours"]))
            self.assertTrue(set(written["horizon_hours"]).issubset(VALID_HORIZONS))

    def test_uncertainty_km_present_as_number_when_available(self):
        df = _make_predictions_df(uncertainty_km=2.3)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "preds.csv"
            write_predictions_csv(df, path)
            written = pd.read_csv(path)
            self.assertTrue((written["uncertainty_km"] == 2.3).all())

    def test_uncertainty_km_is_empty_not_fabricated_when_unavailable(self):
        df = _make_predictions_df(uncertainty_km=None)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "preds.csv"
            write_predictions_csv(df, path)
            written = pd.read_csv(path)
            # pandas writes None -> NaN for a numeric-ish column; the key
            # guarantee is it is NOT populated with a real-looking number.
            self.assertTrue(written["uncertainty_km"].isna().all())

    def test_json_structure_matches_canonical_nested_schema(self):
        df = _make_predictions_df(uncertainty_km=1.1)
        structure = build_json_structure(df)

        self.assertEqual(len(structure), 1)  # one iceberg/timestamp group
        record = structure[0]
        self.assertEqual(set(record.keys()), {"iceberg_id", "prediction_timestamp", "predictions"})
        self.assertEqual(record["iceberg_id"], "TESTA")
        self.assertEqual(len(record["predictions"]), 2)

        for entry in record["predictions"]:
            self.assertEqual(
                set(entry.keys()),
                {"horizon_hours", "predicted_latitude", "predicted_longitude", "uncertainty_km"},
            )
            self.assertIn(entry["horizon_hours"], VALID_HORIZONS)

    def test_json_uncertainty_is_null_not_fabricated_when_unavailable(self):
        df = _make_predictions_df(uncertainty_km=None)
        structure = build_json_structure(df)
        for entry in structure[0]["predictions"]:
            self.assertIsNone(entry["uncertainty_km"])

    def test_json_artifact_is_valid_json_on_disk(self):
        df = _make_predictions_df()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "preds.json"
            write_predictions_json(df, path)
            with path.open() as f:
                data = json.load(f)  # will raise if malformed
            self.assertIsInstance(data, list)


if __name__ == "__main__":
    unittest.main()
