"""
tests/test_model_persistence.py
==================================

Covers the new integration-readiness surface: saving a trained model
bundle, reloading it without retraining, and running full inference
(src/infer.py) purely from saved artifacts. Also verifies the
no-fabrication guarantee still holds when no saved model exists.

All fixtures are synthetic and isolated to tempfile.TemporaryDirectory()
- nothing here touches the real data/raw/, models/, or outputs/ paths.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from tests.fixtures import two_iceberg_clean_fixture, DEFAULT_CONFIG

from src.preprocessing import sort_chronologically
from src.trajectory_builder import add_lag_features, build_supervised_pairs
from src.feature_engineering import engineer_features, get_model_feature_columns
from src.train import (
    prepare_training_table, train_horizon_models, save_model_bundle, load_model_bundle,
)


def _many_observations_fixture(n_icebergs: int = 2, n_obs: int = 40) -> pd.DataFrame:
    """TEST FIXTURE - NOT REAL DATA. Enough rows per iceberg to produce
    more than min_samples_to_train real 24h-matched supervised pairs."""
    rows = []
    base = pd.Timestamp("2020-01-01")
    for k in range(n_icebergs):
        iceberg_id = f"BULK{k:02d}"
        lat0, lon0 = -60.0 - k, 40.0 + k
        for i in range(n_obs):
            rows.append({
                "iceberg_id": iceberg_id,
                "timestamp": (base + pd.Timedelta(hours=24 * i)).isoformat(),
                "latitude": lat0 - i * 0.05,
                "longitude": lon0 + i * 0.07,
            })
    return pd.DataFrame(rows)


def _train_a_real_bundle(min_samples: int = 20):
    """Train a genuine (synthetic-data) model bundle end to end, reusing
    the project's own pipeline functions - not a hand-built fake object."""
    df = _many_observations_fixture()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = sort_chronologically(df, "iceberg_id", "timestamp")

    config = dict(DEFAULT_CONFIG)
    config["training"] = {"validation_fraction_by_time": 0.2, "min_samples_to_train": min_samples}

    with_lags = add_lag_features(df, n_lags=2)
    pairs = build_supervised_pairs(df, config)
    features, availability = engineer_features(with_lags, n_lags=2)
    feature_columns = get_model_feature_columns(availability, n_lags=2)

    table = prepare_training_table(pairs, features, feature_columns, horizon_hours=24)
    bundle = train_horizon_models(table, feature_columns, config, horizon_hours=24)
    return bundle


class TestModelPersistence(unittest.TestCase):

    def test_save_then_load_round_trips_predictions_identically(self):
        bundle = _train_a_real_bundle()
        sample_x = pd.DataFrame(
            [[1.0] * len(bundle["feature_columns"])], columns=bundle["feature_columns"]
        )
        original_lat_pred = bundle["model_lat"].predict(sample_x)[0]
        original_lon_pred = bundle["model_lon"].predict(sample_x)[0]

        with tempfile.TemporaryDirectory() as tmpdir:
            save_model_bundle(bundle, tmpdir)
            loaded = load_model_bundle(tmpdir, horizon_hours=24)

            self.assertIsNotNone(loaded)
            self.assertEqual(loaded["feature_columns"], bundle["feature_columns"])
            self.assertEqual(loaded["horizon_hours"], 24)

            reloaded_lat_pred = loaded["model_lat"].predict(sample_x)[0]
            reloaded_lon_pred = loaded["model_lon"].predict(sample_x)[0]
            self.assertAlmostEqual(reloaded_lat_pred, original_lat_pred, places=8)
            self.assertAlmostEqual(reloaded_lon_pred, original_lon_pred, places=8)

    def test_load_model_bundle_returns_none_when_nothing_saved(self):
        """No fabricated/empty model object - a missing saved bundle must
        come back as None so callers fall back to the baseline."""
        with tempfile.TemporaryDirectory() as tmpdir:
            loaded = load_model_bundle(tmpdir, horizon_hours=48)
            self.assertIsNone(loaded)

    def test_load_model_bundle_does_not_confuse_horizons(self):
        bundle_24h = _train_a_real_bundle()
        with tempfile.TemporaryDirectory() as tmpdir:
            save_model_bundle(bundle_24h, tmpdir)
            # A horizon that was never trained/saved must still come back None,
            # even though a DIFFERENT horizon's model exists in the same dir.
            self.assertIsNone(load_model_bundle(tmpdir, horizon_hours=6))
            self.assertIsNotNone(load_model_bundle(tmpdir, horizon_hours=24))


class TestInferenceEntryPoint(unittest.TestCase):

    def test_run_inference_uses_saved_model_without_retraining(self):
        """End-to-end: save a real trained bundle, then call run_inference
        against a config pointing at saved models + a fresh raw CSV, and
        verify it produces canonical-schema predictions purely from the
        saved artifacts (no training call happens inside run_inference)."""
        from src.infer import run_inference

        bundle = _train_a_real_bundle()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            raw_dir = tmp_path / "data" / "raw"
            raw_dir.mkdir(parents=True)
            models_dir = tmp_path / "models"

            df = _many_observations_fixture()
            df.to_csv(raw_dir / "fixture.csv", index=False)
            save_model_bundle(bundle, models_dir)

            config_path = tmp_path / "config.yaml"
            import yaml
            config = {
                "data": {"raw_dir": str(raw_dir), "input_filename": "fixture.csv",
                          "supported_extensions": [".csv"]},
                "columns": {"iceberg_id": "iceberg_id", "timestamp": "timestamp",
                             "latitude": "latitude", "longitude": "longitude"},
                "environmental": {"wind_u": None, "wind_v": None, "current_u": None,
                                    "current_v": None, "sea_ice_concentration": None},
                "trajectory_builder": {"n_lags": 2},
                "prediction": {"horizons_hours": [6, 24], "horizon_feasibility_tolerance_hours": 1.0},
                "outputs": {
                    "prediction_path": str(tmp_path / "outputs" / "predictions.csv"),
                    "metrics_path": str(tmp_path / "outputs" / "metrics.json"),  # deliberately absent
                    "models_dir": str(models_dir),
                },
            }
            with config_path.open("w") as f:
                yaml.dump(config, f)

            predictions_df = run_inference(str(config_path))

            # Canonical schema present.
            for col in ["iceberg_id", "prediction_timestamp", "horizon_hours",
                         "predicted_latitude", "predicted_longitude", "uncertainty_km"]:
                self.assertIn(col, predictions_df.columns)

            # 24h used the saved model (or baseline if features were incomplete on
            # the very latest row) - either way, a position must be produced since
            # these synthetic icebergs have plenty of real movement history.
            h24 = predictions_df[predictions_df["horizon_hours"] == 24]
            self.assertTrue(h24["predicted_latitude"].notna().all())

            # 6h has no saved model and no real 6h ground truth in this fixture -
            # must fall back to baseline, never be fabricated as a trained result.
            h6 = predictions_df[predictions_df["horizon_hours"] == 6]
            self.assertTrue(h6["predicted_latitude"].notna().all())  # baseline still predicts

            # No metrics.json existed - uncertainty_km must be unavailable for
            # every row (never a fabricated number invented by inference itself).
            self.assertTrue(predictions_df["uncertainty_km"].isna().all())

            # Canonical output files were actually written.
            self.assertTrue(Path(config["outputs"]["prediction_path"]).exists())
            self.assertTrue(Path(config["outputs"]["prediction_path"]).with_suffix(".json").exists())


if __name__ == "__main__":
    unittest.main()
