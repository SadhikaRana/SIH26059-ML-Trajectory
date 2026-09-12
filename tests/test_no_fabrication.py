"""
tests/test_no_fabrication.py
===============================

THE MOST IMPORTANT TEST FILE IN THIS SUITE.

Verifies the project's hard requirement: nothing in this pipeline invents
a result it has no real evidence for. Every test here asserts that a
"cannot compute this" situation produces an explicit, honest signal
(None / null / a raised error with a clear reason) rather than a
plausible-looking fabricated number.

Covers:
- missing/insufficient real training data does not produce a fake trained
  model or fake metrics (train.train_horizon_models refuses to fit)
- unavailable uncertainty is never replaced by an invented number
  (uncertainty.estimate_uncertainty_km)
- unsupported horizons are not silently fabricated with fake ground truth
  (trajectory_builder.build_supervised_pairs, evaluate.evaluate_by_horizon)
- insufficient movement history is handled explicitly, not guessed
  (baseline.predict_dead_reckoning, predict.predict_for_iceberg_horizon)
"""

from __future__ import annotations

import unittest

import pandas as pd

from tests.fixtures import (
    two_iceberg_clean_fixture, single_observation_fixture, DEFAULT_CONFIG,
)

from src.preprocessing import sort_chronologically
from src.trajectory_builder import add_lag_features, build_supervised_pairs
from src.feature_engineering import engineer_features, get_model_feature_columns
from src.train import prepare_training_table, train_horizon_models, TrainingError
from src.evaluate import evaluate_by_horizon, NOT_AVAILABLE
from src.uncertainty import estimate_uncertainty_km, NOT_AVAILABLE_REASON
from src.baseline import predict_dead_reckoning
from src.predict import predict_for_iceberg_horizon


def _pipeline_up_to_features(df: pd.DataFrame, n_lags: int = 2):
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = sort_chronologically(df, "iceberg_id", "timestamp")
    with_lags = add_lag_features(df, n_lags=n_lags)
    pairs = build_supervised_pairs(df, DEFAULT_CONFIG)
    features, availability = engineer_features(with_lags, n_lags=n_lags)
    feature_columns = get_model_feature_columns(availability, n_lags=n_lags)
    return df, pairs, features, feature_columns


class TestNoFabricationInTraining(unittest.TestCase):

    def test_insufficient_real_samples_refuses_to_train(self):
        """A tiny synthetic fixture (well below min_samples_to_train) must
        cause training to REFUSE, not silently fit a model on too little
        data and report it as trained."""
        df, pairs, features, feature_columns = _pipeline_up_to_features(two_iceberg_clean_fixture())

        config = dict(DEFAULT_CONFIG)
        config["training"] = {"validation_fraction_by_time": 0.2, "min_samples_to_train": 30}

        training_table = prepare_training_table(pairs, features, feature_columns, horizon_hours=6)
        # Our tiny fixture has far fewer than 30 real samples for 6h.
        self.assertLess(len(training_table), 30)

        with self.assertRaises(TrainingError) as ctx:
            train_horizon_models(training_table, feature_columns, config, horizon_hours=6)

        self.assertIn("Not available", str(ctx.exception))

    def test_zero_samples_also_refuses_cleanly(self):
        """An iceberg with only one observation produces zero supervised
        pairs - training must refuse rather than crash or fabricate."""
        df, pairs, features, feature_columns = _pipeline_up_to_features(
            single_observation_fixture(), n_lags=1
        )
        self.assertTrue(pairs.empty)

        training_table = prepare_training_table(pairs, features, feature_columns, horizon_hours=6)
        self.assertTrue(training_table.empty)

        with self.assertRaises(TrainingError):
            train_horizon_models(training_table, feature_columns, DEFAULT_CONFIG, horizon_hours=6)


class TestNoFabricationInUncertainty(unittest.TestCase):

    def test_uncertainty_unavailable_when_no_validation_has_run(self):
        """Before any model has been trained/validated, uncertainty_km
        MUST be None, never a placeholder or guessed number."""
        value, reason = estimate_uncertainty_km(6, validation_metrics_by_horizon=None)
        self.assertIsNone(value)
        self.assertIn("Not available", reason)

    def test_uncertainty_unavailable_for_horizon_with_no_ground_truth(self):
        """Even if OTHER horizons have real validation results, a horizon
        with none of its own must not borrow or invent a number."""
        validation_metrics = {
            6: {"status": "ok", "n_samples": 50, "mae_km": 1.0, "rmse_km": 1.2,
                "median_km": 0.9, "p95_km": 2.0},
            48: {"status": "unavailable", "reason": NOT_AVAILABLE},
        }
        value, reason = estimate_uncertainty_km(48, validation_metrics)
        self.assertIsNone(value)

        # The horizon that DOES have real validation should return a real number.
        value6, reason6 = estimate_uncertainty_km(6, validation_metrics, method="rmse")
        self.assertEqual(value6, 1.2)
        self.assertIn("Empirical", reason6)

    def test_uncertainty_reason_always_accompanies_the_value(self):
        """Provenance must always be attached - a bare number without an
        explanation of where it came from is not acceptable output."""
        validation_metrics = {
            6: {"status": "ok", "n_samples": 10, "mae_km": 0.5, "rmse_km": 0.6,
                "median_km": 0.4, "p95_km": 1.1},
        }
        value, reason = estimate_uncertainty_km(6, validation_metrics)
        self.assertIsNotNone(value)
        self.assertTrue(len(reason) > 0)


class TestNoFabricationInEvaluation(unittest.TestCase):

    def test_horizon_with_no_ground_truth_marked_unavailable(self):
        """evaluate_by_horizon must explicitly mark a horizon unavailable
        rather than silently omitting it or fabricating a metric."""
        empty_predictions = pd.DataFrame(columns=[
            "horizon_hours", "predicted_latitude", "predicted_longitude",
            "target_latitude", "target_longitude",
        ])
        results = evaluate_by_horizon(empty_predictions, horizons=[6, 12, 24, 48])

        for horizon in (6, 12, 24, 48):
            self.assertEqual(results[horizon]["status"], "unavailable")
            self.assertEqual(results[horizon]["reason"], NOT_AVAILABLE)

    def test_horizon_with_real_ground_truth_computes_real_metrics(self):
        predictions = pd.DataFrame([
            {"horizon_hours": 6, "predicted_latitude": -65.0, "predicted_longitude": 45.0,
             "target_latitude": -65.01, "target_longitude": 45.01},
            {"horizon_hours": 6, "predicted_latitude": -66.0, "predicted_longitude": 46.0,
             "target_latitude": -66.02, "target_longitude": 46.01},
        ])
        results = evaluate_by_horizon(predictions, horizons=[6])
        self.assertEqual(results[6]["status"], "ok")
        self.assertGreater(results[6]["mae_km"], 0)
        self.assertEqual(results[6]["n_samples"], 2)


class TestNoFabricationInBaselineAndPredict(unittest.TestCase):

    def test_baseline_refuses_without_real_movement_history(self):
        result = predict_dead_reckoning(-65.0, 45.0, None, None, horizon_hours=6)
        self.assertFalse(result.supported)
        self.assertIsNone(result.predicted_latitude)
        self.assertIsNone(result.predicted_longitude)

    def test_predict_marks_unsupported_when_no_model_and_no_history(self):
        """The end-to-end predict path: an iceberg with a single real
        observation, no trained model for the horizon, must come back
        explicitly unsupported with predicted_latitude/longitude as None -
        never a guessed position."""
        row = pd.Series({
            "iceberg_id": "LONELY01",
            "timestamp": pd.Timestamp("2026-01-01"),
            "latitude": -65.0,
            "longitude": 45.0,
            "speed_km_per_hour": float("nan"),
            "bearing_deg": float("nan"),
        })
        record = predict_for_iceberg_horizon(
            row, horizon_hours=6, trained_model_bundle=None,
            validation_metrics_by_horizon=None,
        )

        self.assertEqual(record["model_source"], "unsupported")
        self.assertIsNone(record["predicted_latitude"])
        self.assertIsNone(record["predicted_longitude"])
        self.assertIsNone(record["uncertainty_km"])

    def test_predict_falls_back_to_baseline_when_history_exists_but_no_model(self):
        """With real movement history but no trained model, predict.py must
        use the (real, non-fabricated) dead-reckoning baseline - not
        report unsupported, and not silently fabricate an ML-looking
        result either. model_source must correctly say 'baseline'."""
        row = pd.Series({
            "iceberg_id": "TESTA",
            "timestamp": pd.Timestamp("2026-01-01"),
            "latitude": -65.0,
            "longitude": 45.0,
            "speed_km_per_hour": 1.5,
            "bearing_deg": 90.0,
        })
        record = predict_for_iceberg_horizon(
            row, horizon_hours=6, trained_model_bundle=None,
            validation_metrics_by_horizon=None,
        )

        self.assertEqual(record["model_source"], "baseline")
        self.assertIsNotNone(record["predicted_latitude"])
        # No validation has run, so uncertainty must still be unavailable
        # even though a position WAS predicted.
        self.assertIsNone(record["uncertainty_km"])

    def test_matched_horizon_is_none_when_temporal_resolution_does_not_support_it(self):
        """If the real dataset's cadence can't produce a pair near a
        requested horizon, matched_horizon_hours must be None for that
        pair, not a fabricated match."""
        df = two_iceberg_clean_fixture()  # 6h cadence
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = sort_chronologically(df, "iceberg_id", "timestamp")

        # Ask for a horizon (5h) that the 6h-cadence fixture cannot exactly hit,
        # with a tight tolerance that excludes the real 6h gap.
        config = {
            "prediction": {"horizons_hours": [5], "horizon_feasibility_tolerance_hours": 0.1},
            "trajectory_builder": {"max_lookahead_multiplier": 2.0},
        }
        pairs = build_supervised_pairs(df, config)
        self.assertTrue((pairs["matched_horizon_hours"].isna()).all())


if __name__ == "__main__":
    unittest.main()
