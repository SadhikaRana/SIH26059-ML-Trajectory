"""
infer.py
========

Simple inference entry point: loads previously SAVED model bundles
(src/train.py::save_model_bundle) and produces the canonical trajectory
predictions WITHOUT retraining anything.

This is the intended integration surface for the backend/integration
teammate - they should never need to re-run the training pipeline just
to get predictions.

Usage (command line):
    python -m src.infer [--config path/to/config.yaml]

Usage (in-process, e.g. from a backend service):
    from src.infer import run_inference
    predictions_df = run_inference("config/config.yaml")

Behavior:
- For each configured horizon, loads a saved model bundle from
  `outputs.models_dir` if one exists (see train.py::load_model_bundle).
  A horizon with no saved bundle falls back to the dead-reckoning
  baseline automatically - exactly the same fallback behavior as during
  training-time prediction (predict.py is reused unchanged).
- Reads uncertainty_km from the metrics file written by the last training
  run (`outputs.metrics_path`), if present - never recomputes or
  fabricates validation metrics here, since inference does not have
  access to held-out ground truth.
- Writes the same canonical CSV/JSON artifacts as run_pipeline.py, to the
  same configured paths - the output schema is identical either way.

This module does NOT load the raw historical dataset's supervised pairs
and does NOT call trajectory_builder.build_supervised_pairs - inference
only needs each iceberg's latest real observation and its lag history,
not training pairs.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from src.data_loader import load_and_resolve
from src.preprocessing import run_basic_preprocessing
from src.trajectory_builder import add_lag_features
from src.feature_engineering import engineer_features, get_model_feature_columns
from src.train import load_model_bundle
from src.predict import predict_for_iceberg_horizon
from src.output_writer import write_predictions_csv, write_predictions_json

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("sih26059.infer")


def _load_validation_metrics(metrics_path: str) -> Optional[dict]:
    """
    Load real validation metrics from a previous training run's metrics
    JSON, for use as the uncertainty_km source. Returns None (not a
    fabricated default) if the file doesn't exist yet - callers then get
    uncertainty_km=None for every prediction, which is the honest state
    before any training has happened.
    """
    path = Path(metrics_path)
    if not path.exists():
        logger.warning(
            "No metrics file found at %s - uncertainty_km will be unavailable "
            "for every prediction. Run `python -m src.run_pipeline` at least "
            "once to produce real validation metrics.", path,
        )
        return None

    with path.open() as f:
        metrics = json.load(f)

    raw = metrics.get("model_validation_metrics", {})
    # Saved as string keys in JSON; predict.py/uncertainty.py expect int horizons.
    return {int(k): v for k, v in raw.items()}


def run_inference(config_path: str = "config/config.yaml") -> pd.DataFrame:
    """
    Load saved models (if any) and produce canonical predictions for the
    latest real observation of every iceberg, across every configured
    horizon. Writes the canonical CSV/JSON artifacts and returns the
    predictions dataframe.
    """
    df, config, resolved_core, resolved_optional, missing_core = load_and_resolve(config_path)
    if missing_core:
        raise ValueError(f"Cannot run inference - missing required column(s): {missing_core}")

    n_lags = config.get("trajectory_builder", {}).get("n_lags", 3)
    horizons = config["prediction"]["horizons_hours"]
    models_dir = config.get("outputs", {}).get("models_dir", "models")

    processed = run_basic_preprocessing(df, resolved_core, resolved_optional, config)
    with_lags = add_lag_features(processed, n_lags=n_lags)
    features, availability = engineer_features(with_lags, n_lags=n_lags)
    logger.info("Loaded %d real rows, %d icebergs for inference.",
                len(processed), processed["iceberg_id"].nunique())

    model_bundles = {}
    for h in horizons:
        bundle = load_model_bundle(models_dir, h)
        model_bundles[h] = bundle
        if bundle is not None:
            logger.info("Horizon %sh: loaded saved model (%s, trained on %d real samples).",
                        h, bundle["model_backend"], bundle["n_train"])
        else:
            logger.info("Horizon %sh: no saved model found - will use the dead-reckoning "
                        "baseline for this horizon.", h)

    validation_metrics = _load_validation_metrics(config["outputs"]["metrics_path"])

    latest = features.sort_values("timestamp").groupby("iceberg_id", as_index=False).tail(1)
    records = []
    for _, row in latest.iterrows():
        for h in horizons:
            rec = predict_for_iceberg_horizon(row, h, model_bundles.get(h), validation_metrics)
            records.append({
                "iceberg_id": rec["iceberg_id"],
                "prediction_timestamp": rec["prediction_timestamp"],
                "horizon_hours": rec["horizon_hours"],
                "predicted_latitude": rec["predicted_latitude"],
                "predicted_longitude": rec["predicted_longitude"],
                "uncertainty_km": rec["uncertainty_km"],
            })
    predictions_df = pd.DataFrame(records)

    out_pred_path = Path(config["outputs"]["prediction_path"])
    write_predictions_csv(predictions_df, out_pred_path)
    write_predictions_json(predictions_df, out_pred_path.with_suffix(".json"))

    n_model = int((predictions_df["uncertainty_km"].notna()).sum())
    logger.info(
        "Inference complete: %d prediction rows written to %s / %s.",
        len(predictions_df), out_pred_path, out_pred_path.with_suffix(".json"),
    )
    return predictions_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference using saved trajectory models.")
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()
    run_inference(args.config)
