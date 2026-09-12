"""
run_pipeline.py
================

Single entry point tying together the real data -> real features -> real
model -> real evaluation -> canonical output chain. No step here
fabricates data: unsupported horizons are explicitly marked, not forced.

Usage:
    python -m src.run_pipeline [--config path/to/config.yaml]
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import pandas as pd

from src.data_loader import load_and_resolve
from src.preprocessing import run_basic_preprocessing
from src.trajectory_builder import add_lag_features, build_supervised_pairs
from src.feature_engineering import engineer_features, get_model_feature_columns
from src.train import prepare_training_table, train_horizon_models, save_model_bundle, TrainingError
from src.evaluate import evaluate_by_horizon
from src.predict import predict_for_iceberg_horizon
from src.output_writer import write_predictions_csv, write_predictions_json

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("sih26059.run_pipeline")


def run(config_path: str = "config/config.yaml") -> dict:
    t0 = time.time()
    df, config, resolved_core, resolved_optional, missing_core = load_and_resolve(config_path)
    n_lags = config.get("trajectory_builder", {}).get("n_lags", 3)
    horizons = config["prediction"]["horizons_hours"]
    processed = run_basic_preprocessing(df, resolved_core, resolved_optional, config)
    logger.info("Loaded + preprocessed %d real rows, %d icebergs.",
                len(processed), processed["iceberg_id"].nunique())

    with_lags = add_lag_features(processed, n_lags=n_lags)
    pairs = build_supervised_pairs(processed, config)
    logger.info("Built %d candidate supervised pairs (real observations only).", len(pairs))

    features, availability = engineer_features(with_lags, n_lags=n_lags)
    feature_columns = get_model_feature_columns(availability, n_lags=n_lags)
    logger.info("Feature availability: %s", availability)
    logger.info("Model feature columns: %s", feature_columns)

    model_bundles = {}
    validation_metrics = {}
    for h in horizons:
        exact_pairs = int((pairs["matched_horizon_hours"] == h).sum())
        table = prepare_training_table(pairs, features, feature_columns, horizon_hours=h)
        logger.info("Horizon %sh: %d matched pairs, %d rows with complete features.",
                    h, exact_pairs, len(table))
        try:
            bundle = train_horizon_models(table, feature_columns, config, horizon_hours=h)
            model_bundles[h] = bundle
            models_dir = config.get("outputs", {}).get("models_dir", "models")
            save_model_bundle(bundle, models_dir)
            metrics = evaluate_by_horizon(bundle["val_predictions"], horizons=[h])
            validation_metrics.update(metrics)
            logger.info("Horizon %sh trained (%s): n_train=%d n_val=%d MAE=%.3fkm RMSE=%.3fkm",
                        h, bundle["model_backend"], bundle["n_train"], bundle["n_val"],
                        metrics[h]["mae_km"], metrics[h]["rmse_km"])
        except TrainingError as e:
            logger.warning("Horizon %sh: %s", h, e)
            validation_metrics[h] = {"status": "unavailable", "reason": str(e)}

    # Baseline comparison, evaluated on the SAME validation rows used for the ML model
    # at each supported horizon (fair, apples-to-apples real comparison).
    from src.baseline import predict_dead_reckoning
    baseline_metrics = {}
    for h, bundle in model_bundles.items():
        val = bundle["val_predictions"]
        if val is None or val.empty:
            continue
        # Re-derive the SAME chronological validation split used for the ML
        # model at this horizon, so the baseline is compared on identical rows.
        table = prepare_training_table(pairs, features, feature_columns, horizon_hours=h)
        table = table.sort_values("base_timestamp").reset_index(drop=True)
        split_idx = int(len(table) * (1 - config["training"]["validation_fraction_by_time"]))
        val_table = table.iloc[split_idx:]

        preds = []
        for _, row in val_table.iterrows():
            speed = row.get("speed_km_per_hour")
            bearing = row.get("bearing_deg")
            if pd.isna(speed) or pd.isna(bearing):
                continue
            r = predict_dead_reckoning(row["base_latitude"], row["base_longitude"], speed, bearing, h)
            if r.supported:
                preds.append({
                    "horizon_hours": h, "predicted_latitude": r.predicted_latitude,
                    "predicted_longitude": r.predicted_longitude,
                    "target_latitude": row["target_latitude"], "target_longitude": row["target_longitude"],
                })
        if preds:
            baseline_metrics.update(evaluate_by_horizon(pd.DataFrame(preds), horizons=[h]))

    # Canonical predictions: latest real observation per iceberg, all configured horizons.
    latest = features.sort_values("timestamp").groupby("iceberg_id", as_index=False).tail(1)
    records = []
    for _, row in latest.iterrows():
        for h in horizons:
            bundle = model_bundles.get(h)
            rec = predict_for_iceberg_horizon(row, h, bundle, validation_metrics)
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

    metrics_out = {
        "runtime_seconds": round(time.time() - t0, 1),
        "rows_used": int(len(processed)),
        "icebergs_total": int(processed["iceberg_id"].nunique()),
        "horizons_configured": horizons,
        "model_validation_metrics": {str(k): v for k, v in validation_metrics.items()},
        "baseline_validation_metrics": {str(k): v for k, v in baseline_metrics.items()},
        "model_backend": next(iter(model_bundles.values()))["model_backend"] if model_bundles else None,
        "feature_columns": feature_columns,
    }
    metrics_path = Path(config["outputs"]["metrics_path"])
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w") as f:
        json.dump(metrics_out, f, indent=2, default=str)

    logger.info("Pipeline finished in %.1fs. Predictions -> %s. Metrics -> %s.",
                time.time() - t0, out_pred_path, metrics_path)
    return metrics_out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()
    run(args.config)
