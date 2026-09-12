"""
trajectory_builder.py
======================

Builds lag features (past-only, never crossing iceberg boundaries) and
real supervised (base -> target) pairs for training/evaluation. Never
interpolates or fabricates a target: a pair only exists if a genuine
future observation for the SAME iceberg exists at (approximately) the
requested horizon.
"""

from __future__ import annotations

import pandas as pd
import numpy as np


class TrajectoryBuilderError(Exception):
    """Raised when input assumptions (e.g. chronological sort) are violated."""


def _assert_sorted(df: pd.DataFrame) -> None:
    for _, g in df.groupby("iceberg_id", sort=False):
        if not g["timestamp"].is_monotonic_increasing:
            raise TrajectoryBuilderError(
                "Input is not sorted chronologically within each iceberg_id "
                "group. Call preprocessing.sort_chronologically first."
            )


def add_lag_features(df: pd.DataFrame, n_lags: int = 3) -> pd.DataFrame:
    _assert_sorted(df)
    out = df.copy()
    for k in range(1, n_lags + 1):
        lat_col, lon_col, ts_col, dh_col = (
            f"lag{k}_latitude", f"lag{k}_longitude", f"lag{k}_timestamp", f"lag{k}_delta_hours",
        )
        grouped = out.groupby("iceberg_id", sort=False)
        out[lat_col] = grouped["latitude"].shift(k)
        out[lon_col] = grouped["longitude"].shift(k)
        out[ts_col] = grouped["timestamp"].shift(k)
        out[dh_col] = (out["timestamp"] - out[ts_col]).dt.total_seconds() / 3600.0
    return out


def build_supervised_pairs(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    For every (base observation, later real observation) within the same
    iceberg, emit one candidate pair tagged with the configured horizon it
    matches (within horizon_feasibility_tolerance_hours), or NaN if it
    doesn't match any configured horizon. Only real observed rows are ever
    used - no interpolation.

    Search window per base row is bounded by
    max(horizons_hours) * max_lookahead_multiplier hours, and rows are
    scanned in chronological order so we can stop early once a candidate
    falls outside the window (monotonic timestamps).
    """
    horizons = config["prediction"]["horizons_hours"]
    tolerance = config["prediction"]["horizon_feasibility_tolerance_hours"]
    multiplier = config.get("trajectory_builder", {}).get("max_lookahead_multiplier", 1.5)
    max_window_hours = max(horizons) * multiplier

    df_reset = df.reset_index(drop=True)
    records = []

    for _, g in df_reset.groupby("iceberg_id", sort=False):
        idx = g.index.to_numpy()
        ts = g["timestamp"].to_numpy()
        n = len(g)
        for i in range(n):
            base_idx = idx[i]
            base_ts = ts[i]
            for j in range(i + 1, n):
                target_idx = idx[j]
                target_ts = ts[j]
                delta_hours = (target_ts - base_ts) / np.timedelta64(1, "h")
                if delta_hours > max_window_hours:
                    break  # timestamps are monotonic increasing; nothing further fits
                matched_horizon = np.nan
                best_diff = None
                for h in horizons:
                    diff = abs(delta_hours - h)
                    if diff <= tolerance and (best_diff is None or diff < best_diff):
                        matched_horizon = h
                        best_diff = diff
                records.append({
                    "iceberg_id": g["iceberg_id"].iloc[0],
                    "base_index": base_idx,
                    "target_index": target_idx,
                    "base_timestamp": base_ts,
                    "target_timestamp": target_ts,
                    "delta_hours": delta_hours,
                    "matched_horizon_hours": matched_horizon,
                })

    columns = [
        "iceberg_id", "base_index", "target_index", "base_timestamp",
        "target_timestamp", "delta_hours", "matched_horizon_hours",
    ]
    if not records:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(records, columns=columns)
