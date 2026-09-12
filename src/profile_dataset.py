"""
profile_dataset.py
===================

Dataset validation + temporal-resolution profiling for the SIH26059
iceberg trajectory dataset.

This is the module that answers, honestly and without fabrication:

    "What does the real dataset actually look like, and can it support
     6h / 12h / 24h / 48h prediction horizons?"

It performs NO modeling. It performs NO row-dropping by default. It
performs NO interpolation. It only inspects and reports.

Run as a script:
    python -m src.profile_dataset
or:
    python src/profile_dataset.py

Output:
    - A console report (via logging)
    - A text report written to `outputs.profiling_report_path` in config.yaml
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

# Support running both as `python src/profile_dataset.py` and as a package module.
try:
    from src.data_loader import load_and_resolve, DataLoadError, ConfigError
    from src.preprocessing import run_basic_preprocessing, PreprocessingError
except ImportError:  # pragma: no cover - fallback for direct script execution
    from data_loader import load_and_resolve, DataLoadError, ConfigError  # type: ignore
    from preprocessing import run_basic_preprocessing, PreprocessingError  # type: ignore

logger = logging.getLogger("sih26059.profile_dataset")


# =============================================================================
# Data structures for the report
# =============================================================================

@dataclass
class ColumnAvailabilityReport:
    resolved_core: dict[str, str]
    resolved_optional: dict[str, str]
    missing_core: list[str]

    @property
    def core_complete(self) -> bool:
        return len(self.missing_core) == 0


@dataclass
class BasicIntegrityReport:
    n_rows: int
    n_columns: int
    n_missing_values_by_column: dict[str, int]
    n_fully_duplicate_rows: int
    n_duplicate_id_timestamp_pairs: int
    n_invalid_latitude: int
    n_invalid_longitude: int
    n_invalid_timestamps: int
    n_unique_icebergs: int
    observations_per_iceberg: dict[str, int]


@dataclass
class TemporalIcebergSummary:
    iceberg_id: str
    n_observations: int
    start: Optional[pd.Timestamp]
    end: Optional[pd.Timestamp]
    median_timestep_hours: Optional[float]
    min_timestep_hours: Optional[float]
    max_timestep_hours: Optional[float]
    is_chronologically_sorted_originally: bool
    n_gaps_over_2x_median: int


@dataclass
class TemporalSummaryReport:
    per_iceberg: list[TemporalIcebergSummary]
    overall_min_timestamp: Optional[pd.Timestamp]
    overall_max_timestamp: Optional[pd.Timestamp]
    overall_median_timestep_hours: Optional[float]
    overall_min_timestep_hours: Optional[float]
    overall_max_timestep_hours: Optional[float]
    horizon_feasibility: dict[int, str] = field(default_factory=dict)


# =============================================================================
# Validation logic
# =============================================================================

def check_basic_integrity(
    df: pd.DataFrame,
    config: dict[str, Any],
) -> BasicIntegrityReport:
    """
    Run non-destructive structural checks on the standardized dataframe.
    Assumes df has already been through standardize_columns (logical names).

    Does not raise on data-quality issues (that's the point - report, don't
    silently discard); raises only if the dataframe structurally cannot be
    checked at all (e.g. completely empty).
    """
    if df.empty:
        raise PreprocessingError("Cannot profile an empty dataframe.")

    n_rows, n_columns = df.shape

    missing_by_col = {col: int(df[col].isna().sum()) for col in df.columns}

    n_fully_duplicate = int(df.duplicated(keep="first").sum())

    if "iceberg_id" in df.columns and "timestamp" in df.columns:
        n_dup_id_ts = int(df.duplicated(subset=["iceberg_id", "timestamp"], keep="first").sum())
    else:
        n_dup_id_ts = -1  # sentinel: cannot be computed

    lat_range = config.get("validation", {}).get("latitude_range", [-90, 90])
    lon_range = config.get("validation", {}).get("longitude_range", [-180, 180])

    if "latitude" in df.columns:
        lat = pd.to_numeric(df["latitude"], errors="coerce")
        n_invalid_lat = int(((lat < lat_range[0]) | (lat > lat_range[1]) | lat.isna()).sum())
    else:
        n_invalid_lat = -1

    if "longitude" in df.columns:
        lon = pd.to_numeric(df["longitude"], errors="coerce")
        n_invalid_lon = int(((lon < lon_range[0]) | (lon > lon_range[1]) | lon.isna()).sum())
    else:
        n_invalid_lon = -1

    if "timestamp" in df.columns:
        n_invalid_ts = int(df["timestamp"].isna().sum())
    else:
        n_invalid_ts = -1

    if "iceberg_id" in df.columns:
        n_unique_icebergs = int(df["iceberg_id"].nunique(dropna=True))
        obs_per_iceberg = df["iceberg_id"].value_counts(dropna=True).to_dict()
        obs_per_iceberg = {str(k): int(v) for k, v in obs_per_iceberg.items()}
    else:
        n_unique_icebergs = -1
        obs_per_iceberg = {}

    return BasicIntegrityReport(
        n_rows=n_rows,
        n_columns=n_columns,
        n_missing_values_by_column=missing_by_col,
        n_fully_duplicate_rows=n_fully_duplicate,
        n_duplicate_id_timestamp_pairs=n_dup_id_ts,
        n_invalid_latitude=n_invalid_lat,
        n_invalid_longitude=n_invalid_lon,
        n_invalid_timestamps=n_invalid_ts,
        n_unique_icebergs=n_unique_icebergs,
        observations_per_iceberg=obs_per_iceberg,
    )


def analyze_temporal_resolution(
    df: pd.DataFrame,
    config: dict[str, Any],
) -> TemporalSummaryReport:
    """
    For each iceberg: sort by timestamp, compute consecutive time
    differences, and summarize median/min/max timestep. Aggregate into
    an overall summary and check horizon feasibility.

    This performs NO interpolation and NO fabrication - only arithmetic
    on the observations actually present.
    """
    if "iceberg_id" not in df.columns or "timestamp" not in df.columns:
        raise PreprocessingError(
            "Temporal analysis requires both 'iceberg_id' and 'timestamp' columns "
            f"after standardization. Available: {list(df.columns)}"
        )

    per_iceberg_summaries: list[TemporalIcebergSummary] = []
    all_timesteps_hours: list[float] = []

    for iceberg_id, group in df.groupby("iceberg_id", dropna=True):
        original_order = group["timestamp"].tolist()
        sorted_group = group.sort_values("timestamp")
        is_sorted_originally = original_order == sorted_group["timestamp"].tolist()

        timestamps = sorted_group["timestamp"].dropna()
        n_obs = len(timestamps)

        if n_obs < 2:
            per_iceberg_summaries.append(
                TemporalIcebergSummary(
                    iceberg_id=str(iceberg_id),
                    n_observations=n_obs,
                    start=timestamps.iloc[0] if n_obs == 1 else None,
                    end=timestamps.iloc[0] if n_obs == 1 else None,
                    median_timestep_hours=None,
                    min_timestep_hours=None,
                    max_timestep_hours=None,
                    is_chronologically_sorted_originally=is_sorted_originally,
                    n_gaps_over_2x_median=0,
                )
            )
            continue

        diffs = timestamps.diff().dropna()
        diffs_hours = diffs.dt.total_seconds() / 3600.0
        median_h = float(diffs_hours.median())
        min_h = float(diffs_hours.min())
        max_h = float(diffs_hours.max())
        n_gaps_over_2x = int((diffs_hours > (2 * median_h)).sum()) if median_h > 0 else 0

        all_timesteps_hours.extend(diffs_hours.tolist())

        per_iceberg_summaries.append(
            TemporalIcebergSummary(
                iceberg_id=str(iceberg_id),
                n_observations=n_obs,
                start=timestamps.iloc[0],
                end=timestamps.iloc[-1],
                median_timestep_hours=median_h,
                min_timestep_hours=min_h,
                max_timestep_hours=max_h,
                is_chronologically_sorted_originally=is_sorted_originally,
                n_gaps_over_2x_median=n_gaps_over_2x,
            )
        )

    valid_starts = [s.start for s in per_iceberg_summaries if s.start is not None]
    valid_ends = [s.end for s in per_iceberg_summaries if s.end is not None]

    overall_min_ts = min(valid_starts) if valid_starts else None
    overall_max_ts = max(valid_ends) if valid_ends else None

    if all_timesteps_hours:
        overall_median_h = float(np.median(all_timesteps_hours))
        overall_min_h = float(np.min(all_timesteps_hours))
        overall_max_h = float(np.max(all_timesteps_hours))
    else:
        overall_median_h = None
        overall_min_h = None
        overall_max_h = None

    horizons = config.get("prediction", {}).get("horizons_hours", [6, 12, 24, 48])
    tolerance = config.get("prediction", {}).get("horizon_feasibility_tolerance_hours", 1.0)
    feasibility: dict[int, str] = {}

    if overall_median_h is None:
        for h in horizons:
            feasibility[h] = "UNKNOWN - insufficient data to determine temporal resolution"
    else:
        for h in horizons:
            if abs(overall_median_h - h) <= tolerance:
                feasibility[h] = (
                    f"LIKELY SUPPORTED - median timestep (~{overall_median_h:.2f}h) "
                    f"closely matches this horizon"
                )
            elif overall_median_h < h and (h % overall_median_h == 0 or True):
                # Horizon is a coarser multiple of a finer sampling rate.
                n_steps = h / overall_median_h
                feasibility[h] = (
                    f"POSSIBLY SUPPORTED via {n_steps:.1f} native steps of "
                    f"~{overall_median_h:.2f}h each - verify exact multiples exist "
                    f"per-iceberg before trusting this horizon"
                )
            else:
                feasibility[h] = (
                    f"NOT SUPPORTED BY NATIVE DATA - median timestep "
                    f"(~{overall_median_h:.2f}h) is coarser than this {h}h horizon. "
                    f"Ground truth at this horizon does NOT exist without interpolation "
                    f"(which this pipeline does not perform)."
                )

    return TemporalSummaryReport(
        per_iceberg=per_iceberg_summaries,
        overall_min_timestamp=overall_min_ts,
        overall_max_timestamp=overall_max_ts,
        overall_median_timestep_hours=overall_median_h,
        overall_min_timestep_hours=overall_min_h,
        overall_max_timestep_hours=overall_max_h,
        horizon_feasibility=feasibility,
    )


def check_environmental_availability(
    resolved_optional: dict[str, str],
    config: dict[str, Any],
) -> dict[str, bool]:
    """
    Report which configured environmental/characteristic columns were
    actually found in the dataset.
    """
    env_map = config.get("environmental", {}) or {}
    char_map = config.get("iceberg_characteristics", {}) or {}
    all_optional_logical_names = list(env_map.keys()) + list(char_map.keys())

    return {name: (name in resolved_optional) for name in all_optional_logical_names}


# =============================================================================
# Report rendering
# =============================================================================

def render_report(
    column_report: ColumnAvailabilityReport,
    integrity_report: BasicIntegrityReport,
    temporal_report: TemporalSummaryReport,
    env_availability: dict[str, bool],
) -> str:
    """Render all findings into a single human-readable text report."""
    lines: list[str] = []
    add = lines.append

    add("=" * 78)
    add("SIH26059 - ICEBERG TRAJECTORY DATASET PROFILE REPORT")
    add("=" * 78)
    add("")

    # --- Column availability ---
    add("-" * 78)
    add("1. COLUMN AVAILABILITY")
    add("-" * 78)
    add(f"Core columns resolved:     {column_report.resolved_core}")
    add(f"Core columns MISSING:      {column_report.missing_core if column_report.missing_core else 'None'}")
    add(f"Optional columns resolved: {column_report.resolved_optional if column_report.resolved_optional else 'None found'}")
    if not column_report.core_complete:
        add("")
        add("*** WARNING: Required core columns are missing. Update config.yaml's")
        add("*** `columns` mapping to match the real dataset's actual column names.")
    add("")

    # --- Basic integrity ---
    add("-" * 78)
    add("2. BASIC INTEGRITY")
    add("-" * 78)
    add(f"Total rows:                          {integrity_report.n_rows}")
    add(f"Total columns:                       {integrity_report.n_columns}")
    add(f"Unique icebergs:                     {integrity_report.n_unique_icebergs}")
    add(f"Fully duplicate rows:                {integrity_report.n_fully_duplicate_rows}")
    add(f"Duplicate (iceberg_id, timestamp):   {integrity_report.n_duplicate_id_timestamp_pairs}")
    add(f"Invalid/out-of-range latitude:       {integrity_report.n_invalid_latitude}")
    add(f"Invalid/out-of-range longitude:      {integrity_report.n_invalid_longitude}")
    add(f"Invalid/unparseable timestamps:      {integrity_report.n_invalid_timestamps}")
    add("")
    add("Missing values by column:")
    for col, n_missing in integrity_report.n_missing_values_by_column.items():
        if n_missing > 0:
            add(f"    {col}: {n_missing}")
    if not any(integrity_report.n_missing_values_by_column.values()):
        add("    (none)")
    add("")
    add("Observations per iceberg:")
    for iceberg_id, count in sorted(integrity_report.observations_per_iceberg.items()):
        add(f"    {iceberg_id}: {count}")
    add("")

    # --- Temporal resolution ---
    add("-" * 78)
    add("3. TEMPORAL RESOLUTION ANALYSIS")
    add("-" * 78)
    add(f"Overall dataset time range: {temporal_report.overall_min_timestamp} -> {temporal_report.overall_max_timestamp}")
    if temporal_report.overall_median_timestep_hours is not None:
        add(f"Overall median timestep:    {temporal_report.overall_median_timestep_hours:.2f} hours")
        add(f"Overall min timestep:       {temporal_report.overall_min_timestep_hours:.2f} hours")
        add(f"Overall max timestep:       {temporal_report.overall_max_timestep_hours:.2f} hours")
    else:
        add("Overall timestep statistics: UNAVAILABLE (fewer than 2 valid observations per iceberg)")
    add("")
    add("Per-iceberg temporal summary:")
    for s in temporal_report.per_iceberg:
        add(f"  Iceberg {s.iceberg_id}:")
        add(f"      observations:        {s.n_observations}")
        add(f"      start -> end:        {s.start} -> {s.end}")
        add(f"      median timestep:     {s.median_timestep_hours}")
        add(f"      min / max timestep:  {s.min_timestep_hours} / {s.max_timestep_hours}")
        add(f"      originally sorted:   {s.is_chronologically_sorted_originally}")
        add(f"      gaps > 2x median:    {s.n_gaps_over_2x_median}")
    add("")

    # --- Horizon feasibility ---
    add("-" * 78)
    add("4. PREDICTION HORIZON FEASIBILITY (6h / 12h / 24h / 48h)")
    add("-" * 78)
    add("NOTE: This is a DIAGNOSTIC assessment based on observed temporal")
    add("resolution ONLY. It does NOT fabricate or interpolate ground truth.")
    add("")
    for horizon, verdict in temporal_report.horizon_feasibility.items():
        add(f"  {horizon}h horizon: {verdict}")
    add("")

    # --- Environmental availability ---
    add("-" * 78)
    add("5. ENVIRONMENTAL / CHARACTERISTIC COLUMN AVAILABILITY")
    add("-" * 78)
    for name, available in env_availability.items():
        add(f"  {name}: {'AVAILABLE' if available else 'NOT AVAILABLE'}")
    add("")

    add("=" * 78)
    add("END OF REPORT - No model has been trained. No metrics were computed.")
    add("No synthetic or fabricated data was used in producing this report.")
    add("=" * 78)

    return "\n".join(lines)


# =============================================================================
# Orchestration
# =============================================================================

def run_profiling_pipeline(config_path: str | Path = "config/config.yaml") -> str:
    """
    Full Phase-1 pipeline: load -> resolve columns -> preprocess (safe,
    non-destructive) -> validate -> profile temporal resolution -> render
    report -> write report to disk.

    Returns
    -------
    str
        The rendered report text (also written to the configured path).
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    logger.info("Starting SIH26059 dataset profiling pipeline...")

    try:
        df, config, resolved_core, resolved_optional, missing_core = load_and_resolve(config_path)
    except (DataLoadError, ConfigError) as exc:
        logger.error("Pipeline halted during loading: %s", exc)
        raise

    column_report = ColumnAvailabilityReport(
        resolved_core=resolved_core,
        resolved_optional=resolved_optional,
        missing_core=missing_core,
    )

    if not column_report.core_complete:
        logger.error(
            "Cannot proceed with full profiling - required core columns missing: %s. "
            "Fix config.yaml `columns` mapping and re-run.",
            missing_core,
        )
        report = render_report(
            column_report,
            BasicIntegrityReport(
                n_rows=len(df), n_columns=df.shape[1],
                n_missing_values_by_column={}, n_fully_duplicate_rows=-1,
                n_duplicate_id_timestamp_pairs=-1, n_invalid_latitude=-1,
                n_invalid_longitude=-1, n_invalid_timestamps=-1,
                n_unique_icebergs=-1, observations_per_iceberg={},
            ),
            TemporalSummaryReport(
                per_iceberg=[], overall_min_timestamp=None, overall_max_timestamp=None,
                overall_median_timestep_hours=None, overall_min_timestep_hours=None,
                overall_max_timestep_hours=None, horizon_feasibility={},
            ),
            {},
        )
        _write_report(report, config)
        return report

    try:
        processed = run_basic_preprocessing(df, resolved_core, resolved_optional, config)
        integrity_report = check_basic_integrity(processed, config)
        temporal_report = analyze_temporal_resolution(processed, config)
        env_availability = check_environmental_availability(resolved_optional, config)
    except PreprocessingError as exc:
        logger.error("Pipeline halted during preprocessing/validation: %s", exc)
        raise

    report_text = render_report(column_report, integrity_report, temporal_report, env_availability)
    _write_report(report_text, config)

    logger.info("Profiling complete.")
    print("\n" + report_text)

    return report_text


def _write_report(report_text: str, config: dict[str, Any]) -> None:
    out_path = Path(
        config.get("outputs", {}).get(
            "profiling_report_path", "outputs/metrics/dataset_profile_report.txt"
        )
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report_text, encoding="utf-8")
    logger.info("Report written to %s", out_path)


if __name__ == "__main__":
    try:
        run_profiling_pipeline()
    except Exception as exc:  # noqa: BLE001 - top-level CLI error surface
        logger.error("Fatal error: %s", exc)
        sys.exit(1)
