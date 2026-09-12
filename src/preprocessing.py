"""
preprocessing.py
=================

Basic preprocessing framework for the SIH26059 iceberg trajectory dataset.

Scope for Phase 1
------------------
This module provides SAFE, NON-DESTRUCTIVE preprocessing primitives:
- standardizing column names to logical names (via the resolved mapping)
- parsing timestamps
- sorting observations chronologically within each iceberg
- basic type coercion for latitude/longitude

It does NOT:
- drop rows automatically (validation reports problems; dropping is a
  separate, explicit, opt-in step for later phases)
- interpolate or fabricate any values
- perform feature engineering (that is feature_engineering.py, not built yet)

Every function here is intended to be composable and independently
testable, and to raise clear exceptions instead of failing silently.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

logger = logging.getLogger("sih26059.preprocessing")


class PreprocessingError(Exception):
    """Raised when a preprocessing step cannot proceed safely."""


def standardize_columns(
    df: pd.DataFrame,
    resolved_core: dict[str, str],
    resolved_optional: dict[str, str],
) -> pd.DataFrame:
    """
    Return a copy of df with resolved columns renamed to their logical
    names (e.g. actual column 'Lat_deg' -> 'latitude'), keeping only the
    columns that were actually resolved (core + optional). Unrecognized
    columns are dropped from the RETURNED copy but the original df is
    untouched - callers who need extra columns should keep their own
    reference to the raw dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataframe as loaded.
    resolved_core : dict
        Logical -> actual column name mapping for core fields present.
    resolved_optional : dict
        Logical -> actual column name mapping for optional fields present.

    Returns
    -------
    pd.DataFrame
        New dataframe with columns renamed to logical names.
    """
    all_resolved = {**resolved_core, **resolved_optional}
    if not all_resolved:
        raise PreprocessingError(
            "No columns could be resolved against the config mapping. "
            "Check config/config.yaml `columns` and `environmental` sections "
            "against the actual dataset's column names."
        )

    actual_to_logical = {actual: logical for logical, actual in all_resolved.items()}
    subset = df[list(actual_to_logical.keys())].copy()
    subset = subset.rename(columns=actual_to_logical)

    logger.info(
        "Standardized columns: kept %d of %d original columns -> %s",
        subset.shape[1],
        df.shape[1],
        list(subset.columns),
    )
    return subset


def parse_timestamps(df: pd.DataFrame, timestamp_col: str = "timestamp") -> pd.DataFrame:
    """
    Parse the timestamp column to pandas datetime, without dropping rows.
    Unparseable values become NaT and are reported (not silently dropped).

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing a timestamp column with the given name.
    timestamp_col : str
        Logical name of the timestamp column.

    Returns
    -------
    pd.DataFrame
        Copy of df with the timestamp column converted to datetime64.

    Raises
    ------
    PreprocessingError
        If the timestamp column does not exist.
    """
    if timestamp_col not in df.columns:
        raise PreprocessingError(
            f"Column '{timestamp_col}' not found; cannot parse timestamps. "
            f"Available columns: {list(df.columns)}"
        )

    out = df.copy()
    original_non_null = out[timestamp_col].notna().sum()
    out[timestamp_col] = pd.to_datetime(out[timestamp_col], errors="coerce", utc=False)
    new_non_null = out[timestamp_col].notna().sum()

    failed = original_non_null - new_non_null
    if failed > 0:
        logger.warning(
            "%d timestamp value(s) could not be parsed and were set to NaT. "
            "These rows are NOT dropped automatically - inspect them before proceeding.",
            failed,
        )
    else:
        logger.info("All %d timestamp values parsed successfully.", new_non_null)

    return out


def coerce_numeric(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """
    Coerce the given columns to numeric dtype, without dropping rows.
    Values that cannot be coerced become NaN and are reported.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    columns : list[str]
        Column names to coerce (only those present are processed).

    Returns
    -------
    pd.DataFrame
        Copy of df with specified columns coerced to numeric.
    """
    out = df.copy()
    for col in columns:
        if col not in out.columns:
            logger.debug("coerce_numeric: column '%s' not present, skipping.", col)
            continue
        before_non_null = out[col].notna().sum()
        out[col] = pd.to_numeric(out[col], errors="coerce")
        after_non_null = out[col].notna().sum()
        failed = before_non_null - after_non_null
        if failed > 0:
            logger.warning(
                "%d value(s) in column '%s' could not be coerced to numeric "
                "and were set to NaN.",
                failed,
                col,
            )
    return out


def sort_chronologically(
    df: pd.DataFrame,
    iceberg_id_col: str = "iceberg_id",
    timestamp_col: str = "timestamp",
) -> pd.DataFrame:
    """
    Sort the dataframe by iceberg_id, then by timestamp ascending within
    each iceberg. Required before any temporal-difference computation.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with iceberg_id and timestamp columns.
    iceberg_id_col : str
        Logical name of the iceberg ID column.
    timestamp_col : str
        Logical name of the timestamp column.

    Returns
    -------
    pd.DataFrame
        Sorted copy of df, index reset.

    Raises
    ------
    PreprocessingError
        If either required column is missing.
    """
    for col in (iceberg_id_col, timestamp_col):
        if col not in df.columns:
            raise PreprocessingError(
                f"Column '{col}' not found; cannot sort chronologically. "
                f"Available columns: {list(df.columns)}"
            )

    out = df.sort_values(by=[iceberg_id_col, timestamp_col], kind="mergesort")
    out = out.reset_index(drop=True)
    logger.info("Sorted %d rows by [%s, %s].", len(out), iceberg_id_col, timestamp_col)
    return out


def drop_exact_duplicate_rows(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """
    Identify fully duplicate rows. Per project instructions, rows are NOT
    dropped automatically in this phase - the caller decides. This function
    returns the deduplicated frame AND the count, so the caller can choose
    to use either, but does not mutate silently.

    Returns
    -------
    deduped : pd.DataFrame
        A version with exact duplicates removed (caller opts in to use it).
    n_duplicates : int
        Number of duplicate rows found (beyond the first occurrence).
    """
    n_duplicates = int(df.duplicated(keep="first").sum())
    if n_duplicates > 0:
        logger.warning(
            "%d fully duplicate row(s) detected. NOT removed automatically - "
            "this function returns a deduplicated copy for you to opt into.",
            n_duplicates,
        )
    deduped = df.drop_duplicates(keep="first").reset_index(drop=True)
    return deduped, n_duplicates


def run_basic_preprocessing(
    df: pd.DataFrame,
    resolved_core: dict[str, str],
    resolved_optional: dict[str, str],
    config: dict[str, Any],
) -> pd.DataFrame:
    """
    Orchestrate the Phase-1 preprocessing steps in a safe order:
    1. standardize column names to logical names
    2. parse timestamps
    3. coerce lat/lon (and environmental numerics) to numeric
    4. sort chronologically per iceberg

    Does NOT drop rows, does NOT fabricate values, does NOT interpolate.

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataframe as loaded by data_loader.
    resolved_core : dict
        Logical -> actual column mapping for core fields.
    resolved_optional : dict
        Logical -> actual column mapping for optional fields.
    config : dict
        Parsed project configuration.

    Returns
    -------
    pd.DataFrame
        Cleaned (but not row-filtered) dataframe with logical column names.
    """
    standardized = standardize_columns(df, resolved_core, resolved_optional)

    if "timestamp" in standardized.columns:
        standardized = parse_timestamps(standardized, "timestamp")

    numeric_candidates = [
        c for c in ["latitude", "longitude", "wind_u", "wind_v",
                     "current_u", "current_v", "sea_ice_concentration",
                     "area", "length", "width"]
        if c in standardized.columns
    ]
    standardized = coerce_numeric(standardized, numeric_candidates)

    if "iceberg_id" in standardized.columns and "timestamp" in standardized.columns:
        standardized = sort_chronologically(standardized, "iceberg_id", "timestamp")
    else:
        logger.warning(
            "Skipping chronological sort - 'iceberg_id' and/or 'timestamp' "
            "not present after standardization."
        )

    return standardized
