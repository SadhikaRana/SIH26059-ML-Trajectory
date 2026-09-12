"""
data_loader.py
===============

Robust, format-agnostic data loading for the SIH26059 iceberg trajectory
dataset.

Design goals
------------
- No hardcoded filenames or absolute paths.
- Supports CSV, Parquet, and Excel (.xlsx/.xls).
- Does NOT assume the real dataset's column names - it resolves the
  logical column names (iceberg_id, timestamp, latitude, longitude, ...)
  against config.yaml's mapping, and reports clearly if a required
  column is missing.
- Does NOT fabricate, interpolate, or synthesize any data.
- Raises clear, specific exceptions rather than failing silently.

This module intentionally does NOT perform validation logic (see
preprocessing.py / profile_dataset.py) - it is only responsible for
getting a raw DataFrame safely into memory with informative logging.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import yaml

logger = logging.getLogger("sih26059.data_loader")


class DataLoadError(Exception):
    """Raised when the input dataset cannot be located or read."""


class ConfigError(Exception):
    """Raised when config.yaml is missing, malformed, or missing keys."""


def load_config(config_path: str | Path = "config/config.yaml") -> dict[str, Any]:
    """
    Load and parse the project's YAML configuration file.

    Parameters
    ----------
    config_path : str or Path
        Path to config.yaml, relative to the project root by convention.

    Returns
    -------
    dict
        Parsed configuration dictionary.

    Raises
    ------
    ConfigError
        If the file does not exist or is not valid YAML.
    """
    path = Path(config_path)
    if not path.exists():
        raise ConfigError(
            f"Config file not found at '{path}'. "
            "Expected config/config.yaml relative to project root."
        )

    try:
        with path.open("r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as exc:
        raise ConfigError(f"Failed to parse YAML config at '{path}': {exc}") from exc

    if not isinstance(config, dict):
        raise ConfigError(f"Config at '{path}' did not parse into a dictionary.")

    logger.info("Loaded config from %s", path)
    return config


def _discover_input_file(config: dict[str, Any]) -> Path:
    """
    Determine which file to load based on config.

    If `data.input_filename` is set, use it directly (inside raw_dir).
    Otherwise, search raw_dir for exactly one file matching
    `data.supported_extensions`. Multiple or zero candidates raise an
    error rather than guessing.
    """
    data_cfg = config.get("data", {})
    raw_dir = Path(data_cfg.get("raw_dir", "data/raw"))
    explicit_name = data_cfg.get("input_filename")
    supported_ext = data_cfg.get("supported_extensions", [".csv", ".parquet", ".xlsx", ".xls"])

    if not raw_dir.exists():
        raise DataLoadError(
            f"Raw data directory '{raw_dir}' does not exist. "
            "Create it and place the dataset inside, or fix `data.raw_dir` in config.yaml."
        )

    if explicit_name:
        candidate = raw_dir / explicit_name
        if not candidate.exists():
            raise DataLoadError(
                f"Configured input file '{candidate}' does not exist. "
                "Check `data.input_filename` in config.yaml."
            )
        return candidate

    candidates = [
        p for p in raw_dir.iterdir()
        if p.is_file() and p.suffix.lower() in supported_ext and not p.name.startswith(".")
    ]

    if len(candidates) == 0:
        raise DataLoadError(
            f"No supported data files found in '{raw_dir}'. "
            f"Supported extensions: {supported_ext}. "
            "Place the dataset there, or set `data.input_filename` explicitly in config.yaml."
        )

    if len(candidates) > 1:
        names = ", ".join(p.name for p in candidates)
        raise DataLoadError(
            f"Multiple candidate data files found in '{raw_dir}': {names}. "
            "Set `data.input_filename` in config.yaml to disambiguate."
        )

    logger.info("Auto-discovered input file: %s", candidates[0])
    return candidates[0]


def load_raw_dataset(config: dict[str, Any]) -> pd.DataFrame:
    """
    Load the raw trajectory dataset into a pandas DataFrame based on
    the provided config. Supports CSV, Parquet, and Excel.

    This function performs NO column validation, NO renaming, and NO
    cleaning - it only reads the file. Use preprocessing.py / the
    profiling script afterward.

    Parameters
    ----------
    config : dict
        Parsed configuration (see load_config).

    Returns
    -------
    pd.DataFrame
        Raw dataframe exactly as read from disk.

    Raises
    ------
    DataLoadError
        If the file cannot be found, is empty, or is an unsupported/
        unreadable format.
    """
    file_path = _discover_input_file(config)
    suffix = file_path.suffix.lower()

    logger.info("Loading dataset from %s (format=%s)", file_path, suffix)

    try:
        if suffix == ".csv":
            df = pd.read_csv(file_path)
        elif suffix == ".parquet":
            df = pd.read_parquet(file_path)
        elif suffix in (".xlsx", ".xls"):
            df = pd.read_excel(file_path)
        else:
            raise DataLoadError(
                f"Unsupported file extension '{suffix}' for file '{file_path}'. "
                "Supported: .csv, .parquet, .xlsx, .xls"
            )
    except DataLoadError:
        raise
    except FileNotFoundError as exc:
        raise DataLoadError(f"File disappeared before it could be read: {file_path}") from exc
    except pd.errors.EmptyDataError as exc:
        raise DataLoadError(f"File '{file_path}' is empty.") from exc
    except Exception as exc:  # noqa: BLE001 - we want to surface any reader error clearly
        raise DataLoadError(f"Failed to read '{file_path}': {exc}") from exc

    if df.empty:
        raise DataLoadError(f"Loaded dataset from '{file_path}' but it contains zero rows.")

    logger.info("Loaded raw dataset: %d rows, %d columns", len(df), df.shape[1])
    logger.info("Raw columns found: %s", list(df.columns))

    return df


def resolve_column_mapping(
    df: pd.DataFrame, config: dict[str, Any]
) -> tuple[dict[str, str], dict[str, str], list[str]]:
    """
    Resolve logical column names (from config) against the actual columns
    present in the loaded DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The raw, unmodified dataframe.
    config : dict
        Parsed configuration containing `columns` and `environmental` maps.

    Returns
    -------
    resolved_core : dict[str, str]
        Mapping of logical name -> actual column name, for CORE columns
        that ARE present in the dataframe.
    resolved_optional : dict[str, str]
        Mapping of logical name -> actual column name, for OPTIONAL
        (environmental / characteristic) columns that ARE present.
    missing_core : list[str]
        Logical names of core columns that are configured but NOT found
        in the dataframe. This is reported, not silently ignored.
    """
    actual_columns = set(df.columns)

    core_map: dict[str, str] = config.get("columns", {}) or {}
    env_map: dict[str, str] = config.get("environmental", {}) or {}
    char_map: dict[str, str] = config.get("iceberg_characteristics", {}) or {}
    optional_map = {**env_map, **char_map}

    resolved_core: dict[str, str] = {}
    missing_core: list[str] = []
    for logical_name, actual_name in core_map.items():
        if actual_name is not None and actual_name in actual_columns:
            resolved_core[logical_name] = actual_name
        else:
            missing_core.append(logical_name)

    resolved_optional: dict[str, str] = {}
    for logical_name, actual_name in optional_map.items():
        if actual_name is not None and actual_name in actual_columns:
            resolved_optional[logical_name] = actual_name

    if missing_core:
        logger.warning(
            "Missing REQUIRED core columns (configured name not found in data): %s",
            missing_core,
        )
    logger.info("Resolved core columns: %s", resolved_core)
    logger.info(
        "Resolved optional/environmental columns (%d of %d configured found): %s",
        len(resolved_optional),
        len(optional_map),
        resolved_optional,
    )

    return resolved_core, resolved_optional, missing_core


def load_and_resolve(
    config_path: str | Path = "config/config.yaml",
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, str], dict[str, str], list[str]]:
    """
    Convenience entry point: load config, load raw data, resolve columns.

    Returns
    -------
    df : pd.DataFrame
        Raw dataframe (unmodified).
    config : dict
        Parsed config.
    resolved_core : dict
        Logical -> actual column names for core fields found.
    resolved_optional : dict
        Logical -> actual column names for optional fields found.
    missing_core : list
        Logical names of core fields NOT found in the data.
    """
    config = load_config(config_path)
    df = load_raw_dataset(config)
    resolved_core, resolved_optional, missing_core = resolve_column_mapping(df, config)
    return df, config, resolved_core, resolved_optional, missing_core
