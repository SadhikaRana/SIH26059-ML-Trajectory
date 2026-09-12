"""
tests/fixtures.py
==================

Shared TEST-ONLY fixture builders for the SIH26059 trajectory test suite.

Every fixture here is synthetic and clearly labeled as such. None of it is
ever written into data/raw/, data/interim/, data/processed/, models/, or
outputs/ - fixtures either stay in memory or are written to a
tempfile.TemporaryDirectory() that is cleaned up automatically when each
test finishes. See test_no_fabrication.py for tests that specifically
verify this separation is respected pipeline-wide.

Centralizing fixture construction here (rather than duplicating it across
every test_*.py file) keeps the test suite itself free of the "duplicate
functionality" problem the project's own code is required to avoid.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

# Make `src` importable when tests are run as `python -m unittest discover -s tests`
# or directly as `python tests/test_x.py` from the project root.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def two_iceberg_clean_fixture() -> pd.DataFrame:
    """
    TEST FIXTURE - NOT REAL DATA.

    Two icebergs ("TESTA", "TESTB"), 6 observations each at a clean 6-hour
    cadence, no environmental columns. Used for tests that only need
    well-behaved core trajectory data (chronological ordering, lag
    features, supervised pair construction, geometry).
    """
    rows = []
    base = pd.Timestamp("2026-01-01")
    for iceberg_id, lat0, lon0 in [("TESTA", -65.0, 45.0), ("TESTB", -70.0, 50.0)]:
        for i in range(6):
            rows.append({
                "iceberg_id": iceberg_id,
                "timestamp": (base + pd.Timedelta(hours=6 * i)).isoformat(),
                "latitude": lat0 - i * 0.1,
                "longitude": lon0 + i * 0.1,
            })
    return pd.DataFrame(rows)


def two_iceberg_fixture_with_environmental() -> pd.DataFrame:
    """
    TEST FIXTURE - NOT REAL DATA.

    Same shape as two_iceberg_clean_fixture, plus wind_u/wind_v columns,
    for tests that need to verify optional-column passthrough behavior.
    """
    df = two_iceberg_clean_fixture()
    df["wind_u"] = [1.0 + 0.1 * i for i in range(len(df))]
    df["wind_v"] = [-0.5 + 0.05 * i for i in range(len(df))]
    return df


def single_observation_fixture() -> pd.DataFrame:
    """
    TEST FIXTURE - NOT REAL DATA.

    One iceberg with exactly ONE observation - deliberately insufficient
    history for any movement-based feature or baseline. Used to verify
    the pipeline reports "cannot predict" rather than fabricating a
    velocity/direction it has no evidence for.
    """
    return pd.DataFrame([{
        "iceberg_id": "LONELY01",
        "timestamp": pd.Timestamp("2026-01-01").isoformat(),
        "latitude": -65.0,
        "longitude": 45.0,
    }])


DEFAULT_CONFIG = {
    "columns": {
        "iceberg_id": "iceberg_id", "timestamp": "timestamp",
        "latitude": "latitude", "longitude": "longitude",
    },
    "environmental": {
        "wind_u": "wind_u", "wind_v": "wind_v",
        "current_u": "current_u", "current_v": "current_v",
        "sea_ice_concentration": "sea_ice_concentration",
    },
    "iceberg_characteristics": {"area": "area", "length": "length", "width": "width"},
    "validation": {"latitude_range": [-90, 90], "longitude_range": [-180, 180],
                   "min_observations_per_iceberg": 3},
    "prediction": {"horizons_hours": [6, 12, 24, 48], "horizon_feasibility_tolerance_hours": 1.0},
    "trajectory_builder": {"n_lags": 3, "max_lookahead_multiplier": 1.5},
    "training": {"validation_fraction_by_time": 0.2, "min_samples_to_train": 30},
    "uncertainty": {"method": "rmse"},
}
