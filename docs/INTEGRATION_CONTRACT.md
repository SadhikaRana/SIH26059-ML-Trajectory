# SIH26059 — Trajectory Module Integration Contract

**Owner:** Person 3 (Trajectory Prediction / ML)
**Consumers:** Person 4 (Risk Mapping / Route Optimization), Person 6 (Backend / Integration)
**Status:** Software implemented and tested against synthetic fixtures. Real
Antarctic dataset from Person 5 has **not yet been delivered** — see
"Real-Data Boundary" at the bottom of this document before relying on any
numbers produced by this module.

---

## 1. INPUT TO THE TRAJECTORY MODULE

The module reads a single tabular file (CSV / Parquet / Excel) from
`data/raw/` (path/format configurable in `config/config.yaml`).

### Required fields

| Logical name | Type | Notes |
|---|---|---|
| `iceberg_id` | string | Unique identifier per iceberg. Any string/categorical value. |
| `timestamp` | datetime-parseable | Any format `pandas.to_datetime` can parse (e.g. ISO 8601). Must be able to sort chronologically per iceberg. |
| `latitude` | float | Degrees, range **[-90, 90]**. |
| `longitude` | float | Degrees, range **[-180, 180]**. |

The **actual column names** in the real dataset do not need to match these
logical names — `config/config.yaml`'s `columns:` section maps logical
name → real column name. If a required field's configured column name is
not found in the actual data, the pipeline halts with an explicit error
naming exactly which field is missing. It does **not** guess or proceed
with partial data.

### Optional fields

| Logical name | Notes |
|---|---|
| `wind_u`, `wind_v` | Wind vector components. Units not assumed — whatever Person 5's dataset uses; not converted internally. |
| `current_u`, `current_v` | Ocean current vector components. |
| `sea_ice_concentration` | Any unit/scale Person 5's dataset provides. |
| `area`, `length`, `width` | Iceberg physical characteristics. |

**These optional fields must NOT be assumed to exist.** The pipeline
detects which are actually present at runtime (`resolve_column_mapping`
in `src/data_loader.py`) and reports availability explicitly
(`feature_engineering.engineer_features` returns an `availability` dict).
A dataset with only the four required fields still produces valid
(if less rich) predictions via the movement-history features and the
dead-reckoning baseline.

---

## 2. OUTPUT FROM THE TRAJECTORY MODULE

### Canonical fields (both CSV and JSON)

| Field | Type | Units / Meaning |
|---|---|---|
| `iceberg_id` | string | Same identifier as the input. |
| `prediction_timestamp` | ISO 8601 datetime string | The timestamp of the **most recent real observation** this prediction was made from — i.e. "now" from the model's point of view, not the predicted future time. |
| `horizon_hours` | integer | How many hours into the future from `prediction_timestamp` this prediction targets. One of the configured horizons (currently 6, 12, 24, 48 — see `config/config.yaml` → `prediction.horizons_hours`). |
| `predicted_latitude` | float, degrees | Predicted position at `prediction_timestamp + horizon_hours`. |
| `predicted_longitude` | float, degrees | Predicted position at `prediction_timestamp + horizon_hours`. |
| `uncertainty_km` | float (kilometres) **or null** | Empirical prediction error estimate for this horizon (see below). **May be null.** |

### CSV column order (exact)

```
iceberg_id,prediction_timestamp,horizon_hours,predicted_latitude,predicted_longitude,uncertainty_km
```

### JSON structure (exact, produced by `src/output_writer.py::build_json_structure`)

```json
[
  {
    "iceberg_id": "IB001",
    "prediction_timestamp": "2026-01-10T18:00:00",
    "predictions": [
      {
        "horizon_hours": 6,
        "predicted_latitude": -65.7,
        "predicted_longitude": 45.5,
        "uncertainty_km": 3.0
      },
      {
        "horizon_hours": 12,
        "predicted_latitude": -66.0,
        "predicted_longitude": 46.2,
        "uncertainty_km": null
      }
    ]
  }
]
```

**All coordinate/uncertainty values above are schema examples only — not
real predictions.** See `schemas/trajectory_prediction.schema.json` for
the machine-checkable version of this structure.

### When `uncertainty_km` is null

`uncertainty_km` is the RMSE (configurable — see `uncertainty.method` in
`config/config.yaml`) of real, held-out validation errors for that
specific horizon. It is **null** whenever:
- No model has been trained yet for that horizon (e.g. real dataset not
  yet available, or that horizon didn't have enough real supervised
  samples — see `training.min_samples_to_train`).
- A model was trained but no validation samples were held out for it.

**`uncertainty_km` is never a placeholder, a guess, or an example value.
A null value is the honest and expected state before/without sufficient
real validation data — it is not a bug.**

### Unsupported horizons

If a horizon cannot be supported (no trained model AND no real movement
history to run the dead-reckoning baseline from — e.g. an iceberg with
only one observation), that (iceberg, horizon) row is **not fabricated**.
Internally this is tracked via a `model_source` field (`"xgboost"` |
`"baseline"` | `"unsupported"`) written to the **debug CSV**
(`outputs/predictions/trajectory_predictions_debug.csv`) alongside a
human-readable `notes` field — this debug file is for troubleshooting
only and is not part of the canonical schema Person 4/6 should parse.
An `"unsupported"` row still appears in the canonical CSV/JSON with
`predicted_latitude`/`predicted_longitude`/`uncertainty_km` all null —
Person 4's consumer code should treat null coordinates as "no prediction
available for this horizon" rather than crashing.

---

## 3. PERSON 4 DEPENDENCY (Risk Mapping / Route Optimization)

Person 4 consumes exactly:

- `iceberg_id`
- `prediction_timestamp`
- `horizon_hours`
- `predicted_latitude`
- `predicted_longitude`
- `uncertainty_km`

from **`outputs/predictions/trajectory_predictions.csv`** (or the
equivalent `.json`).

Intended usage on Person 4's side (as communicated to Person 3):
- `predicted_latitude` / `predicted_longitude` → the predicted trajectory
  point, used to build the risk map / risk zone for that horizon.
- `uncertainty_km` → radius (or similar spatial buffer) around the
  predicted point representing positional uncertainty, used to size the
  uncertainty/risk zone. **When null, Person 4 should treat the
  positional uncertainty for that (iceberg, horizon) as unknown** rather
  than assuming zero uncertainty — treating a null as zero would
  understate risk.
- `horizon_hours` → which time slice of the route-planning horizon this
  risk zone applies to.

---

## 4. PERSON 6 DEPENDENCY (Backend / Integration)

**This is currently a file-based interface, not an API.** No HTTP
endpoint or message queue is implemented in this repository at this
stage — if the overall system needs one, that lives in Person 6's
backend layer, wrapping the artifacts below.

### Files Person 6 needs from this repository

1. **`outputs/predictions/trajectory_predictions.csv`** — canonical CSV,
   regenerated each time `python -m src.run_pipeline` is run.
2. **`outputs/predictions/trajectory_predictions.json`** — canonical
   JSON, same content as the CSV, nested per iceberg.
3. **`schemas/trajectory_prediction.schema.json`** — for validating the
   JSON if Person 6 wants a schema check before ingesting it.

### Callable entry point (if Person 6 wants to invoke this in-process rather than shelling out)

```python
from src.run_pipeline import run

result = run(config_path="config/config.yaml")
# result["predictions_df"]      -> pandas DataFrame, canonical schema + debug columns
# result["csv_path"], result["json_path"] -> Path objects to the written artifacts
# result["trained_bundles"]     -> dict[horizon_hours, model bundle or None]
# result["validation_metrics"]  -> dict[horizon_hours, metrics dict] (real validation only)
```

This is the actual function signature in `src/run_pipeline.py` as of this
writing — nothing here is aspirational or planned-but-not-built.

### Command-line entry point

```bash
python -m src.run_pipeline
```

Reads `config/config.yaml` by default, writes the three output files
above, logs progress and any skipped/unsupported horizons to stdout.

---

## 5. EXAMPLE REQUEST/RESPONSE

There is no request/response cycle in the current file-based design.
The closest analogue is: "the dataset in `data/raw/` at the time
`run_pipeline` executes" (request) → "the CSV/JSON files in
`outputs/predictions/`" (response). Example JSON response shown in
Section 2 above — labelled example values, not real output.

---

## 6. REAL-DATA BOUNDARY (read before trusting any number from this module)

As of this document, **Person 5's real dataset has not been delivered**.
Everything in this repository is implemented and tested against
synthetic, clearly-labelled test fixtures (see `tests/`) — never against
real Antarctic observations.

This means:
- No real MAE/RMSE/accuracy numbers exist yet.
- No real `uncertainty_km` values exist yet (it will be null for every
  row until real training/validation happens).
- The 6h/12h/24h/48h horizons are **targets**, not guarantees — whether
  each is actually supportable depends on the real dataset's temporal
  resolution, which `src/profile_dataset.py` will report once real data
  is loaded.

Once Person 5 delivers the dataset: drop it into `data/raw/`, update
`config/config.yaml`'s `columns:` mapping to the real column names, run
`python -m src.profile_dataset` first to see the real temporal
resolution and horizon feasibility, then run `python -m src.run_pipeline`
to train, evaluate, and produce real predictions.
