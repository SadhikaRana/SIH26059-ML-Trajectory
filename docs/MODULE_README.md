# SIH26059 — Trajectory Module (Person 3) — Module Documentation

## 1. Person 3 Responsibility

Person 3 owns **Trajectory Prediction / ML**: turning historical +
current iceberg position observations into predicted future positions
(with an uncertainty estimate), consumed downstream by Person 4's
risk-mapping / route-optimization module.

```
Iceberg Detection (Person 2)
        ↓
Historical + Current Iceberg Position
        ↓
Trajectory Prediction (THIS MODULE)
        ↓
Predicted Future Iceberg Positions + Uncertainty
        ↓
Risk Map / Risk Zones (Person 4)
```

## 2. Inputs

Read via `src/data_loader.py`. Required: `iceberg_id`, `timestamp`,
`latitude`, `longitude`. Optional: `wind_u`, `wind_v`, `current_u`,
`current_v`, `sea_ice_concentration`, `area`, `length`, `width` — none
of these are assumed to exist; availability is detected at runtime.
Actual column names are mapped via `config/config.yaml`. Full detail in
`docs/INTEGRATION_CONTRACT.md`.

## 3. Preprocessing

`src/preprocessing.py` (Phase 1, unchanged): standardizes column names to
logical names, parses timestamps (unparseable → NaT, never dropped
silently), coerces numeric types, sorts chronologically per iceberg.
Never drops rows automatically, never interpolates.

`src/profile_dataset.py` (Phase 1, unchanged): full validation +
temporal-resolution report. **Run this first against any new dataset** —
it tells you whether the 6h/12h/24h/48h horizons are even supportable
by the data's actual sampling interval before you try to train anything.

## 4. Trajectory Building

`src/trajectory_builder.py`:
- `add_lag_features(df, n_lags)` — attaches up to `n_lags` prior
  observations (position + elapsed time) per iceberg, via
  `groupby().shift()`, which structurally cannot cross iceberg
  boundaries or look into the future.
- `build_supervised_pairs(df, config)` — for each iceberg, finds real
  (base observation → later real observation) pairs and tags which
  configured horizon (if any) the elapsed time matches, within a
  configurable tolerance. **A pair is only created between two
  observations that were both actually recorded** — nothing is
  interpolated to manufacture a pair at exactly 6h/12h/24h/48h if the
  data's cadence doesn't produce one.

## 5. Feature Engineering

`src/feature_engineering.py`: computes, wherever the source columns
allow, previous position, displacement (km, via `geo_utils`), bearing,
speed, one step further back in history (prior displacement/bearing/
speed), environmental passthrough, and cyclic temporal features (day of
year). Returns an explicit `availability` dict so callers know exactly
which feature groups were computed vs skipped due to missing columns —
this is never silent. `get_model_feature_columns()` turns that
availability dict into the concrete list of columns safe to feed a model.

## 6. Baseline

`src/baseline.py` + `src/geo_utils.py`: a **non-ML** dead-reckoning
baseline. Estimates current speed/bearing from the most recent real
displacement, then projects a constant-velocity position forward by the
requested horizon using standard great-circle destination-point geometry.
If there isn't enough real history (fewer than 2 observations) to
estimate a velocity, the baseline explicitly reports it cannot predict —
it does not guess a direction.

This also serves as the automatic fallback in `predict.py` whenever no
trained ML model exists for a given horizon.

## 7. ML Model

`src/train.py`: XGBoost, one regressor per (horizon, target-axis) pair.
**Design decision:** the model predicts **displacement**
(`delta_latitude`, `delta_longitude`) relative to the base observation,
not absolute future latitude/longitude — reasoning is documented in full
in the module's docstring (physical motivation: displacement is driven by
local wind/current forces, not by absolute position on the globe;
generalizes better across icebergs/regions than memorizing absolute
coordinates). The predicted future position is reconstructed as
`base_position + predicted_displacement`.

Training uses a **chronological** train/validation split
(`chronological_split`) — never random — to prevent temporal leakage,
per project requirements. Training is **refused** (raises
`TrainingError`) if fewer than `training.min_samples_to_train` (default
30, configurable) real supervised samples exist for a horizon — no model
is ever fit on a token amount of data and passed off as trained.

## 8. Prediction

`src/predict.py`: for the latest real observation of every iceberg,
across every configured horizon: use the trained XGBoost model if one
exists and its required features are available on that observation;
otherwise fall back to the baseline; otherwise mark the horizon
unsupported. Every output row carries a `model_source` tag
(`"xgboost"` | `"baseline"` | `"unsupported"`, written to the debug CSV)
so the provenance of every number is always inspectable.

## 9. Evaluation

`src/evaluate.py`: geographic error (MAE/RMSE/median/p95, in km, via
proper geodesic distance — never raw degree differences) computed
**only** for (horizon) combinations where real held-out ground truth
exists. Horizons without ground truth are explicitly marked
`"unavailable"` with the reason `"Not available — real dataset/training
required."` — never silently omitted, never backfilled with a guess.

## 10. Uncertainty

`src/uncertainty.py`: `uncertainty_km` is the empirical RMSE (or another
configurable statistic) of real validation errors for that horizon.
Returns `(None, reason)` whenever no real validation has happened yet —
this is the expected, honest state before real data exists, not a bug.
See the module docstring for documented (not yet implemented) future
work: per-iceberg uncertainty, uncertainty growing with time-since-last-
observation, quantile regression.

## 11. Outputs

`src/output_writer.py` writes:
- `outputs/predictions/trajectory_predictions.csv` — canonical schema.
- `outputs/predictions/trajectory_predictions.json` — canonical nested
  schema (see `schemas/trajectory_prediction.schema.json`).
- `outputs/predictions/trajectory_predictions_debug.csv` — canonical
  columns + `model_source`/`notes`/`uncertainty_note`, for internal
  troubleshooting only, not for Person 4/6 consumption.

Full schema documentation: `docs/INTEGRATION_CONTRACT.md`.

## 12. Person 4 Integration

Person 4 consumes the canonical CSV/JSON's six fields
(`iceberg_id`, `prediction_timestamp`, `horizon_hours`,
`predicted_latitude`, `predicted_longitude`, `uncertainty_km`) to build
predicted trajectories, risk zones, uncertainty zones, and route
cost/risk calculations. See `docs/INTEGRATION_CONTRACT.md` Section 3 for
the full expected usage and the null-handling requirement for
`uncertainty_km`.

## 13. Person 6 Integration

Currently a **file-based interface**: Person 6 reads the CSV/JSON
artifacts from `outputs/predictions/`, or calls `src.run_pipeline.run()`
in-process if integrating directly into a Python backend. No HTTP API is
implemented in this repository. Full detail in
`docs/INTEGRATION_CONTRACT.md` Section 4.

## 14. Current Limitations

- **No real dataset yet.** Everything has been tested against isolated,
  clearly-labelled synthetic fixtures (`tests/`) — never real Antarctic
  observations. See Section 15.
- LSTM/PyTorch is intentionally not implemented — XGBoost is the sole
  primary model per project priority (working prototype over research
  sophistication).
- The output writer's debug CSV is for internal use; it is not validated
  against the JSON schema (only the canonical artifacts are).
- Uncertainty is a single scalar per horizon (empirical RMSE), not a
  per-iceberg or growing-with-recency estimate — documented as future
  work in `src/uncertainty.py`.

## 15. What Requires Person 5's Actual Dataset

- Confirmation of real column names (to finalize `config.yaml`'s
  `columns:`/`environmental:` mappings).
- Confirmation of the real temporal sampling interval — determines which
  of 6h/12h/24h/48h are genuinely trainable/evaluable (run
  `python -m src.profile_dataset` first).
- Real trained models (`python -m src.run_pipeline` will produce them
  once enough real supervised samples exist per horizon).
- Real `uncertainty_km` values (currently null for every prediction,
  since no real validation has occurred).
- Real MAE/RMSE/accuracy numbers for any report or presentation — **none
  currently exist and none should be claimed** until produced by this
  pipeline against the real dataset.
