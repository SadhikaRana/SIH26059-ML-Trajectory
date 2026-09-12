import pandas as pd
from pathlib import Path

raw = Path("data/raw")

historical_path = raw / "historical_iceberg_standardized.csv"
wind_path = raw / "iceberg_wind_2020_2025.csv"
output_path = raw / "historical_iceberg_with_wind.csv"

print("Loading historical dataset...")
historical = pd.read_csv(historical_path)

print("Loading wind dataset...")
wind = pd.read_csv(wind_path)

print(f"Historical rows: {len(historical)}")
print(f"Wind rows: {len(wind)}")

# Keep only the required wind columns.
wind = wind[
    ["iceberg_id", "date", "wind_u10_ms", "wind_v10_ms"]
].copy()

# Make sure the merge keys have the same type.
historical["date"] = pd.to_datetime(historical["date"])
wind["date"] = pd.to_datetime(wind["date"])

# Ensure there is at most one wind record per iceberg/date.
if wind.duplicated(["iceberg_id", "date"]).any():
    raise ValueError("Wind dataset contains duplicate iceberg_id + date records.")

# Left join: preserve every historical trajectory observation.
merged = historical.merge(
    wind,
    on=["iceberg_id", "date"],
    how="left",
    validate="one_to_one"
)

merged.to_csv(output_path, index=False)

print("\nMerge complete.")
print(f"Output: {output_path}")
print(f"Rows: {len(merged)}")
print(f"Columns: {list(merged.columns)}")
print(f"Rows with wind data: {merged['wind_u10_ms'].notna().sum()}")
print(f"Rows without wind data: {merged['wind_u10_ms'].isna().sum()}")