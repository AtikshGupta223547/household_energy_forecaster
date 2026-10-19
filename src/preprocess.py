"""
preprocess.py
Preprocesses the UCI Individual Household Electric Power Consumption dataset:
1. Loads raw 1-minute data, handling delimiter and NaN values ('?').
2. Combines Date and Time into a datetime index and drops original columns.
3. Converts numerical columns to float and forward-fills missing values.
4. Resamples data to hourly intervals ('h') using mean aggregation.
5. Saves the clean hourly dataset to data/processed/hourly_energy.csv and validates.
"""

import sys
from pathlib import Path
import pandas as pd

NUMERICAL_COLS = [
    "Global_active_power",
    "Global_reactive_power",
    "Voltage",
    "Global_intensity",
    "Sub_metering_1",
    "Sub_metering_2",
    "Sub_metering_3",
]


def get_project_root() -> Path:
    """Returns the project root directory."""
    return Path(__file__).resolve().parent.parent


def preprocess():
    project_root = get_project_root()
    raw_path = project_root / "data" / "raw" / "household_power_consumption.txt"
    processed_dir = project_root / "data" / "processed"
    output_path = processed_dir / "hourly_energy.csv"

    if not raw_path.exists():
        print(f"ERROR: Raw data file not found at {raw_path}")
        sys.exit(1)

    print(f"1. Loading raw dataset from: {raw_path}")
    df = pd.read_csv(
        raw_path,
        sep=";",
        na_values=["?"],
        low_memory=False,
    )
    raw_shape = df.shape
    print(f"   Raw dataset loaded with shape: {raw_shape}")

    print("2. Parsing and setting datetime index...")
    df["datetime"] = pd.to_datetime(
        df["Date"] + " " + df["Time"],
        format="%d/%m/%Y %H:%M:%S",
    )
    df.set_index("datetime", inplace=True)
    df.drop(columns=["Date", "Time"], inplace=True)
    df.sort_index(inplace=True)

    print("3. Converting numerical columns to float & forward-filling...")
    for col in NUMERICAL_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Forward fill to maintain time-series continuity, backward fill as safety fallback
    df.ffill(inplace=True)
    df.bfill(inplace=True)

    print("4. Resampling to hourly intervals ('h') using mean aggregation...")
    hourly_df = df.resample("h").mean()

    # Forward fill any potential gaps created by entirely missing raw hour blocks
    if hourly_df.isna().sum().sum() > 0:
        print("   Filling gaps for completely unrecorded hours...")
        hourly_df.ffill(inplace=True)
        hourly_df.bfill(inplace=True)

    print(f"5. Saving cleaned hourly dataset to: {output_path}")
    processed_dir.mkdir(parents=True, exist_ok=True)
    hourly_df.to_csv(output_path)

    # Validation
    nan_count = int(hourly_df.isna().sum().sum())
    print("\n" + "=" * 60)
    print("VALIDATION & DATASET SUMMARY")
    print("=" * 60)
    print(f"Start date:                      {hourly_df.index.min()}")
    print(f"End date:                        {hourly_df.index.max()}")
    print(f"Shape before resampling (1-min): {raw_shape[0]:,} rows x {len(NUMERICAL_COLS)} columns")
    print(f"Shape after resampling (hourly): {hourly_df.shape[0]:,} rows x {hourly_df.shape[1]} columns")
    print(f"Total NaN values remaining:      {nan_count}")
    print("=" * 60)

    if nan_count != 0:
        print("ERROR: Validation failed! NaN values detected.")
        sys.exit(1)
    else:
        print("Validation SUCCESSFUL: 0 NaN values remain.")


if __name__ == "__main__":
    preprocess()
