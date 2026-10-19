"""
feature_engineering.py
Generates comprehensive time-series and exogenous weather features for Global_active_power:
1. Loads hourly_energy.csv and merges with Open-Meteo weather_sceaux.csv on datetime index.
2. Calendar Features & Cyclical Encodings:
   - hour, dayofweek, quarter, month, year, is_weekend
   - Sine/Cosine transformations for hour (24), dayofweek (7), and month (12)
3. French Holiday Flag:
   - Binary is_holiday column using the holidays package for France.
4. Exogenous Weather Features:
   - temperature_2m, relative_humidity_2m, apparent_temperature, precipitation
   - Heating Degree Hours: max(0, 18 - temperature_2m)
   - Cooling Degree Hours: max(0, temperature_2m - 22)
   - Weather Lags: temp_lag_1, temp_lag_24
5. Historical Consumption Lag Features:
   - lag_1, lag_24, lag_168 for Global_active_power.
6. Multi-Scale Rolling Statistics:
   - 6-hour, 24-hour, and 168-hour rolling means and standard deviations.
7. Exponential Moving Average:
   - 24-hour span EMA (ewm(span=24).mean()).
8. Drops initial NaN rows created by lag/rolling windows and saves to data/processed/engineered_energy.csv.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import holidays


def get_project_root() -> Path:
    """Returns the project root directory."""
    return Path(__file__).resolve().parent.parent


def engineer_features():
    project_root = get_project_root()
    energy_path = project_root / "data" / "processed" / "hourly_energy.csv"
    weather_path = project_root / "data" / "raw" / "weather_sceaux.csv"
    output_path = project_root / "data" / "processed" / "engineered_energy.csv"

    if not energy_path.exists():
        print(f"ERROR: Processed hourly dataset not found at {energy_path}")
        sys.exit(1)

    if not weather_path.exists():
        print(f"ERROR: Weather dataset not found at {weather_path}. Run src/fetch_weather.py first.")
        sys.exit(1)

    print(f"1. Loading hourly energy dataset from: {energy_path}")
    df_energy = pd.read_csv(
        energy_path,
        parse_dates=["datetime"],
        index_col="datetime"
    )
    print(f"   Energy shape: {df_energy.shape}")

    print(f"2. Loading weather dataset from: {weather_path}")
    df_weather = pd.read_csv(
        weather_path,
        parse_dates=["datetime"],
        index_col="datetime"
    )
    print(f"   Weather shape: {df_weather.shape}")

    # Merge on datetime index
    print("3. Merging energy and weather datasets on datetime...")
    df = df_energy.merge(df_weather, left_index=True, right_index=True, how="left")
    # Forward/backward fill any missing weather rows if any exist
    df[["temperature_2m", "relative_humidity_2m", "apparent_temperature", "precipitation"]] = (
        df[["temperature_2m", "relative_humidity_2m", "apparent_temperature", "precipitation"]].ffill().bfill()
    )
    print(f"   Merged shape: {df.shape}")

    new_features = []

    # Calendar Features
    print("4. Generating calendar features & cyclical encodings...")
    df["hour"] = df.index.hour
    df["dayofweek"] = df.index.dayofweek
    df["quarter"] = df.index.quarter
    df["month"] = df.index.month
    df["year"] = df.index.year
    df["is_weekend"] = (df.index.dayofweek >= 5).astype(int)
    cal_cols = ["hour", "dayofweek", "quarter", "month", "year", "is_weekend"]
    new_features.extend(cal_cols)

    # Cyclical Encodings (sin/cos for hour, dayofweek, month)
    df["sin_hour"] = np.sin(2 * np.pi * df["hour"] / 24.0)
    df["cos_hour"] = np.cos(2 * np.pi * df["hour"] / 24.0)
    df["sin_dayofweek"] = np.sin(2 * np.pi * df["dayofweek"] / 7.0)
    df["cos_dayofweek"] = np.cos(2 * np.pi * df["dayofweek"] / 7.0)
    df["sin_month"] = np.sin(2 * np.pi * df["month"] / 12.0)
    df["cos_month"] = np.cos(2 * np.pi * df["month"] / 12.0)
    cyclical_cols = ["sin_hour", "cos_hour", "sin_dayofweek", "cos_dayofweek", "sin_month", "cos_month"]
    new_features.extend(cyclical_cols)

    # French Holiday Flag
    print("5. Generating French holiday calendar flag...")
    fr_holidays = holidays.France()
    df["is_holiday"] = df.index.map(lambda dt: 1 if dt in fr_holidays else 0).astype(int)
    new_features.append("is_holiday")

    # Exogenous Weather Features
    print("6. Engineering exogenous weather features & degree hours...")
    df["heating_degree"] = np.maximum(0.0, 18.0 - df["temperature_2m"])
    df["cooling_degree"] = np.maximum(0.0, df["temperature_2m"] - 22.0)
    df["temp_lag_1"] = df["temperature_2m"].shift(1)
    df["temp_lag_24"] = df["temperature_2m"].shift(24)
    weather_engineered_cols = [
        "temperature_2m", "relative_humidity_2m", "apparent_temperature", "precipitation",
        "heating_degree", "cooling_degree", "temp_lag_1", "temp_lag_24"
    ]
    new_features.extend(weather_engineered_cols)

    # Historical Consumption Lags
    print("7. Generating historical power consumption lags...")
    df["lag_1"] = df["Global_active_power"].shift(1)
    df["lag_24"] = df["Global_active_power"].shift(24)
    df["lag_168"] = df["Global_active_power"].shift(168)
    lag_cols = ["lag_1", "lag_24", "lag_168"]
    new_features.extend(lag_cols)

    # Multi-Scale Rolling Statistics (6h, 24h, 168h)
    print("8. Generating multi-scale rolling statistics (6h, 24h, 168h)...")
    df["rolling_mean_6"] = df["Global_active_power"].rolling(window=6).mean()
    df["rolling_std_6"] = df["Global_active_power"].rolling(window=6).std()

    df["rolling_mean_24"] = df["Global_active_power"].rolling(window=24).mean()
    df["rolling_std_24"] = df["Global_active_power"].rolling(window=24).std()

    df["rolling_mean_168"] = df["Global_active_power"].rolling(window=168).mean()
    df["rolling_std_168"] = df["Global_active_power"].rolling(window=168).std()
    rolling_cols = [
        "rolling_mean_6", "rolling_std_6",
        "rolling_mean_24", "rolling_std_24",
        "rolling_mean_168", "rolling_std_168",
    ]
    new_features.extend(rolling_cols)

    # Exponential Moving Average (24h span)
    print("9. Generating 24-hour Exponential Moving Average (EMA)...")
    df["ema_24"] = df["Global_active_power"].ewm(span=24).mean()
    new_features.append("ema_24")

    # Clean NaNs and Export
    print("10. Cleaning NaNs and exporting dataset...")
    df.dropna(inplace=True)
    final_shape = df.shape

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path)
    print(f"    Engineered dataset saved to: {output_path}")

    # Validation & Display
    print("\n" + "=" * 75)
    print("WEATHER-ENHANCED FEATURE ENGINEERING SUMMARY")
    print("=" * 75)
    print(f"Final dataset shape: {final_shape}")
    print(f"Total new features engineered ({len(new_features)}):")
    for i in range(0, len(new_features), 4):
        print("  " + ", ".join(new_features[i:i+4]))
    print(f"Remaining NaN values: {df.isna().sum().sum()}")
    print("=" * 75)
    print("\nFirst 3 rows of weather features:")
    print("-" * 75)
    print(df[weather_engineered_cols].head(3).to_string())
    print("-" * 75)


if __name__ == "__main__":
    engineer_features()
