"""
fetch_weather.py
Fetches historical hourly weather data for Sceaux / Paris, France from Open-Meteo Archive API:
- Coordinates: Latitude 48.77, Longitude 2.29
- Date Range: 2006-12-16 to 2010-11-26
- Variables: temperature_2m, relative_humidity_2m, apparent_temperature, precipitation
- Output: data/raw/weather_sceaux.csv
"""

import json
import sys
import urllib.request
from pathlib import Path
import pandas as pd

API_URL = (
    "https://archive-api.open-meteo.com/v1/archive?"
    "latitude=48.77&longitude=2.29&"
    "start_date=2006-12-16&end_date=2010-11-26&"
    "hourly=temperature_2m,relative_humidity_2m,apparent_temperature,precipitation"
)


def get_project_root() -> Path:
    """Returns the project root directory."""
    return Path(__file__).resolve().parent.parent


def fetch_weather():
    project_root = get_project_root()
    output_path = project_root / "data" / "raw" / "weather_sceaux.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"1. Fetching historical weather data from Open-Meteo Archive API...")
    print(f"   URL: {API_URL}")

    headers = {"User-Agent": "HouseholdEnergyForecaster/1.0 (academic research)"}
    req = urllib.request.Request(API_URL, headers=headers)

    try:
        with urllib.request.urlopen(req, timeout=60) as response:
            data = json.loads(response.read().decode("utf-8"))
    except Exception as e:
        print(f"ERROR: Failed to fetch weather data: {e}")
        sys.exit(1)

    if "hourly" not in data:
        print(f"ERROR: Unexpected API response format: {list(data.keys())}")
        sys.exit(1)

    print("2. Parsing hourly weather payload...")
    hourly = data["hourly"]
    df = pd.DataFrame(hourly)
    df["datetime"] = pd.to_datetime(df["time"])
    df.set_index("datetime", inplace=True)
    df.drop(columns=["time"], inplace=True)

    print(f"   Fetched {len(df):,} hourly weather records.")
    print(f"   Date range: {df.index.min()} to {df.index.max()}")
    print(f"   Columns: {list(df.columns)}")

    # Forward fill any potential single-hour sensor gaps
    nans = df.isna().sum().to_dict()
    if sum(nans.values()) > 0:
        print(f"   Imputing missing values via forward/backward fill: {nans}")
        df.ffill(inplace=True)
        df.bfill(inplace=True)

    print(f"3. Saving weather data to: {output_path}")
    df.to_csv(output_path)
    print("   Download and save successful!")


if __name__ == "__main__":
    fetch_weather()
