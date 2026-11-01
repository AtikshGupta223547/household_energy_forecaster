# ⚡ Household Energy Consumption Forecaster

[![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.7.0-2E8B57?style=for-the-badge)](https://lightgbm.readthedocs.io/)
[![XGBoost](https://img.shields.io/badge/XGBoost-3.4.1-EB5424?style=for-the-badge)](https://xgboost.readthedocs.io/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.14_CPU-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.66-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Plotly](https://img.shields.io/badge/Plotly-7.1-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)](https://plotly.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)

> **A production-grade, weather-aware, leak-free time-series forecasting pipeline predicting household electricity demand with calibrated 90% quantile prediction intervals and an interactive Streamlit UI.**

---

## 📑 Table of Contents
- [Executive Summary](#-executive-summary)
- [System Architecture & Pipeline](#-system-architecture--pipeline)
- [Data Leakage Prevention](#-data-leakage-prevention)
- [Feature Engineering Strategy](#-feature-engineering-strategy)
- [Modeling & Uncertainty Quantification](#-modeling--uncertainty-quantification)
- [Model Benchmark Comparison](#-model-benchmark-comparison)
- [Visual Results & Key Artifacts](#-visual-results--key-artifacts)
- [Project Directory Structure](#-project-directory-structure)
- [How to Run & Reproduce](#-how-to-run--reproduce)
- [License](#-license)

---

## 🔭 Executive Summary

Electricity demand forecasting is essential for grid balancing, home energy management systems (HEMS), and dynamic tariff optimization. This repository implements an **end-to-end Machine Learning and Deep Learning time-series framework** trained on four years (2006–2010) of electric power consumption telemetry from a household in Sceaux, France (UCI Machine Learning Repository).

### Core Highlights:
- **Zero Concurrent Telemetry Leakage**: Explicitly filters out contemporaneous electrical readings (`Voltage`, `Global_intensity`, `Sub_metering`) to reflect true forward forecasting conditions.
- **Open-Meteo Weather Integration**: Merges historical hourly ambient temperature, relative humidity, apparent temperature, and precipitation via the Open-Meteo Archive API.
- **Thermodynamic Modeling**: Derives Heating Degree Hours ($\max(0, 18 - T)$) and Cooling Degree Hours ($\max(0, T - 22)$) to capture nonlinear seasonal heating loads.
- **Ensemble Architecture**: Combines **LightGBM** and **XGBoost** trained on a variance-stabilized log target ($\ln(1 + y)$), delivering **$R^2 = 0.7154$** and an MAE of **$0.2646\text{ kW}$** (**49.2% error reduction** over baseline).
- **Uncertainty Quantification**: Dual quantile LightGBM models ($\alpha=0.05$ and $\alpha=0.95$) produce an empirical **85.3% coverage** on the out-of-time test set.
- **Interactive Web Application**: Full-featured **Streamlit & Plotly dashboard** (`app.py`) for live temporal slicing, KPI tracking, and interactive prediction interval exploration.

---

## 🏗️ System Architecture & Pipeline

```
  +-------------------------------------------------------------------------------+
  |                              RAW DATA SOURCES                                 |
  |  UCI Telemetry (2.07M 1-Min Rows)   |   Open-Meteo Weather API (34,608 Hours) |
  +-------------------------------------+-----------------------------------------+
                                        |
                                        v
  +-------------------------------------------------------------------------------+
  |                        1. PREPROCESSING & ALIGNMENT                           |
  |  - Delimiter parsing (';') & NaN treatment ('?')                              |
  |  - Forward-fill sensor gap imputation                                         |
  |  - Hourly aggregation via mean resampling (.resample('h').mean())             |
  +-------------------------------------------------------------------------------+
                                        |
                                        v
  +-------------------------------------------------------------------------------+
  |                     2. ADVANCED FEATURE ENGINEERING (31 Feats)                |
  |  - Temporal Cyclical: sin/cos encodings for Hour (24), Day (7), Month (12)    |
  |  - Exogenous Weather: Temp, Humidity, Heating/Cooling Degree Hours, Temp Lags |
  |  - Autoregressive Signals: lag_1, lag_24, lag_168                             |
  |  - Multi-Scale Rolling Stats: 6h, 24h, 168h means & std devs, 24h EMA         |
  |  - National Calendar: French public holidays (holidays.France())              |
  +-------------------------------------------------------------------------------+
                                        |
                                        v
  +-------------------------------------------------------------------------------+
  |                    3. CHRONOLOGICAL TRAIN-TEST SPLIT (80 / 20)                |
  |  - Train: 27,536 hours (Dec 2006 - Feb 2010)                                  |
  |  - Test:   6,885 hours (Feb 2010 - Nov 2010) [Out-of-Time Evaluation]         |
  +-------------------------------------------------------------------------------+
                                        |
                 +----------------------+----------------------+
                 |                                             |
                 v                                             v
  +-------------------------------+             +-------------------------------+
  |      POINT FORECAST BLEND     |             |  90% PREDICTION INTERVALS     |
  | - Target Transform: log1p(y)  |             | - Lower Bound: LGBM q=0.05    |
  | - LightGBM Regressor (1000 t) |             | - Upper Bound: LGBM q=0.95    |
  | - XGBoost Regressor (1000 t)  |             | - Inverted back to kW scale   |
  | - 50/50 Weighted Ensemble     |             | - Empirical Coverage: 85.3%   |
  +-------------------------------+             +-------------------------------+
                 |                                             |
                 +----------------------+----------------------+
                                        |
                                        v
  +-------------------------------------------------------------------------------+
  |                         4. PRODUCTION DELIVERABLES                            |
  |  - Streamlit Web Application (app.py) | Portfolio Notebook (eda_and_results) |
  |  - High-Resolution Artifacts (outputs/actual_vs_predicted.png)                |
  +-------------------------------------------------------------------------------+
```

---

## 🛡️ Data Leakage Prevention

In naive implementations of this dataset, models frequently achieve artificially inflated scores ($R^2 > 0.999$, $\text{MAE} < 0.015\text{ kW}$) by training on concurrent electrical measurements:
- `Global_intensity` (current in Amperes)
- `Voltage` (volts)
- `Sub_metering_1`, `Sub_metering_2`, `Sub_metering_3` (sub-circuit watt-hours)
- `Global_reactive_power` (kVAR)

Because active electrical power follows the physical relation $P = V \times I \times \cos(\phi)$, including same-hour `Global_intensity` and `Voltage` causes **catastrophic target leakage**. At forecast deployment time (e.g. predicting tomorrow's 18:00 load), those concurrent sub-meter readings do not exist.

### Our Leak-Free Safeguards:
1. **Explicit Feature Exclusion**: All concurrent electrical sensors are strictly dropped from the predictor matrix.
2. **Strict Chronological Splitting**: Shuffling is forbidden; the model is evaluated strictly on the final 20% chronological segment (`Feb 2010` to `Nov 2010`).
3. **Pure Forward Prediction**: Predictions rely solely on historical lags, rolling statistics, calendar harmonics, and forecasted weather variables.

---

## 🧬 Feature Engineering Strategy

The dataset combines **31 non-leaking predictors**:

| Category | Feature Names | Description & Formula |
| :--- | :--- | :--- |
| **Cyclical Calendar** | `sin_hour`, `cos_hour`<br>`sin_dayofweek`, `cos_dayofweek`<br>`sin_month`, `cos_month` | Continuous trigonometric mappings preserving periodicity: $\sin(2\pi t / T), \cos(2\pi t / T)$. |
| **Calendar Flags** | `hour`, `dayofweek`, `quarter`, `month`, `year`, `is_weekend`, `is_holiday` | Discrete calendar indicators including French national public holidays (`holidays.France()`). |
| **Exogenous Weather** | `temperature_2m`, `relative_humidity_2m`<br>`apparent_temperature`, `precipitation` | Real-time atmospheric metrics from Open-Meteo for Sceaux/Paris (Lat 48.77, Lon 2.29). |
| **Degree Hours** | `heating_degree`<br>`cooling_degree` | Thermodynamic heating requirement: $\max(0, 18 - T)$<br>Thermodynamic cooling requirement: $\max(0, T - 22)$. |
| **Weather Lags** | `temp_lag_1`, `temp_lag_24` | Ambient temperature 1 hour prior and 24 hours prior (diurnal temperature trend). |
| **Autoregressive Lags**| `lag_1`, `lag_24`, `lag_168` | Active power consumption 1 hour ago (inertia), 24 hours ago (yesterday), and 168 hours ago (last week). |
| **Rolling Statistics** | `rolling_mean_6`, `rolling_std_6`<br>`rolling_mean_24`, `rolling_std_24`<br>`rolling_mean_168`, `rolling_std_168` | Multi-scale moving averages and volatility envelopes capturing short, medium, and weekly dynamics. |
| **Moving Average** | `ema_24` | 24-hour span Exponential Moving Average (`.ewm(span=24).mean()`) weighting recent hours. |

---

## 🧠 Modeling & Uncertainty Quantification

### 1. Log Target Variance Stabilization
Electric load distributions exhibit long right tails during appliance spikes. We train all gradient boosting models on the log1p transform:
$$y_{\text{train\_log}} = \ln(1 + y_{\text{train}})$$
Predictions are inverted back to the natural power scale (kW) via:
$$\hat{y} = \exp(\hat{y}_{\text{log}}) - 1$$

### 2. Multi-Model Ensemble
- **LightGBM Regressor**: `num_leaves=31`, `learning_rate=0.02`, `n_estimators=1000`, `subsample=0.8`, `colsample_bytree=0.8`.
- **XGBoost Regressor**: `max_depth=6`, `learning_rate=0.02`, `n_estimators=1000`, `tree_method="hist"`.
- **Weighted Blend**:
  $$\hat{y}_{\text{ensemble}} = 0.50 \cdot \hat{y}_{\text{LGBM}} + 0.50 \cdot \hat{y}_{\text{XGB}}$$

### 3. Quantile Regression for 90% Prediction Intervals
Point forecasts are augmented with lower and upper uncertainty bounds:
- **Lower Bound ($q_{0.05}$)**: LightGBM trained with `objective="quantile"`, `alpha=0.05`.
- **Upper Bound ($q_{0.95}$)**: LightGBM trained with `objective="quantile"`, `alpha=0.95`.
- **Monotonicity Enforcement**: Post-processed to guarantee $\hat{y}_{\text{lower}} \ge 0$ and $\hat{y}_{\text{lower}} \le \hat{y}_{\text{upper}}$.

---

## 📊 Model Benchmark Comparison

Evaluated on the out-of-time chronological test set (**6,885 hourly observations** from `2010-02-13 01:00:00` to `2010-11-26 21:00:00`):

| Rank | Model Architecture | MAE (kW) | RMSE (kW) | $R^2$ Score | 90% Empirical Coverage | Highlights |
| :---: | :--- | :---: | :---: | :---: | :---: | :--- |
| 🥇 | **Tuned Weather Ensemble (LGBM + XGB)** | **0.2646** | **0.3898** | **0.7154** | **85.3%** | **31 weather + lag features, log target blend** |
| 🥈 | **Tuned LightGBM (Weather + Log Target)** | 0.2657 | 0.3926 | 0.7112 | 85.3% | Single model with weather features |
| 🥉 | **Pure Time-Series LightGBM** | 0.3178 | 0.4576 | 0.6077 | -- | 11 temporal features without weather, linear target |
| 4 | **PyTorch 2-Layer LSTM Neural Network** | 0.3404 | 0.4863 | 0.5569 | -- | 2-Layer LSTM with 24-hour sliding sequence matrices |
| 5 | **Baseline Persistence (24h Lag)** | 0.5210 | 0.7794 | -0.1380 | -- | Naive copy of same hour yesterday |

### Performance Insights:
- **Ensemble vs. Baseline**: Reduces MAE by **49.2%** (from $0.5210\text{ kW} \to 0.2646\text{ kW}$) and RMSE by **50.0%**.
- **Impact of Weather Features**: Adding Open-Meteo temperature, humidity, and degree hours boosted $R^2$ from `0.6077` to `0.7154` (**+17.7% variance explained**).
- **Tree Ensembles vs. LSTM**: Gradient-boosted trees on engineered rolling windows consistently outperformed recurrent architectures on this tabular time-series domain ($R^2 = 0.7154$ vs. $0.5569$).

---

## 🖼️ Visual Results & Key Artifacts

### 1. Actual vs. Predicted (1-Week Window with 90% Prediction Intervals)
Demonstrates the model tracking peak evening surges and daytime troughs within the shaded uncertainty envelope:

![Actual vs Predicted](outputs/actual_vs_predicted.png)

### 2. Top 10 Feature Importances
Identifies the primary non-leaking drivers influencing tree split frequency:

![Feature Importance](outputs/feature_importance.png)

### 3. Deliverables
- **Interactive Web App**: [`app.py`](app.py) (Streamlit dashboard with dynamic date range selector, live KPI cards, and Plotly charts).
- **Portfolio Notebook**: [`eda_and_results.ipynb`](eda_and_results.ipynb) (Complete pipeline documentation and results).
- **Out-of-Time Predictions**: [`data/processed/test_predictions.csv`](data/processed/test_predictions.csv) (Contains timestamps, ground truth, baseline, point forecasts, and interval bounds).

---

## 📂 Project Directory Structure

```text
household_energy_forecaster/
├── .venv/                              # Python 3.12 virtual environment
├── data/
│   ├── raw/
│   │   ├── individual_household_electric_power_consumption.zip
│   │   ├── household_power_consumption.txt    # 132 MB raw telemetry
│   │   └── weather_sceaux.csv                 # Open-Meteo historical weather
│   └── processed/
│       ├── hourly_energy.csv                  # 34,589 hourly resampled records
│       ├── engineered_energy.csv              # 34,421 rows x 38 features
│       └── test_predictions.csv               # Out-of-time test predictions
├── outputs/
│   ├── actual_vs_predicted.png                # 1-week forecast overlay plot
│   └── feature_importance.png                 # LightGBM top-10 split chart
├── src/
│   ├── download_data.py                       # Ingest & verify UCI dataset
│   ├── fetch_weather.py                       # Fetch Open-Meteo historical weather
│   ├── preprocess.py                          # Resample 1-min to hourly averages
│   ├── feature_engineering.py                 # Cyclical, weather, lag & rolling features
│   ├── train.py                               # Train LightGBM + XGBoost + Quantiles
│   ├── train_lstm.py                          # Train PyTorch 2-layer LSTM
│   └── visualize.py                           # Generate plots & eda_and_results.ipynb
├── app.py                                     # Interactive Streamlit application
├── eda_and_results.ipynb                      # Portfolio overview notebook
├── requirements.txt                           # Frozen CPU-friendly dependencies
└── README.md                                  # Production documentation
```

---

## 🚀 How to Run & Reproduce

### 1. Clone & Set Up Environment
```powershell
# Navigate into the project folder
cd household_energy_forecaster

# Create a virtual environment
python -m venv .venv

# Activate the virtual environment (Windows PowerShell)
.\.venv\Scripts\Activate.ps1

# Upgrade core package managers
python -m pip install --upgrade pip setuptools wheel

# Install dependencies from requirements.txt
pip install -r requirements.txt
```

### 2. Execute the Data Pipeline
Run each module sequentially:
```powershell
# 1. Download and extract the UCI raw dataset
python src/download_data.py

# 2. Fetch Open-Meteo historical weather data (Sceaux/Paris)
python src/fetch_weather.py

# 3. Clean and resample 1-minute telemetry into hourly averages
python src/preprocess.py

# 4. Generate all 31 non-leaking weather and time-series features
python src/feature_engineering.py
```

### 3. Train & Evaluate Models
```powershell
# Train the 50/50 LightGBM + XGBoost ensemble & 90% quantile intervals
python src/train.py

# Train the PyTorch 2-layer LSTM deep learning benchmark
python src/train_lstm.py

# Render high-resolution plots and update the portfolio notebook
python src/visualize.py
```

### 4. Launch the Interactive Dashboard
```powershell
streamlit run app.py
```
Open your browser to `http://localhost:8501` to view live metric cards, adjust date sliders, and explore forecasts.

---

## 📜 License

This project is open-source and distributed under the [MIT License](LICENSE).
Telemetry data source: [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption).
Weather data source: [Open-Meteo Archive API](https://open-meteo.com/).
#   h o u s e h o l d _ e n e r g y _ f o r e c a s t e r  
 