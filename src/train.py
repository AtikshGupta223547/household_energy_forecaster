"""
train.py
Trains and evaluates advanced time-series forecasting models with weather features and 90% prediction intervals:
1. Point Forecast Ensemble:
   - Target log transformation: y_train_log = np.log1p(y_train)
   - LightGBM Regressor (num_leaves=31, lr=0.02, n_estimators=1000)
   - XGBoost Regressor (max_depth=6, lr=0.02, n_estimators=1000)
   - 50/50 Ensemble Blend: y_pred_ensemble = 0.5 * y_pred_lgbm + 0.5 * y_pred_xgb
2. Quantile Regression for 90% Prediction Intervals:
   - Lower bound (5th percentile): LightGBM with objective='quantile', alpha=0.05
   - Upper bound (95th percentile): LightGBM with objective='quantile', alpha=0.95
   - Expm1 inverse transform back to kW scale
3. Evaluation:
   - Metrics (MAE, RMSE, R²) for Baseline, LightGBM, XGBoost, and Ensemble.
   - Empirical coverage probability (% of test points within [lower_bound, upper_bound]).
4. Saves predictions to data/processed/test_predictions.csv and updates outputs/actual_vs_predicted.png.
"""

import sys
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import xgboost as xgb
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Telemetry features explicitly dropped to prevent concurrent leakage
DROPPED_TELEMETRY = [
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


def calculate_metrics(y_true, y_pred):
    """Calculates MAE, RMSE, and R2 score."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return mae, rmse, r2


def plot_actual_vs_predicted(predictions_df: pd.DataFrame, output_image_path: Path):
    """Plots actual, ensemble predictions, and shaded 90% prediction intervals for a 1-week test slice."""
    print(f"\nRe-rendering Actual vs. Predicted plot with 90% Prediction Interval: {output_image_path}")
    week_df = predictions_df.iloc[:168].copy()

    plt.figure(figsize=(14, 6), dpi=300)

    # Shaded 90% Prediction Interval
    plt.fill_between(
        week_df["datetime"],
        week_df["lower_bound"],
        week_df["upper_bound"],
        color="#ff7f0e",
        alpha=0.25,
        label="90% Prediction Interval [q0.05 - q0.95]",
    )

    # Ground truth actual
    plt.plot(
        week_df["datetime"],
        week_df["actual"],
        label="Actual Consumption",
        color="#1f77b4",
        linewidth=2.2,
        alpha=0.95,
    )

    # Ensemble Prediction
    plt.plot(
        week_df["datetime"],
        week_df["ensemble_pred"],
        label="Ensemble Forecast (50/50 LightGBM + XGBoost)",
        color="#d62728",
        linestyle="--",
        linewidth=2.0,
        alpha=0.95,
    )

    plt.title(
        "Household Energy Forecasting: Actual vs. Ensemble Predictions with 90% Prediction Interval (1-Week Window)",
        fontsize=13,
        pad=12,
        fontweight="bold",
    )
    plt.xlabel("Datetime", fontsize=11, fontweight="bold")
    plt.ylabel("Global Active Power (kW)", fontsize=11, fontweight="bold")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(frameon=True, facecolor="white", framealpha=0.9, fontsize=10, loc="upper right")

    ax = plt.gca()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d, %H:%M"))
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()

    output_image_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_image_path, dpi=300)
    plt.close()
    print("   Plot saved successfully.")


def main():
    project_root = get_project_root()
    data_path = project_root / "data" / "processed" / "engineered_energy.csv"
    output_predictions_path = project_root / "data" / "processed" / "test_predictions.csv"
    output_plot_path = project_root / "outputs" / "actual_vs_predicted.png"

    if not data_path.exists():
        print(f"ERROR: Dataset not found at {data_path}")
        sys.exit(1)

    print(f"1. Loading weather-enhanced dataset from: {data_path}")
    df = pd.read_csv(
        data_path,
        parse_dates=["datetime"],
        index_col="datetime",
    )
    print(f"   Loaded shape: {df.shape}")

    # 2. Chronological Train-Test Split (80% Train, 20% Test)
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    print("\n2. Chronological Train-Test Split:")
    print(f"   Train Set: {len(train_df):,} rows | {train_df.index.min()} to {train_df.index.max()}")
    print(f"   Test Set:  {len(test_df):,} rows  | {test_df.index.min()} to {test_df.index.max()}")

    # 3. Features and Target
    target_col = "Global_active_power"
    feature_cols = [
        col for col in df.columns
        if col != target_col and col not in DROPPED_TELEMETRY
    ]

    X_train = train_df[feature_cols]
    y_train = train_df[target_col]
    X_test = test_df[feature_cols]
    y_test = test_df[target_col]

    y_train_log = np.log1p(y_train)

    print(f"\n3. Target Variable: {target_col}")
    print(f"   Total Non-Leaking Predictors ({len(feature_cols)}):")
    for i in range(0, len(feature_cols), 4):
        print("     " + ", ".join(feature_cols[i:i+4]))
    print(f"   Explicitly Dropped Concurrent Telemetry ({len(DROPPED_TELEMETRY)}): {', '.join(DROPPED_TELEMETRY)}")

    # 4. Baseline Persistence Model
    print("\n4. Evaluating Baseline Persistence Model (lag_24)...")
    y_baseline = X_test["lag_24"]
    baseline_mae, baseline_rmse, baseline_r2 = calculate_metrics(y_test, y_baseline)

    # 5. Point Forecast 1: LightGBM on Log Target
    print("\n5. Training LightGBM Point Regressor on log target...")
    model_lgbm = LGBMRegressor(
        num_leaves=31,
        learning_rate=0.02,
        n_estimators=1000,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    model_lgbm.fit(X_train, y_train_log)
    y_pred_lgbm = np.expm1(model_lgbm.predict(X_test))
    lgbm_mae, lgbm_rmse, lgbm_r2 = calculate_metrics(y_test, y_pred_lgbm)

    # 6. Point Forecast 2: XGBoost on Log Target
    print("6. Training XGBoost Point Regressor on log target...")
    model_xgb = xgb.XGBRegressor(
        n_estimators=1000,
        learning_rate=0.02,
        max_depth=6,
        random_state=42,
        n_jobs=-1,
        tree_method="hist",
    )
    model_xgb.fit(X_train, y_train_log)
    y_pred_xgb = np.expm1(model_xgb.predict(X_test))
    xgb_mae, xgb_rmse, xgb_r2 = calculate_metrics(y_test, y_pred_xgb)

    # 7. Ensemble Blend (50/50)
    print("7. Computing Ensemble Blend (0.5 * LightGBM + 0.5 * XGBoost)...")
    y_pred_ensemble = 0.5 * y_pred_lgbm + 0.5 * y_pred_xgb
    ens_mae, ens_rmse, ens_r2 = calculate_metrics(y_test, y_pred_ensemble)

    # 8. Quantile Regression for 90% Prediction Intervals (alpha=0.05 and alpha=0.95)
    print("\n8. Training Quantile Regression Models (alpha=0.05 & alpha=0.95)...")
    print("   Training Lower Bound model (q=0.05)...")
    model_q05 = LGBMRegressor(
        objective="quantile",
        alpha=0.05,
        num_leaves=31,
        learning_rate=0.02,
        n_estimators=1000,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    model_q05.fit(X_train, y_train_log)
    lower_bound = np.maximum(0.0, np.expm1(model_q05.predict(X_test)))

    print("   Training Upper Bound model (q=0.95)...")
    model_q95 = LGBMRegressor(
        objective="quantile",
        alpha=0.95,
        num_leaves=31,
        learning_rate=0.02,
        n_estimators=1000,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    model_q95.fit(X_train, y_train_log)
    upper_bound = np.maximum(lower_bound, np.expm1(model_q95.predict(X_test)))

    # Coverage Probability
    in_interval = (y_test.values >= lower_bound) & (y_test.values <= upper_bound)
    coverage_pct = np.mean(in_interval) * 100
    mean_interval_width = np.mean(upper_bound - lower_bound)

    # 9. Save Predictions & Render Visualizations
    output_predictions_path.parent.mkdir(parents=True, exist_ok=True)
    predictions_df = pd.DataFrame(
        {
            "datetime": X_test.index,
            "actual": y_test.values,
            "baseline_pred": y_baseline.values,
            "lgbm_pred": y_pred_lgbm,
            "xgb_pred": y_pred_xgb,
            "ensemble_pred": y_pred_ensemble,
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
        }
    )
    predictions_df.to_csv(output_predictions_path, index=False)
    print(f"\n9. Predictions & intervals saved to: {output_predictions_path}")

    plot_actual_vs_predicted(predictions_df, output_plot_path)

    # Performance & Uncertainty Summary Table
    print("\n" + "=" * 74)
    print(f"{'WEATHER-ENHANCED MODEL PERFORMANCE & UNCERTAINTY (TEST SET)':^74}")
    print("=" * 74)
    print(f"{'Model':<34} | {'MAE':>10} | {'RMSE':>10} | {'R² Score':>10}")
    print("-" * 74)
    print(f"{'Baseline (Persistence 24h)':<34} | {baseline_mae:>10.4f} | {baseline_rmse:>10.4f} | {baseline_r2:>10.4f}")
    print(f"{'LightGBM (Weather + Log Target)':<34} | {lgbm_mae:>10.4f} | {lgbm_rmse:>10.4f} | {lgbm_r2:>10.4f}")
    print(f"{'XGBoost (Weather + Log Target)':<34} | {xgb_mae:>10.4f} | {xgb_rmse:>10.4f} | {xgb_r2:>10.4f}")
    print(f"{'Ensemble Blend (50/50)':<34} | {ens_mae:>10.4f} | {ens_rmse:>10.4f} | {ens_r2:>10.4f}")
    print("-" * 74)
    mae_imp = ((baseline_mae - ens_mae) / baseline_mae) * 100
    rmse_imp = ((baseline_rmse - ens_rmse) / baseline_rmse) * 100
    print(f"{'Ensemble Improvement vs. Baseline':<34} | {f'{mae_imp:+.1f}%':>10} | {f'{rmse_imp:+.1f}%':>10} | {f'{(ens_r2 - baseline_r2):+.4f}':>10}")
    print("=" * 74)

    print("\n" + "=" * 74)
    print(f"{'90% PREDICTION INTERVAL (UNCERTAINTY ESTIMATION)':^74}")
    print("=" * 74)
    print(f"Target Nominal Coverage:          90.00%")
    print(f"Empirical Coverage on Test Set:   {coverage_pct:.2f}%")
    print(f"Average Interval Width:           {mean_interval_width:.4f} kW")
    print("=" * 74)


if __name__ == "__main__":
    main()
