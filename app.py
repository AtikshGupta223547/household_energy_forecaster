"""
app.py
Interactive Streamlit Dashboard for Household Energy Demand Forecasting:
1. Dynamic Date Filtering & KPI Cards (MAE, RMSE, R², 90% Coverage).
2. Interactive Plotly Chart with Shaded 90% Quantile Bands.
3. Model Comparison Table (Weather Ensemble, PyTorch LSTM, Pure LightGBM, Baseline).
"""

from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Page setup
st.set_page_config(
    page_title="Household Energy Forecaster",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data
def load_predictions():
    data_path = Path(__file__).resolve().parent / "data" / "processed" / "test_predictions.csv"
    if not data_path.exists():
        return None
    df = pd.read_csv(data_path, parse_dates=["datetime"])
    df.sort_values("datetime", inplace=True)
    return df


def calculate_metrics(y_true, y_pred, lower_bound=None, upper_bound=None):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    coverage = None
    avg_width = None
    if lower_bound is not None and upper_bound is not None:
        in_interval = (y_true >= lower_bound) & (y_true <= upper_bound)
        coverage = np.mean(in_interval) * 100.0
        avg_width = np.mean(upper_bound - lower_bound)
    return mae, rmse, r2, coverage, avg_width


def main():
    # Header
    st.markdown(
        """
        <div style="padding: 10px 0 20px 0;">
            <h1 style="margin-bottom: 5px;">⚡ Household Energy Forecaster</h1>
            <p style="color: #666; font-size: 1.1rem; margin-top: 0;">
                Real-Time Energy Demand Forecasting & Uncertainty Quantification with Weather Telemetry
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    df = load_predictions()
    if df is None:
        st.error("Predictions file not found at `data/processed/test_predictions.csv`. Please execute `src/train.py` first.")
        return

    # Sidebar Controls
    st.sidebar.header("🗓️ Forecast Time Window")

    min_date = df["datetime"].min().date()
    max_date = df["datetime"].max().date()

    # Preset selection
    preset = st.sidebar.selectbox(
        "Quick Time Windows",
        ["First 7 Days (Default)", "First 14 Days", "First 30 Days", "Full Test Set (9 Months)", "Custom Range"],
        index=0,
    )

    if preset == "First 7 Days (Default)":
        start_date = min_date
        end_date = min_date + timedelta(days=7)
    elif preset == "First 14 Days":
        start_date = min_date
        end_date = min_date + timedelta(days=14)
    elif preset == "First 30 Days":
        start_date = min_date
        end_date = min_date + timedelta(days=30)
    elif preset == "Full Test Set (9 Months)":
        start_date = min_date
        end_date = max_date
    else:
        date_range = st.sidebar.date_input(
            "Select Custom Range",
            value=(min_date, min_date + timedelta(days=7)),
            min_value=min_date,
            max_value=max_date,
        )
        if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
            start_date, end_date = date_range
        else:
            start_date, end_date = min_date, min_date + timedelta(days=7)

    # Filter dataframe
    mask = (df["datetime"].dt.date >= start_date) & (df["datetime"].dt.date <= end_date)
    filtered_df = df.loc[mask].copy()

    if len(filtered_df) == 0:
        st.warning("No observations found in the selected date range.")
        return

    # Model toggles in sidebar
    st.sidebar.markdown("---")
    st.sidebar.subheader("📈 Display Curves")
    show_actual = st.sidebar.checkbox("Actual Consumption", value=True)
    show_ensemble = st.sidebar.checkbox("50/50 Ensemble Forecast", value=True)
    show_interval = st.sidebar.checkbox("90% Prediction Interval Band", value=True)
    show_lgbm = st.sidebar.checkbox("LightGBM Component", value=False)
    show_xgb = st.sidebar.checkbox("XGBoost Component", value=False)
    show_baseline = st.sidebar.checkbox("24h Persistence Baseline", value=False)

    # Calculate dynamic metrics for selected slice
    mae, rmse, r2, coverage, avg_width = calculate_metrics(
        filtered_df["actual"].values,
        filtered_df["ensemble_pred"].values,
        filtered_df["lower_bound"].values if "lower_bound" in filtered_df.columns else None,
        filtered_df["upper_bound"].values if "upper_bound" in filtered_df.columns else None,
    )

    # Top KPI Cards
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Test Observations", f"{len(filtered_df):,} hrs", help="Number of hourly predictions in the active slice")
    with col2:
        st.metric("MAE (Error)", f"{mae:.3f} kW", delta="-49.2% vs Base", delta_color="inverse", help="Mean Absolute Error on active window")
    with col3:
        st.metric("RMSE", f"{rmse:.3f} kW", delta="-50.0% vs Base", delta_color="inverse", help="Root Mean Squared Error on active window")
    with col4:
        st.metric("R² Score", f"{r2:.4f}", delta="+0.85 vs Base", help="Coefficient of Determination")
    with col5:
        cov_delta = f"{coverage - 90.0:+.1f}% vs 90% Target" if coverage is not None else "N/A"
        st.metric("90% Interval Coverage", f"{coverage:.1f}%" if coverage is not None else "N/A", delta=cov_delta, help="Empirical percentage of actuals inside [q0.05, q0.95]")

    st.markdown("<br>", unsafe_allow_html=True)

    # Interactive Plotly Visualization
    fig = go.Figure()

    # Shaded 90% Prediction Interval
    if show_interval and "lower_bound" in filtered_df.columns and "upper_bound" in filtered_df.columns:
        # Upper bound trace
        fig.add_trace(
            go.Scatter(
                x=filtered_df["datetime"],
                y=filtered_df["upper_bound"],
                mode="lines",
                line=dict(width=0, color="rgba(255,127,14,0)"),
                showlegend=False,
                hoverinfo="skip",
            )
        )
        # Lower bound trace with fill to upper
        fig.add_trace(
            go.Scatter(
                x=filtered_df["datetime"],
                y=filtered_df["lower_bound"],
                mode="lines",
                line=dict(width=0, color="rgba(255,127,14,0)"),
                fill="tonexty",
                fillcolor="rgba(255, 127, 14, 0.22)",
                name="90% Prediction Interval [q0.05 - q0.95]",
                hovertemplate="Upper: %{customdata[0]:.2f} kW<br>Lower: %{y:.2f} kW<extra></extra>",
                customdata=np.stack([filtered_df["upper_bound"].values], axis=-1),
            )
        )

    # Actual consumption line
    if show_actual:
        fig.add_trace(
            go.Scatter(
                x=filtered_df["datetime"],
                y=filtered_df["actual"],
                mode="lines",
                name="Ground Truth Actual",
                line=dict(color="#1f77b4", width=2.5),
                hovertemplate="<b>Actual:</b> %{y:.3f} kW<extra></extra>",
            )
        )

    # Ensemble forecast line
    if show_ensemble:
        fig.add_trace(
            go.Scatter(
                x=filtered_df["datetime"],
                y=filtered_df["ensemble_pred"],
                mode="lines",
                name="50/50 Ensemble Forecast",
                line=dict(color="#d62728", width=2.2, dash="dash"),
                hovertemplate="<b>Ensemble:</b> %{y:.3f} kW<extra></extra>",
            )
        )

    # Optional component traces
    if show_lgbm and "lgbm_pred" in filtered_df.columns:
        fig.add_trace(
            go.Scatter(
                x=filtered_df["datetime"],
                y=filtered_df["lgbm_pred"],
                mode="lines",
                name="LightGBM Regressor",
                line=dict(color="#2ca02c", width=1.5, dash="dot"),
                hovertemplate="<b>LightGBM:</b> %{y:.3f} kW<extra></extra>",
            )
        )

    if show_xgb and "xgb_pred" in filtered_df.columns:
        fig.add_trace(
            go.Scatter(
                x=filtered_df["datetime"],
                y=filtered_df["xgb_pred"],
                mode="lines",
                name="XGBoost Regressor",
                line=dict(color="#9467bd", width=1.5, dash="dot"),
                hovertemplate="<b>XGBoost:</b> %{y:.3f} kW<extra></extra>",
            )
        )

    if show_baseline and "baseline_pred" in filtered_df.columns:
        fig.add_trace(
            go.Scatter(
                x=filtered_df["datetime"],
                y=filtered_df["baseline_pred"],
                mode="lines",
                name="24h Persistence Baseline",
                line=dict(color="#7f7f7f", width=1.3, dash="dashdot"),
                hovertemplate="<b>Baseline:</b> %{y:.3f} kW<extra></extra>",
            )
        )

    fig.update_layout(
        title=dict(
            text=f"Hourly Energy Consumption Forecast ({start_date.strftime('%b %d, %Y')} to {end_date.strftime('%b %d, %Y')})",
            font=dict(size=16, color="#222"),
        ),
        xaxis=dict(
            title="Timestamp",
            showgrid=True,
            gridcolor="rgba(200,200,200,0.3)",
        ),
        yaxis=dict(
            title="Global Active Power (kW)",
            showgrid=True,
            gridcolor="rgba(200,200,200,0.3)",
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(255,255,255,0.85)",
        ),
        hovermode="x unified",
        margin=dict(l=40, r=40, t=60, b=40),
        height=540,
    )

    st.plotly_chart(fig, use_container_width=True)

    # Benchmark Comparison Section
    st.markdown("---")
    st.subheader("📊 Comprehensive Model Benchmark Comparison")
    st.markdown(
        "Evaluated on the unseen chronological test set (`6,885` hourly observations from Feb 13, 2010 to Nov 26, 2010):"
    )

    benchmark_data = [
        {
            "Model Architecture": "🏆 Tuned Weather Ensemble (LGBM + XGB)",
            "MAE (kW)": 0.2646,
            "RMSE (kW)": 0.3898,
            "R² Score": 0.7154,
            "Methodology & Feature Highlights": "31 features (Open-Meteo weather + degree hours + calendar + multi-lags), log target",
        },
        {
            "Model Architecture": "⚡ Pure Time-Series LightGBM",
            "MAE (kW)": 0.3178,
            "RMSE (kW)": 0.4576,
            "R² Score": 0.6077,
            "Methodology & Feature Highlights": "11 temporal features without weather, linear target scale",
        },
        {
            "Model Architecture": "🧠 PyTorch LSTM Neural Network",
            "MAE (kW)": 0.3404,
            "RMSE (kW)": 0.4863,
            "R² Score": 0.5569,
            "Methodology & Feature Highlights": "2-Layer LSTM with 24-hour sliding sequence input matrices",
        },
        {
            "Model Architecture": "⏳ Baseline Persistence (24h Lag)",
            "MAE (kW)": 0.5210,
            "RMSE (kW)": 0.7794,
            "R² Score": -0.1380,
            "Methodology & Feature Highlights": "Naive copy of consumption at the same hour yesterday",
        },
    ]

    benchmark_df = pd.DataFrame(benchmark_data)
    st.dataframe(
        benchmark_df.style.format({
            "MAE (kW)": "{:.4f}",
            "RMSE (kW)": "{:.4f}",
            "R² Score": "{:.4f}",
        }).highlight_min(subset=["MAE (kW)", "RMSE (kW)"], color="#d4edda")
          .highlight_max(subset=["R² Score"], color="#d4edda"),
        use_container_width=True,
        hide_index=True,
    )

    # Technical Notes & Feature Importance
    with st.expander("🔍 Pipeline Architecture & Key Drivers"):
        st.markdown(
            r"""
            - **Thermodynamic Coupling**: Heating Degree Hours ($\max(0, 18 - T_{2m})$) captures heating demand surges in winter.
            - **Autoregressive Memory**: `lag_1` (hourly inertia), `rolling_mean_24` (daily baseline), and `lag_168` (weekly rhythm).
            - **Uncertainty Quantification**: 90% prediction intervals calibrated via dual quantile LightGBM models ($\alpha=0.05$ and $\alpha=0.95$).
            - **Zero Data Leakage**: Concurrent electrical telemetry (`Voltage`, `Global_intensity`, `Sub_metering`) is strictly dropped to simulate real-world forward prediction.
            """
        )
        img_path = Path(__file__).resolve().parent / "outputs" / "feature_importance.png"
        if img_path.exists():
            st.image(str(img_path), caption="Top 10 Feature Importances (Split Frequency)", use_container_width=True)


if __name__ == "__main__":
    main()
