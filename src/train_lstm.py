"""
train_lstm.py
Trains and evaluates a PyTorch Deep Learning LSTM Regressor on the energy dataset:
1. Loads engineered_energy.csv and selects all 31 non-leaking predictors.
2. Performs 80/20 chronological split and MinMax scaling on features and target.
3. Constructs 24-hour sliding sequence windows for multi-scale sequential input.
4. Trains a 2-layer LSTMRegressor (hidden_size=64, dropout=0.2, Linear(64,32) -> ReLU -> Linear(32,1)).
5. Inverts predictions back to original kW scale and benchmarks against Baseline, Pure LightGBM, and Weather Ensemble.
"""

import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
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

SEQUENCE_LENGTH = 24  # 24-hour sliding window


def get_project_root() -> Path:
    """Returns the project root directory."""
    return Path(__file__).resolve().parent.parent


def calculate_metrics(y_true, y_pred):
    """Calculates MAE, RMSE, and R2 score."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return mae, rmse, r2


class LSTMRegressor(nn.Module):
    """2-Layer LSTM with Dense Projection Head."""
    def __init__(self, input_size: int = 31, hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len, input_size)
        lstm_out, _ = self.lstm(x)
        # Take the hidden state of the final time step
        last_step = lstm_out[:, -1, :]
        out = self.fc(last_step)
        return out


def create_train_sequences(X: np.ndarray, y: np.ndarray, seq_len: int = 24):
    """Creates (seq_len, num_features) sliding windows for training."""
    X_seq, y_seq = [], []
    for i in range(seq_len, len(X)):
        X_seq.append(X[i - seq_len : i])
        y_seq.append(y[i])
    return np.array(X_seq, dtype=np.float32), np.array(y_seq, dtype=np.float32)


def create_test_sequences(X_train: np.ndarray, X_test: np.ndarray, seq_len: int = 24):
    """Prepends the last seq_len train steps so every test timestamp has a full 24h window."""
    X_padded = np.vstack([X_train[-seq_len:], X_test])
    X_seq = []
    for i in range(len(X_test)):
        X_seq.append(X_padded[i : i + seq_len])
    return np.array(X_seq, dtype=np.float32)


def main():
    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using compute device: {device}")

    project_root = get_project_root()
    data_path = project_root / "data" / "processed" / "engineered_energy.csv"

    if not data_path.exists():
        print(f"ERROR: Dataset not found at {data_path}")
        sys.exit(1)

    print(f"1. Loading engineered dataset from: {data_path}")
    df = pd.read_csv(
        data_path,
        parse_dates=["datetime"],
        index_col="datetime",
    )
    print(f"   Loaded shape: {df.shape}")

    # Chronological Train-Test Split (80% / 20%)
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    target_col = "Global_active_power"
    feature_cols = [
        col for col in df.columns
        if col != target_col and col not in DROPPED_TELEMETRY
    ]
    num_features = len(feature_cols)

    print(f"\n2. Data Preparation & Scaling:")
    print(f"   Features ({num_features}): {', '.join(feature_cols[:6])}... (+{num_features - 6} more)")
    print(f"   Train samples: {len(train_df):,} | Test samples: {len(test_df):,}")

    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    X_train_raw = train_df[feature_cols].values
    y_train_raw = train_df[[target_col]].values
    X_test_raw = test_df[feature_cols].values
    y_test_raw = test_df[[target_col]].values

    X_train_scaled = scaler_X.fit_transform(X_train_raw)
    y_train_scaled = scaler_y.fit_transform(y_train_raw)
    X_test_scaled = scaler_X.transform(X_test_raw)
    y_test_scaled = scaler_y.transform(y_test_raw)

    # 3. Create 24-hour Sliding Window Sequences
    print(f"   Creating {SEQUENCE_LENGTH}-hour sliding sequence windows...")
    X_train_seq, y_train_seq = create_train_sequences(X_train_scaled, y_train_scaled, SEQUENCE_LENGTH)
    X_test_seq = create_test_sequences(X_train_scaled, X_test_scaled, SEQUENCE_LENGTH)

    print(f"   X_train_seq shape: {X_train_seq.shape} | y_train_seq shape: {y_train_seq.shape}")
    print(f"   X_test_seq shape:  {X_test_seq.shape}  | Test rows: {len(test_df):,}")

    train_dataset = TensorDataset(torch.from_numpy(X_train_seq), torch.from_numpy(y_train_seq))
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    # 4. Instantiate PyTorch LSTM Architecture
    print("\n3. PyTorch LSTM Architecture:")
    print("   - Layer 1: LSTM(input_size=31, hidden_size=64, num_layers=2, batch_first=True, dropout=0.2)")
    print("   - Layer 2: Linear(64, 32) -> ReLU() -> Linear(32, 1)")
    print("   - Loss: MSELoss | Optimizer: Adam(lr=0.001) | Batch Size: 64 | Epochs: 15")

    model = LSTMRegressor(
        input_size=num_features,
        hidden_size=64,
        num_layers=2,
        dropout=0.2,
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training Loop
    print("\n4. Training LSTM Neural Network:")
    start_time = time.time()
    for epoch in range(1, 16):
        model.train()
        epoch_loss = 0.0
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            preds = model(batch_X)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * len(batch_X)

        avg_loss = epoch_loss / len(train_dataset)
        if epoch % 3 == 0 or epoch == 1 or epoch == 15:
            print(f"   Epoch [{epoch:2d}/15] - Loss: {avg_loss:.6f}")

    elapsed = time.time() - start_time
    print(f"   Training completed in {elapsed:.1f}s.")

    # 5. Evaluation & Inversion
    print("\n5. Generating Test Predictions & Inverting Scale...")
    model.eval()
    with torch.no_grad():
        test_tensor = torch.from_numpy(X_test_seq).to(device)
        y_pred_scaled = model(test_tensor).cpu().numpy()

    # Inverse transform predictions back to original kW scale
    y_pred_kw = scaler_y.inverse_transform(y_pred_scaled).ravel()
    y_test_kw = test_df[target_col].values

    lstm_mae, lstm_rmse, lstm_r2 = calculate_metrics(y_test_kw, y_pred_kw)

    # Benchmark comparison values
    baseline_mae, baseline_rmse, baseline_r2 = 0.5210, 0.7794, -0.1380
    pure_lgbm_mae, pure_lgbm_rmse, pure_lgbm_r2 = 0.3178, 0.4576, 0.6077
    ens_mae, ens_rmse, ens_r2 = 0.2646, 0.3898, 0.7154

    print("\n" + "=" * 76)
    print(f"{'COMPREHENSIVE MODEL BENCHMARK COMPARISON (TEST SET)':^76}")
    print("=" * 76)
    print(f"{'Model Architecture':<36} | {'MAE (kW)':>10} | {'RMSE (kW)':>10} | {'R² Score':>10}")
    print("-" * 76)
    print(f"{'1. Baseline Persistence (24h)':<36} | {baseline_mae:>10.4f} | {baseline_rmse:>10.4f} | {baseline_r2:>10.4f}")
    print(f"{'2. Pure Time-Series LightGBM (11 fts)':<36} | {pure_lgbm_mae:>10.4f} | {pure_lgbm_rmse:>10.4f} | {pure_lgbm_r2:>10.4f}")
    print(f"{'3. Tuned Weather Ensemble (LGBM+XGB)':<36} | {ens_mae:>10.4f} | {ens_rmse:>10.4f} | {ens_r2:>10.4f}")
    print(f"{'4. PyTorch LSTM Neural Network (2-L)':<36} | {lstm_mae:>10.4f} | {lstm_rmse:>10.4f} | {lstm_r2:>10.4f}")
    print("-" * 76)
    lstm_imp_base = ((baseline_mae - lstm_mae) / baseline_mae) * 100
    lstm_imp_pure = ((pure_lgbm_mae - lstm_mae) / pure_lgbm_mae) * 100
    print(f"{'LSTM Improvement vs. Baseline':<36} | {f'{lstm_imp_base:+.1f}%':>10} | {'--':>10} | {f'{(lstm_r2 - baseline_r2):+.4f}':>10}")
    print(f"{'LSTM Improvement vs. Pure LightGBM':<36} | {f'{lstm_imp_pure:+.1f}%':>10} | {'--':>10} | {f'{(lstm_r2 - pure_lgbm_r2):+.4f}':>10}")
    print("=" * 76)


if __name__ == "__main__":
    main()
