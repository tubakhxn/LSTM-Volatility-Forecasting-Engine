"""
test_lstm.py -- Unit tests for the LSTM Volatility Forecasting Engine.

Covers:
    1. Training on synthetic GBM data with MinMaxScaler achieves RMSE < 0.05
    2. Model can save to disk and reload with identical predictions
    3. Walk-forward validation returns reasonable results
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# Ensure sibling repo is importable
_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT / "LSTM-Volatility-Forecasting-Engine"))

from lstm_vol_forecasting_engine import (
    LSTMVolModel,
    create_sequences,
    generate_gbm,
    realized_volatility,
)


# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------
SEQ_LENGTH = 20
HIDDEN_SIZE = 32
NUM_LAYERS = 1
EPOCHS = 80


def _prepare_data(n_days=800, seq_length=SEQ_LENGTH):
    """Generate GBM data and return scaled sequences + scaler."""
    df = generate_gbm(n_days=n_days)
    df["log_return"] = np.log(df["close"]).diff()
    df["realized_vol"] = realized_volatility(df["log_return"], window=seq_length)
    df = df.dropna().reset_index(drop=True)

    vol = df["realized_vol"].values.astype(np.float32)
    scaler = MinMaxScaler()
    vol_scaled = scaler.fit_transform(vol.reshape(-1, 1)).flatten()

    X, y = create_sequences(vol_scaled, seq_length)
    X = X[..., np.newaxis]
    y = y[..., np.newaxis]
    return X, y, scaler


def _train_model(X_train, y_train, epochs=EPOCHS, lr=0.001):
    """Train and return an LSTMVolModel."""
    device = torch.device("cpu")
    model = LSTMVolModel(input_size=1, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    X_t = torch.tensor(X_train, dtype=torch.float32)
    y_t = torch.tensor(y_train, dtype=torch.float32)

    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        loss = criterion(model(X_t), y_t)
        loss.backward()
        optimizer.step()

    return model


# ---------------------------------------------------------------
# Tests
# ---------------------------------------------------------------
class TestLSTMTraining:
    """LSTM trains on synthetic GBM data with scaling and achieves RMSE < 0.05."""

    def test_rmse_below_threshold(self):
        X, y, scaler = _prepare_data()
        split = int(0.8 * len(X))
        X_train, X_val = X[:split], X[split:]
        y_train, y_val = y[:split], y[split:]

        model = _train_model(X_train, y_train, epochs=EPOCHS)
        model.eval()
        with torch.no_grad():
            preds = model(torch.tensor(X_val, dtype=torch.float32)).numpy().flatten()

        rmse = float(np.sqrt(mean_squared_error(y_val.flatten(), preds)))
        assert rmse < 0.05, f"RMSE {rmse:.4f} exceeds 0.05 threshold"


class TestLSTMPersistence:
    """Model can save and load from disk with identical predictions."""

    def test_save_load_roundtrip(self):
        X, y, _ = _prepare_data(n_days=500)
        model = _train_model(X, y, epochs=30)
        model.eval()

        sample = torch.tensor(X[:5], dtype=torch.float32)
        with torch.no_grad():
            pred_before = model(sample).numpy()

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test_model.pt"
            torch.save(model.state_dict(), path)

            loaded = LSTMVolModel(input_size=1, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS)
            loaded.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
            loaded.eval()
            with torch.no_grad():
                pred_after = loaded(sample).numpy()

        np.testing.assert_allclose(pred_before, pred_after, atol=1e-6)


class TestWalkForwardValidation:
    """Walk-forward validation returns reasonable results."""

    def test_walk_forward(self):
        X, y, _ = _prepare_data(n_days=600)
        n_folds = 3
        fold_size = len(X) // (n_folds + 1)
        rmses = []

        for fold in range(n_folds):
            train_end = fold_size * (fold + 1)
            val_end = train_end + fold_size
            if val_end > len(X):
                break
            X_train, y_train = X[:train_end], y[:train_end]
            X_val, y_val = X[train_end:val_end], y[train_end:val_end]

            model = _train_model(X_train, y_train, epochs=50)
            model.eval()
            with torch.no_grad():
                preds = model(torch.tensor(X_val, dtype=torch.float32)).numpy().flatten()

            rmse = float(np.sqrt(mean_squared_error(y_val.flatten(), preds)))
            rmses.append(rmse)

        assert len(rmses) > 0, "No walk-forward folds produced"
        avg_rmse = np.mean(rmses)
        # Lenient threshold -- walk-forward is harder than a single split
        assert avg_rmse < 0.10, f"Average walk-forward RMSE {avg_rmse:.4f} too high"
