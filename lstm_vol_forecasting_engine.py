import streamlit as st
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from io import StringIO
import copy
from pathlib import Path

# ==============================
# Utility Functions
# ==============================
def generate_gbm(n_days=1000, mu=0.05, sigma=0.2, s0=100, seed=42):
    np.random.seed(seed)
    dt = 1/252
    returns = np.random.normal((mu - 0.5 * sigma ** 2) * dt, sigma * np.sqrt(dt), n_days)
    price = s0 * np.exp(np.cumsum(returns))
    dates = pd.date_range(end=pd.Timestamp.today(), periods=n_days)
    return pd.DataFrame({'date': dates, 'close': price})

def realized_volatility(returns, window):
    return returns.rolling(window).std() * np.sqrt(252)

def create_sequences(data, seq_length):
    """Create sequences from data. Handles both 1D and 2D arrays.
    For 2D input (samples, features): X is (num_seq, seq_length, features), y is (num_seq,) using first column as target.
    For 1D input: X is (num_seq, seq_length), y is (num_seq,).
    """
    xs, ys = [], []
    if data.ndim == 1:
        for i in range(len(data) - seq_length):
            xs.append(data[i:i+seq_length])
            ys.append(data[i+seq_length])
    else:
        for i in range(len(data) - seq_length):
            xs.append(data[i:i+seq_length, :])  # all features for the window
            ys.append(data[i+seq_length, 0])     # target is first column (realized_vol)
    return np.array(xs), np.array(ys)

# ==============================
# PyTorch LSTM Model
# ==============================
class LSTMVolModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.0):
        super().__init__()
        # dropout between LSTM layers (only active when num_layers > 1)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.dropout(out)
        return self.fc(out)

# ==============================
# Walk-Forward Cross-Validation
# ==============================
def walk_forward_cv(features_scaled, target_scaler, seq_length, hidden_size,
                    num_layers, epochs, lr, device, n_folds=5):
    """Walk-forward CV with expanding training window and 5 temporal folds.
    Returns list of per-fold RMSE values (in original scale).
    """
    n = len(features_scaled)
    fold_size = (n - seq_length) // n_folds
    rmse_scores = []

    for fold in range(n_folds):
        # Expanding training window: start from 0 up to fold boundary
        train_end = seq_length + fold_size * (fold + 1)
        val_end = min(train_end + fold_size, n)
        if train_end >= n or val_end <= train_end + seq_length:
            break

        train_data = features_scaled[:train_end]
        val_data = features_scaled[train_end - seq_length:val_end]  # overlap for sequence context

        X_tr, y_tr = create_sequences(train_data, seq_length)
        X_vl, y_vl = create_sequences(val_data, seq_length)

        if len(X_tr) == 0 or len(X_vl) == 0:
            continue

        y_tr = y_tr[..., np.newaxis]
        y_vl = y_vl[..., np.newaxis]

        X_tr_t = torch.tensor(X_tr, dtype=torch.float32).to(device)
        y_tr_t = torch.tensor(y_tr, dtype=torch.float32).to(device)
        X_vl_t = torch.tensor(X_vl, dtype=torch.float32).to(device)
        y_vl_t = torch.tensor(y_vl, dtype=torch.float32).to(device)

        model = LSTMVolModel(input_size=3, hidden_size=hidden_size,
                             num_layers=num_layers, dropout=0.2).to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)

        best_val_loss = float('inf')
        best_state = None
        patience_counter = 0

        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            output = model(X_tr_t)
            loss = criterion(output, y_tr_t)
            loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            model.eval()
            with torch.no_grad():
                vp = model(X_vl_t)
                vl = criterion(vp, y_vl_t).item()
            if vl < best_val_loss:
                best_val_loss = vl
                best_state = copy.deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= 20:
                    break

        if best_state is not None:
            model.load_state_dict(best_state)

        model.eval()
        with torch.no_grad():
            preds_scaled = model(X_vl_t).cpu().numpy().flatten()
            actuals_scaled = y_vl.flatten()

        # Inverse-transform (target is first column of feature scaler)
        dummy_pred = np.zeros((len(preds_scaled), 3))
        dummy_pred[:, 0] = preds_scaled
        preds_orig = target_scaler.inverse_transform(dummy_pred)[:, 0]

        dummy_act = np.zeros((len(actuals_scaled), 3))
        dummy_act[:, 0] = actuals_scaled
        actuals_orig = target_scaler.inverse_transform(dummy_act)[:, 0]

        fold_rmse = np.sqrt(mean_squared_error(actuals_orig, preds_orig))
        rmse_scores.append(fold_rmse)

    return rmse_scores

# ==============================
# Streamlit App
# ==============================
st.set_page_config(page_title="LSTM Volatility Forecasting Engine", layout="wide")
st.title("LSTM Volatility Forecasting Engine")

with st.sidebar:
    st.header("Configuration")
    seq_length = st.slider("Sequence Length", 5, 60, 20)
    hidden_size = st.slider("Hidden Units", 8, 128, 32)
    num_layers = st.slider("LSTM Layers", 1, 3, 1)
    epochs = st.slider("Epochs", 10, 200, 50)
    lr = st.number_input("Learning Rate", 0.0001, 0.1, 0.001, format="%f")
    st.markdown("---")
    data_source = st.radio("Data Source", ["Upload CSV", "Synthetic GBM"], horizontal=True)
    st.markdown("---")
    run_wfcv = st.button("Run Walk-Forward Validation")

# ==============================
# Data Loading
# ==============================
if data_source == "Upload CSV":
    uploaded = st.file_uploader("Upload CSV (date, close)", type=["csv"])
    if uploaded:
        df = pd.read_csv(uploaded)
        if not {'date', 'close'}.issubset(df.columns):
            st.error("CSV must have 'date' and 'close' columns.")
            st.stop()
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
    else:
        st.info("Please upload a CSV file.")
        st.stop()
else:
    n_days = st.slider("Synthetic GBM Days", 500, 3000, 1000)
    df = generate_gbm(n_days=n_days)

# ==============================
# Feature Engineering (1C: 3 features)
# ==============================
df['log_return'] = np.log(df['close']).diff()
df['realized_vol'] = realized_volatility(df['log_return'], window=seq_length)
df['momentum_5'] = df['log_return'].rolling(5).mean()
df = df.dropna().reset_index(drop=True)

# Build multi-feature array: [realized_vol, log_return, momentum_5]
feature_cols = ['realized_vol', 'log_return', 'momentum_5']
features_raw = df[feature_cols].values.astype(np.float32)  # shape: (N, 3)

# ==============================
# 1A/1C: MinMaxScaler normalization BEFORE sequence creation
# ==============================
# We fit the scaler on training portion only (first 80%)
split_raw = int(0.8 * len(features_raw))
feature_scaler = MinMaxScaler(feature_range=(0, 1))
feature_scaler.fit(features_raw[:split_raw])
features_scaled = feature_scaler.transform(features_raw)

# Create sequences from scaled multi-feature data
X, y = create_sequences(features_scaled, seq_length)
# X shape: (samples, seq_length, 3), y shape: (samples,) — target is realized_vol (col 0)
y = y[..., np.newaxis]  # shape: (samples, 1)

# Train/val split
split = int(0.8 * len(X))
X_train, X_val = X[:split], X[split:]
y_train, y_val = y[:split], y[split:]

# Torch tensors
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train_t = torch.tensor(y_train, dtype=torch.float32).to(device)
X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
y_val_t = torch.tensor(y_val, dtype=torch.float32).to(device)

# ==============================
# Model Training (1B: early stopping, 1D: dropout + grad clipping)
# ==============================
model = LSTMVolModel(input_size=3, hidden_size=hidden_size,
                     num_layers=num_layers, dropout=0.2).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=lr)
train_losses, val_losses = [], []

best_val_loss = float('inf')
best_model_state = None
patience_counter = 0

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    output = model(X_train_t)
    loss = criterion(output, y_train_t)
    loss.backward()
    clip_grad_norm_(model.parameters(), max_norm=1.0)  # 1D: gradient clipping
    optimizer.step()
    train_losses.append(loss.item())

    # Validation
    model.eval()
    with torch.no_grad():
        val_pred = model(X_val_t)
        val_loss = criterion(val_pred, y_val_t)
        val_losses.append(val_loss.item())

    # 1B: Early stopping with patience=20
    if val_loss.item() < best_val_loss:
        best_val_loss = val_loss.item()
        best_model_state = copy.deepcopy(model.state_dict())
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= 20:
            break

# Load best model state
if best_model_state is not None:
    model.load_state_dict(best_model_state)

# Save best model to models/lstm_vol_best.pt
models_dir = Path(__file__).resolve().parent / "models"
models_dir.mkdir(exist_ok=True)
torch.save(model.state_dict(), models_dir / "lstm_vol_best.pt")

# ==============================
# Evaluation (1A: inverse-transform before metrics)
# ==============================
model.eval()
with torch.no_grad():
    y_pred_scaled = model(X_val_t).cpu().numpy().flatten()
    y_true_scaled = y_val.flatten()

# Inverse-transform predictions and actuals back to original scale
# realized_vol is column 0 of the feature scaler, so we build dummy arrays
def inverse_transform_col0(values, scaler, n_features=3):
    """Inverse-transform values that correspond to column 0 of the scaler."""
    dummy = np.zeros((len(values), n_features))
    dummy[:, 0] = values
    return scaler.inverse_transform(dummy)[:, 0]

y_pred = inverse_transform_col0(y_pred_scaled, feature_scaler)
y_true = inverse_transform_col0(y_true_scaled, feature_scaler)

rmse = np.sqrt(mean_squared_error(y_true, y_pred))
r2 = r2_score(y_true, y_pred)
errors = y_pred - y_true

# ==============================
# Plots
# ==============================
def plot_forecast(y_true, y_pred):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(y_true, label='Actual', color='black')
    ax.plot(y_pred, label='Predicted', color='royalblue', alpha=0.7)
    ax.set_title('Predicted vs Actual Volatility')
    ax.set_xlabel('Time')
    ax.set_ylabel('Volatility')
    ax.legend()
    st.pyplot(fig)

def plot_loss(train_losses, val_losses):
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(train_losses, label='Train Loss')
    ax.plot(val_losses, label='Val Loss')
    ax.set_title('Training Loss Curve')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE Loss')
    ax.legend()
    st.pyplot(fig)

def plot_error_dist(errors):
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.hist(errors, bins=30, color='tomato', alpha=0.7)
    ax.set_title('Forecast Error Distribution')
    ax.set_xlabel('Error')
    st.pyplot(fig)

# ==============================
# Streamlit Layout
# ==============================
col1, col2 = st.columns([2, 1])
with col1:
    st.subheader("Forecast Plot")
    plot_forecast(y_true, y_pred)
    st.subheader("Training Loss Curve")
    plot_loss(train_losses, val_losses)
with col2:
    st.subheader("Metrics")
    st.metric("RMSE", f"{rmse:.4f}")
    st.metric("R-squared", f"{r2:.4f}")
    st.subheader("Error Distribution")
    plot_error_dist(errors)

# ==============================
# 1E: Walk-Forward Cross-Validation
# ==============================
if run_wfcv:
    with st.spinner("Running walk-forward cross-validation (5 folds)..."):
        rmse_scores = walk_forward_cv(
            features_scaled=features_scaled,
            target_scaler=feature_scaler,
            seq_length=seq_length,
            hidden_size=hidden_size,
            num_layers=num_layers,
            epochs=epochs,
            lr=lr,
            device=device,
            n_folds=5
        )
    if rmse_scores:
        st.subheader("Walk-Forward Cross-Validation Results")
        for i, score in enumerate(rmse_scores):
            st.write(f"Fold {i+1} RMSE: {score:.4f}")
        st.metric("Mean RMSE", f"{np.mean(rmse_scores):.4f}")
        st.metric("Std RMSE", f"{np.std(rmse_scores):.4f}")
    else:
        st.warning("Not enough data to run walk-forward validation with current settings.")

st.caption("© 2026 LSTM Volatility Forecasting Engine. Powered by PyTorch & Streamlit.")
