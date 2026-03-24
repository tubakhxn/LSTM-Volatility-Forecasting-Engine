"""
auto_refine.py -- Phase 3: Self-diagnosing refinement loop.

Fetches live SPY data, diagnoses LSTM / HMM / DQN health, and
optionally triggers retraining when drift thresholds are exceeded.

Usage:
    python auto_refine.py              # diagnose + retrain if needed
    python auto_refine.py --dry-run    # diagnose only, never retrain
"""

import sys
import os
import json
import argparse
import datetime
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import joblib
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# ---------------------------------------------------------------------------
# sys.path manipulation so we can import from the three sibling repos
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parent.parent          # NBA_Syndicate_Lab/
_LSTM_DIR = _ROOT / "LSTM-Volatility-Forecasting-Engine"
_HMM_DIR = _ROOT / "Market-Regime-Detection-Engine"
_RL_DIR = _ROOT / "Reinforcement-Learning-Trading-Agent"

for _p in (_LSTM_DIR, _HMM_DIR, _RL_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# Imports from sibling repos ------------------------------------------------
from lstm_vol_forecasting_engine import (
    LSTMVolModel,
    create_sequences,
    realized_volatility,
    generate_gbm as lstm_generate_gbm,
)
from model_registry import (
    save_lstm, load_lstm,
    save_hmm, load_hmm,
    save_dqn, load_dqn,
    MODEL_DIR,
)
from market_regime_detection_engine import compute_features, fit_hmm
from rl_trading_agent import (
    TradingEnv,
    DQNAgent,
    train_agent,
    sharpe_ratio,
    generate_synthetic_prices,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
LOG_DIR = _LSTM_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / "refine_log.jsonl"

LSTM_RMSE_THRESHOLD = 0.030
HMM_DRIFT_THRESHOLD = 0.15
DQN_SHARPE_THRESHOLD = 0.5

SEQ_LENGTH = 20
HIDDEN_SIZE = 32
NUM_LAYERS = 1
WINDOW_SIZE = 30

logger = logging.getLogger("auto_refine")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _log_diagnostic(record: dict) -> None:
    """Append a JSON record to the refine log."""
    record["timestamp"] = datetime.datetime.utcnow().isoformat() + "Z"
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(record) + "\n")
    logger.info(json.dumps(record, indent=2))


def fetch_spy_data(days: int = 252) -> pd.DataFrame:
    """Download ~`days` calendar days of SPY daily close prices."""
    end = datetime.date.today()
    # Fetch extra calendar days to ensure enough trading days
    start = end - datetime.timedelta(days=int(days * 1.6))
    df = yf.download("SPY", start=str(start), end=str(end), progress=False)
    df = df.reset_index()
    # Normalise column names (yfinance may return MultiIndex)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    df = df.rename(columns={"Date": "date", "Close": "close"})
    df = df[["date", "close"]].dropna().tail(days).reset_index(drop=True)
    df["close"] = df["close"].astype(float)
    return df


# ---------------------------------------------------------------------------
# Diagnosis: LSTM
# ---------------------------------------------------------------------------
def diagnose_lstm(df: pd.DataFrame) -> dict:
    """Evaluate saved LSTM on live vol data; return diagnostics dict."""
    result = {"model": "LSTM", "needs_retrain": False, "rmse": None, "error": None}

    try:
        # Feature engineering
        data = df.copy()
        data["log_return"] = np.log(data["close"]).diff()
        data["realized_vol"] = realized_volatility(data["log_return"], window=SEQ_LENGTH)
        data = data.dropna().reset_index(drop=True)
        vol = data["realized_vol"].values.astype(np.float32)

        # Scale
        scaler = MinMaxScaler()
        vol_scaled = scaler.fit_transform(vol.reshape(-1, 1)).flatten()

        X, y = create_sequences(vol_scaled, SEQ_LENGTH)
        X = X[..., np.newaxis]

        # Last 20 %
        split = int(0.8 * len(X))
        X_val, y_val = X[split:], y[split:]

        # Load model
        device = torch.device("cpu")
        model = LSTMVolModel(input_size=1, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS).to(device)
        model = load_lstm(model)
        model.eval()

        X_t = torch.tensor(X_val, dtype=torch.float32).to(device)
        with torch.no_grad():
            preds = model(X_t).cpu().numpy().flatten()

        rmse = float(np.sqrt(mean_squared_error(y_val, preds)))
        result["rmse"] = rmse
        if rmse > LSTM_RMSE_THRESHOLD:
            result["needs_retrain"] = True
    except Exception as exc:
        result["error"] = str(exc)
        result["needs_retrain"] = True

    return result


# ---------------------------------------------------------------------------
# Diagnosis: HMM
# ---------------------------------------------------------------------------
def diagnose_hmm(df: pd.DataFrame) -> dict:
    """Refit HMM on live data, compare transition matrix to saved baseline."""
    result = {"model": "HMM", "needs_refit": False, "mean_abs_drift": None, "error": None}

    try:
        _, features = compute_features(df, window=SEQ_LENGTH)
        _, _, transmat_new, _, _ = fit_hmm(features, n_states=3)

        saved_model = load_hmm()
        if saved_model is None:
            result["needs_refit"] = True
            result["error"] = "No saved HMM baseline found."
            return result

        transmat_saved = saved_model.transmat_
        drift = float(np.mean(np.abs(transmat_new - transmat_saved)))
        result["mean_abs_drift"] = drift
        if drift > HMM_DRIFT_THRESHOLD:
            result["needs_refit"] = True
    except Exception as exc:
        result["error"] = str(exc)
        result["needs_refit"] = True

    return result


# ---------------------------------------------------------------------------
# Diagnosis: DQN
# ---------------------------------------------------------------------------
def diagnose_dqn(df: pd.DataFrame, n_episodes: int = 5) -> dict:
    """Run inference episodes on recent price data; check Sharpe ratio."""
    result = {"model": "DQN", "needs_retrain": False, "sharpe": None, "error": None}

    try:
        prices = df["close"].values.astype(float)
        env = TradingEnv(prices, window_size=WINDOW_SIZE, transaction_cost=0.001)
        agent = DQNAgent(state_dim=WINDOW_SIZE, action_dim=3, lr=1e-3)
        agent = load_dqn(agent)
        # Greedy inference
        agent.epsilon = 0.0

        all_returns = []
        for _ in range(n_episodes):
            state = env.reset()
            done = False
            while not done:
                action = agent.select_action(state)
                state, reward, done, _ = env.step(action)
            all_returns.extend(env.returns_history)

        sr = float(sharpe_ratio(all_returns))
        result["sharpe"] = sr
        if sr < DQN_SHARPE_THRESHOLD:
            result["needs_retrain"] = True
    except Exception as exc:
        result["error"] = str(exc)
        result["needs_retrain"] = True

    return result


# ---------------------------------------------------------------------------
# Retraining helpers
# ---------------------------------------------------------------------------
def retrain_lstm(df: pd.DataFrame, epochs: int = 100, lr: float = 0.001) -> None:
    """Retrain LSTM on live data and save the best model."""
    logger.info("Retraining LSTM ...")
    data = df.copy()
    data["log_return"] = np.log(data["close"]).diff()
    data["realized_vol"] = realized_volatility(data["log_return"], window=SEQ_LENGTH)
    data = data.dropna().reset_index(drop=True)
    vol = data["realized_vol"].values.astype(np.float32)

    scaler = MinMaxScaler()
    vol_scaled = scaler.fit_transform(vol.reshape(-1, 1)).flatten()

    X, y = create_sequences(vol_scaled, SEQ_LENGTH)
    X = X[..., np.newaxis]
    y = y[..., np.newaxis]

    split = int(0.8 * len(X))
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    device = torch.device("cpu")
    model = LSTMVolModel(input_size=1, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    X_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_t = torch.tensor(y_train, dtype=torch.float32).to(device)

    best_val_loss = float("inf")
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        loss = criterion(model(X_t), y_t)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = criterion(
                model(torch.tensor(X_val, dtype=torch.float32).to(device)),
                torch.tensor(y_val, dtype=torch.float32).to(device),
            ).item()
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_lstm(model.state_dict())

    logger.info(f"LSTM retrained. Best val MSE={best_val_loss:.6f}")


def retrain_hmm(df: pd.DataFrame) -> None:
    """Refit HMM on live data and overwrite the saved baseline."""
    logger.info("Refitting HMM ...")
    from hmmlearn.hmm import GaussianHMM

    _, features = compute_features(df, window=SEQ_LENGTH)
    hmm = GaussianHMM(n_components=3, covariance_type="full", n_iter=1000, random_state=42)
    hmm.fit(features)
    save_hmm(hmm)
    logger.info("HMM baseline saved.")


def retrain_dqn(df: pd.DataFrame, episodes: int = 50) -> None:
    """Retrain DQN on recent price data and save policy weights."""
    logger.info("Retraining DQN ...")
    prices = df["close"].values.astype(float)
    env = TradingEnv(prices, window_size=WINDOW_SIZE, transaction_cost=0.001)
    agent = DQNAgent(state_dim=WINDOW_SIZE, action_dim=3, lr=1e-3)
    train_agent(env, agent, episodes=episodes, verbose=False)
    save_dqn(agent.policy_net.state_dict())
    logger.info("DQN policy saved.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Auto-refinement loop for quant pipeline.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Diagnose only; do not retrain any model.",
    )
    args = parser.parse_args()

    logger.info("Fetching live SPY data ...")
    df = fetch_spy_data(days=252)
    logger.info(f"Fetched {len(df)} rows of SPY data.")

    # --- LSTM ---
    lstm_diag = diagnose_lstm(df)
    _log_diagnostic(lstm_diag)

    # --- HMM ---
    hmm_diag = diagnose_hmm(df)
    _log_diagnostic(hmm_diag)

    # --- DQN ---
    dqn_diag = diagnose_dqn(df)
    _log_diagnostic(dqn_diag)

    # --- Summary ---
    summary = {
        "phase": "auto_refine",
        "dry_run": args.dry_run,
        "lstm_needs_retrain": lstm_diag.get("needs_retrain", False),
        "hmm_needs_refit": hmm_diag.get("needs_refit", False),
        "dqn_needs_retrain": dqn_diag.get("needs_retrain", False),
    }
    _log_diagnostic(summary)

    if args.dry_run:
        logger.info("--dry-run set. Skipping retraining.")
        return

    # Retrain as needed
    if lstm_diag.get("needs_retrain"):
        retrain_lstm(df)
        _log_diagnostic({"action": "retrained_lstm"})

    if hmm_diag.get("needs_refit"):
        retrain_hmm(df)
        _log_diagnostic({"action": "refitted_hmm"})

    if dqn_diag.get("needs_retrain"):
        retrain_dqn(df)
        _log_diagnostic({"action": "retrained_dqn"})

    logger.info("Auto-refinement complete.")


if __name__ == "__main__":
    main()
