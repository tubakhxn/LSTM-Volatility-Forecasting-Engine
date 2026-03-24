#!/usr/bin/env python3
"""
Master Pipeline Orchestrator — Phase 2A/2E
Connects LSTM Volatility Forecasting, HMM Regime Detection, and DQN Trading Agent
into a single end-to-end quant signal pipeline.
"""

import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

# ---------------------------------------------------------------------------
# Resolve sibling repo paths so we can import from each engine
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent  # /Users/Sharman/NBA_Syndicate_Lab
LSTM_DIR = BASE_DIR / "LSTM-Volatility-Forecasting-Engine"
HMM_DIR  = BASE_DIR / "Market-Regime-Detection-Engine"
RL_DIR   = BASE_DIR / "Reinforcement-Learning-Trading-Agent"

for p in (LSTM_DIR, HMM_DIR, RL_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

# Engine imports
from lstm_vol_forecasting_engine import LSTMVolModel, create_sequences, realized_volatility
from market_regime_detection_engine import compute_features, fit_hmm, GaussianHMM
from rl_trading_agent import TradingEnv, DQNAgent, train_agent, sharpe_ratio, max_drawdown

# Registry imports (local)
from model_registry import (
    save_lstm, load_lstm,
    save_hmm, load_hmm,
    save_dqn, load_dqn,
    MODEL_DIR,
)

import torch
import torch.nn as nn
import torch.optim as optim

# ---------------------------------------------------------------------------
# 1. DATA
# ---------------------------------------------------------------------------

def fetch_data(ticker: str = "SPY", lookback: int = 504) -> pd.DataFrame:
    """Download OHLCV data via yfinance. Returns df with 'date' and 'close'."""
    import yfinance as yf
    end = pd.Timestamp.today()
    # Fetch extra days to account for weekends/holidays
    start = end - pd.Timedelta(days=int(lookback * 1.6))
    raw = yf.download(ticker, start=start.strftime("%Y-%m-%d"),
                       end=end.strftime("%Y-%m-%d"), progress=False)
    if raw.empty:
        raise RuntimeError(f"No data returned for {ticker}")
    df = raw.reset_index()
    # Handle multi-level columns from yfinance
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    # Normalise column names to lowercase
    df.columns = [c.lower() for c in df.columns]
    df = df.rename(columns={"adj close": "close"})
    df = df[["date", "close"]].dropna()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").tail(lookback).reset_index(drop=True)
    return df

# ---------------------------------------------------------------------------
# 2. LSTM STAGE
# ---------------------------------------------------------------------------

def run_lstm_stage(df: pd.DataFrame, retrain: bool = False,
                   seq_length: int = 20, hidden_size: int = 32,
                   num_layers: int = 1, epochs: int = 60, lr: float = 1e-3):
    """
    Generate volatility predictions for the full price series.
    Returns vol_predictions array aligned with df (NaN-padded at the front).
    """
    print("\n=== LSTM Volatility Forecasting Stage ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Feature engineering
    log_ret = np.log(df["close"]).diff()
    vol_series = (log_ret.rolling(seq_length).std() * np.sqrt(252)).dropna().values.astype(np.float32)

    X, y = create_sequences(vol_series, seq_length)
    X = X[..., np.newaxis]  # (N, seq, 1)
    y = y[..., np.newaxis]  # (N, 1)

    model = LSTMVolModel(input_size=1, hidden_size=hidden_size, num_layers=num_layers).to(device)

    if not retrain:
        model = load_lstm(model)
        # Check whether weights were actually loaded
        chk = MODEL_DIR / "lstm_vol_best.pt"
        if not chk.exists():
            print("  No saved LSTM found — training from scratch.")
            retrain = True

    if retrain:
        split = int(0.8 * len(X))
        X_train_t = torch.tensor(X[:split]).to(device)
        y_train_t = torch.tensor(y[:split]).to(device)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)

        best_loss = float("inf")
        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            out = model(X_train_t)
            loss = criterion(out, y_train_t)
            loss.backward()
            optimizer.step()
            if loss.item() < best_loss:
                best_loss = loss.item()
                best_state = model.state_dict()
            if (epoch + 1) % 20 == 0:
                print(f"  Epoch {epoch+1}/{epochs}  loss={loss.item():.6f}")

        model.load_state_dict(best_state)
        save_lstm(best_state)
        print(f"  LSTM saved  (best train loss={best_loss:.6f})")

    # Inference on full series
    model.eval()
    X_all = torch.tensor(X[..., :]).to(device)
    with torch.no_grad():
        preds = model(X_all).cpu().numpy().flatten()

    # Pad front so array aligns with df index
    pad_len = len(df) - len(preds)
    vol_predictions = np.concatenate([np.full(pad_len, np.nan), preds])
    print(f"  Latest vol forecast: {preds[-1]:.4f}")
    return vol_predictions

# ---------------------------------------------------------------------------
# 3. HMM STAGE
# ---------------------------------------------------------------------------

def run_hmm_stage(df: pd.DataFrame, retrain: bool = False,
                  n_states: int = 3, window: int = 20):
    """
    Generate regime labels for the full series.
    Returns (labels, regime_probs) arrays aligned with the feature-trimmed df,
    plus the trimmed df itself.
    """
    print("\n=== HMM Regime Detection Stage ===")
    df_feat, features = compute_features(df.copy(), window)

    hmm_model = None if retrain else load_hmm()

    if hmm_model is None:
        print("  Fitting new HMM …")
        hmm_model = GaussianHMM(n_components=n_states, covariance_type="full",
                                 n_iter=1000, random_state=42)
        hmm_model.fit(features)
        save_hmm(hmm_model)
        print("  HMM saved.")
    else:
        print("  Loaded cached HMM.")

    labels = hmm_model.predict(features)
    regime_probs = hmm_model.predict_proba(features)

    # Sort regimes by mean volatility so 0=bull, 1=sideways, 2=bear
    regime_vols = []
    for r in range(n_states):
        mask = labels == r
        regime_vols.append(df_feat.loc[mask, "log_return"].std())
    order = np.argsort(regime_vols)  # ascending vol → bull first
    remap = {old: new for new, old in enumerate(order)}
    labels = np.array([remap[l] for l in labels])
    regime_probs = regime_probs[:, order]

    regime_names = {0: "Bull", 1: "Sideways", 2: "Bear"}
    current_regime = regime_names.get(labels[-1], f"Regime {labels[-1]}")
    print(f"  Current regime: {current_regime}  (label={labels[-1]})")
    print(f"  Regime distribution: { {regime_names.get(i,i): int((labels==i).sum()) for i in range(n_states)} }")
    return labels, regime_probs, df_feat

# ---------------------------------------------------------------------------
# 4. DQN STAGE (augmented state)
# ---------------------------------------------------------------------------

class AugmentedTradingEnv(TradingEnv):
    """
    Extends TradingEnv to append vol-forecast and regime one-hot to the state.
    State = [window returns] + [vol_forecast] + [regime_onehot (3)]
    """
    def __init__(self, prices, vol_preds, regime_labels, n_regimes=3,
                 window_size=30, transaction_cost=0.001):
        # Store augmentation arrays BEFORE super().__init__ calls reset -> _get_state
        self.vol_preds = np.array(vol_preds, dtype=np.float32)
        self.regime_labels = np.array(regime_labels, dtype=np.int64)
        self.n_regimes = n_regimes
        super().__init__(prices, window_size, transaction_cost)

    def _get_state(self):
        base = super()._get_state()  # (window_size,)
        # vol prediction for current step (use last valid if NaN)
        idx = min(self.current_step, len(self.vol_preds) - 1)
        vp = self.vol_preds[idx]
        if np.isnan(vp):
            vp = 0.0
        # regime one-hot
        r_idx = min(self.current_step, len(self.regime_labels) - 1)
        onehot = np.zeros(self.n_regimes, dtype=np.float32)
        onehot[self.regime_labels[r_idx]] = 1.0
        return np.concatenate([base, [vp], onehot]).astype(np.float32)


def run_dqn_stage(df: pd.DataFrame, vol_predictions: np.ndarray,
                  regime_labels: np.ndarray, retrain: bool = False,
                  window_size: int = 30, episodes: int = 40, lr: float = 1e-3):
    """
    Train or load DQN agent with augmented state.  Returns final action + stats.
    """
    print("\n=== DQN Trading Agent Stage ===")
    prices = df["close"].values.astype(np.float64)

    # Align vol_predictions and regime_labels to prices length
    # vol_predictions is already len(df); regime_labels may be shorter (trimmed by rolling window)
    if len(regime_labels) < len(prices):
        pad = len(prices) - len(regime_labels)
        regime_labels = np.concatenate([np.zeros(pad, dtype=np.int64), regime_labels])

    n_regimes = 3
    state_dim = window_size + 1 + n_regimes  # returns window + vol + regime onehot

    env = AugmentedTradingEnv(prices, vol_predictions, regime_labels,
                               n_regimes=n_regimes, window_size=window_size)
    agent = DQNAgent(state_dim=state_dim, action_dim=3, lr=lr)

    if not retrain:
        agent = load_dqn(agent)
        chk = MODEL_DIR / "dqn_policy.pt"
        if not chk.exists():
            print("  No saved DQN found — training from scratch.")
            retrain = True

    if retrain:
        rewards, _, action_counts = train_agent(env, agent, episodes=episodes, verbose=False)
        save_dqn(agent.policy_net.state_dict())
        sr = sharpe_ratio(env.returns_history)
        mdd = max_drawdown(env.equity_curve)
        print(f"  DQN trained  ({episodes} eps)  Sharpe={sr:.3f}  MaxDD={mdd:.3f}")
        print(f"  DQN saved.")

    # Final forward pass for current signal
    env_eval = AugmentedTradingEnv(prices, vol_predictions, regime_labels,
                                    n_regimes=n_regimes, window_size=window_size)
    state = env_eval.reset()
    # Step through the entire series with greedy policy
    agent.epsilon = 0.0  # pure exploitation
    done = False
    while not done:
        action = agent.select_action(state)
        state, _, done, _ = env_eval.step(action)

    sr = sharpe_ratio(env_eval.returns_history)
    mdd = max_drawdown(env_eval.equity_curve)
    final_action = env_eval.actions[-1] if env_eval.actions else 0
    print(f"  Eval  Sharpe={sr:.3f}  MaxDD={mdd:.3f}  FinalEquity={env_eval.equity:.4f}")
    return final_action, sr, mdd, env_eval

# ---------------------------------------------------------------------------
# 5. MAIN ORCHESTRATOR
# ---------------------------------------------------------------------------

ACTION_MAP = {0: "HOLD", 1: "BUY", 2: "SELL"}
REGIME_MAP = {0: "Bull", 1: "Sideways", 2: "Bear"}


def main():
    parser = argparse.ArgumentParser(
        description="Quant Pipeline: LSTM vol → HMM regime → DQN trading signal"
    )
    parser.add_argument("--ticker", type=str, default="SPY", help="Ticker symbol (default: SPY)")
    parser.add_argument("--lookback", type=int, default=504, help="Trading days of history (default: 504)")
    parser.add_argument("--retrain-lstm", action="store_true", help="Force retrain LSTM model")
    parser.add_argument("--retrain-hmm", action="store_true", help="Force refit HMM model")
    parser.add_argument("--retrain-dqn", action="store_true", help="Force retrain DQN agent")
    args = parser.parse_args()

    print(f"Pipeline start  ticker={args.ticker}  lookback={args.lookback}")
    print(f"Model directory: {MODEL_DIR}")

    # --- Data ---
    df = fetch_data(ticker=args.ticker, lookback=args.lookback)
    print(f"Fetched {len(df)} rows  [{df['date'].iloc[0].date()} → {df['date'].iloc[-1].date()}]")

    # --- LSTM ---
    vol_predictions = run_lstm_stage(df, retrain=args.retrain_lstm)

    # --- HMM ---
    regime_labels, regime_probs, df_feat = run_hmm_stage(df, retrain=args.retrain_hmm)

    # --- DQN ---
    action_id, sr, mdd, env_eval = run_dqn_stage(
        df, vol_predictions, regime_labels, retrain=args.retrain_dqn
    )

    # --- Summary ---
    latest_vol = vol_predictions[~np.isnan(vol_predictions)][-1] if np.any(~np.isnan(vol_predictions)) else float("nan")
    current_regime_id = regime_labels[-1]

    print("\n" + "=" * 55)
    print("  PIPELINE SIGNAL OUTPUT")
    print("=" * 55)
    print(f"  Ticker          : {args.ticker}")
    print(f"  Date            : {df['date'].iloc[-1].date()}")
    print(f"  Action          : {ACTION_MAP.get(action_id, action_id)}")
    print(f"  Regime          : {REGIME_MAP.get(current_regime_id, current_regime_id)}")
    print(f"  Vol Forecast    : {latest_vol:.4f}")
    print(f"  Sharpe (eval)   : {sr:.3f}")
    print(f"  Max DD (eval)   : {mdd:.3f}")
    print(f"  Final Equity    : {env_eval.equity:.4f}")
    print("=" * 55)


if __name__ == "__main__":
    main()
