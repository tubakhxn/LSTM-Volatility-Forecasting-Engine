"""
test_pipeline.py -- Integration tests for the full quant pipeline.

Covers:
    1. Pipeline imports work (all three engines can be imported together)
    2. Synthetic data pipeline produces valid (action, regime, vol_forecast) tuple
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

# Ensure all three sibling repos are importable
_ROOT = Path(__file__).resolve().parent.parent.parent
for repo in (
    "LSTM-Volatility-Forecasting-Engine",
    "Market-Regime-Detection-Engine",
    "Reinforcement-Learning-Trading-Agent",
):
    p = str(_ROOT / repo)
    if p not in sys.path:
        sys.path.insert(0, p)


# ---------------------------------------------------------------
# Tests
# ---------------------------------------------------------------
class TestPipelineImports:
    """All three engines can be imported in one process."""

    def test_lstm_imports(self):
        from lstm_vol_forecasting_engine import LSTMVolModel, create_sequences, realized_volatility
        assert callable(LSTMVolModel)

    def test_hmm_imports(self):
        from market_regime_detection_engine import compute_features, fit_hmm
        assert callable(compute_features)
        assert callable(fit_hmm)

    def test_dqn_imports(self):
        from rl_trading_agent import TradingEnv, DQNAgent, train_agent
        assert callable(TradingEnv)
        assert callable(DQNAgent)

    def test_model_registry_imports(self):
        from model_registry import save_lstm, load_lstm, save_hmm, load_hmm, save_dqn, load_dqn
        assert callable(save_lstm)


class TestSyntheticPipeline:
    """Synthetic data pipeline produces a valid (action, regime, vol_forecast) tuple."""

    def test_end_to_end(self):
        from lstm_vol_forecasting_engine import (
            LSTMVolModel,
            create_sequences,
            realized_volatility,
            generate_gbm,
        )
        from market_regime_detection_engine import compute_features, fit_hmm
        from rl_trading_agent import TradingEnv, DQNAgent

        # --- Step 1: Generate synthetic data ---
        df = generate_gbm(n_days=500)

        # --- Step 2: LSTM volatility forecast ---
        data = df.copy()
        data["log_return"] = np.log(data["close"]).diff()
        data["realized_vol"] = realized_volatility(data["log_return"], window=20)
        data = data.dropna().reset_index(drop=True)
        vol = data["realized_vol"].values.astype(np.float32)

        X, y = create_sequences(vol, seq_length=20)
        X_tensor = torch.tensor(X[..., np.newaxis], dtype=torch.float32)

        model = LSTMVolModel(input_size=1, hidden_size=32, num_layers=1)
        model.eval()
        with torch.no_grad():
            vol_forecast = model(X_tensor[-1:]).item()

        assert isinstance(vol_forecast, float)
        assert np.isfinite(vol_forecast)

        # --- Step 3: HMM regime detection ---
        _, features = compute_features(df, window=20)
        hidden_states, _, _, _, _ = fit_hmm(features, n_states=3)
        regime = int(hidden_states[-1])

        assert regime in (0, 1, 2)

        # --- Step 4: DQN action selection ---
        prices = df["close"].values.astype(float)
        env = TradingEnv(prices, window_size=30, transaction_cost=0.001)
        agent = DQNAgent(state_dim=30, action_dim=3)
        state = env.reset()
        action = agent.select_action(state)

        assert action in (0, 1, 2)

        # --- Final tuple ---
        result = (action, regime, vol_forecast)
        assert len(result) == 3
        assert isinstance(result[0], (int, np.integer))
        assert isinstance(result[1], int)
        assert isinstance(result[2], float)
