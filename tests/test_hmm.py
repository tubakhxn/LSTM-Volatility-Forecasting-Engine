"""
test_hmm.py -- Unit tests for the Market Regime Detection Engine.

Covers:
    1. HMM returns exactly 3 distinct regime labels
    2. Transition matrix rows sum to approximately 1.0
    3. compute_features returns correct shape
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Ensure sibling repo is importable
_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT / "Market-Regime-Detection-Engine"))

from market_regime_detection_engine import (
    compute_features,
    fit_hmm,
    generate_gbm,
)

# ---------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------
@pytest.fixture
def synthetic_df():
    return generate_gbm(start_price=100, mu=0.08, sigma=0.2, days=500, seed=42)


@pytest.fixture
def features(synthetic_df):
    _, feats = compute_features(synthetic_df, window=20)
    return feats


@pytest.fixture
def hmm_results(features):
    return fit_hmm(features, n_states=3, random_state=42)


# ---------------------------------------------------------------
# Tests
# ---------------------------------------------------------------
class TestHMMRegimeLabels:
    """HMM returns exactly 3 distinct regime labels."""

    def test_three_regimes(self, hmm_results):
        hidden_states, _, _, _, _ = hmm_results
        unique = np.unique(hidden_states)
        assert len(unique) == 3, f"Expected 3 regimes, got {len(unique)}: {unique}"

    def test_labels_in_range(self, hmm_results):
        hidden_states, _, _, _, _ = hmm_results
        assert hidden_states.min() >= 0
        assert hidden_states.max() <= 2


class TestTransitionMatrix:
    """Transition matrix rows sum to approximately 1.0."""

    def test_row_sums(self, hmm_results):
        _, _, transmat, _, _ = hmm_results
        row_sums = transmat.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-6)

    def test_shape(self, hmm_results):
        _, _, transmat, _, _ = hmm_results
        assert transmat.shape == (3, 3)

    def test_non_negative(self, hmm_results):
        _, _, transmat, _, _ = hmm_results
        assert (transmat >= 0).all()


class TestComputeFeatures:
    """compute_features returns correct shape."""

    def test_feature_columns(self, synthetic_df):
        df_out, features = compute_features(synthetic_df, window=20)
        # features should have 2 columns: log_return, volatility
        assert features.shape[1] == 2

    def test_no_nans(self, synthetic_df):
        df_out, features = compute_features(synthetic_df, window=20)
        assert not np.isnan(features).any()

    def test_row_count_matches(self, synthetic_df):
        df_out, features = compute_features(synthetic_df, window=20)
        assert len(df_out) == features.shape[0]
