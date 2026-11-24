"""Tests for strategies: SMA signals, pairs cointegration and z-scores."""

import numpy as np
import pandas as pd
import pytest

from quant_backtester.data import generate_cointegrated_pair, generate_synthetic_prices
from quant_backtester.strategies import PairsTradingStrategy, SMACrossoverStrategy


def test_sma_long_when_short_above_long():
    # Steadily rising prices: short SMA must be above long SMA.
    prices = pd.DataFrame({"A": np.arange(1.0, 101.0)}, index=pd.bdate_range("2024-01-01", periods=100))
    strat = SMACrossoverStrategy(short_window=5, long_window=20)
    signals = strat.generate_signals(prices)
    assert signals.iloc[-1, 0] == 1.0
    assert signals.iloc[10, 0] == 0.0  # long SMA not yet available


def test_sma_flat_when_falling():
    prices = pd.DataFrame({"A": np.arange(100.0, 0.0, -1.0)}, index=pd.bdate_range("2024-01-01", periods=100))
    strat = SMACrossoverStrategy(short_window=5, long_window=20)
    signals = strat.generate_signals(prices)
    assert signals.iloc[-1, 0] == 0.0


def test_sma_invalid_windows():
    with pytest.raises(ValueError):
        SMACrossoverStrategy(short_window=50, long_window=20)
    with pytest.raises(ValueError):
        SMACrossoverStrategy(short_window=1, long_window=20)


def test_pairs_cointegration_on_synthetic_cointegrated_series():
    pair = generate_cointegrated_pair(n=500, seed=7)
    result = PairsTradingStrategy.test_cointegration(pair)
    assert result["is_cointegrated"] is True
    assert result["coint_pvalue"] < 0.05
    assert result["adf_pvalue"] < 0.05


def test_pairs_not_cointegrated_on_independent_walks():
    pair = generate_synthetic_prices(n=500, tickers=("A", "B"), seed=99)
    result = PairsTradingStrategy.test_cointegration(pair)
    # Independent GBMs are (almost always) not cointegrated.
    assert result["coint_pvalue"] > 0.05


def test_pairs_signals_are_bounded_and_two_legged():
    pair = generate_cointegrated_pair(n=300, seed=11)
    strat = PairsTradingStrategy(entry_z=2.0, exit_z=0.5, lookback=60)
    signals = strat.generate_signals(pair)
    assert signals.shape == pair.shape
    assert signals.abs().max().max() <= 2.0
    # Legs are opposite-signed whenever a spread position is open.
    spread_pos = signals.iloc[:, 1]
    assert ((signals.iloc[:, 0] == 0) | (np.sign(signals.iloc[:, 0]) == -np.sign(spread_pos))).all()


def test_pairs_requires_two_series():
    prices = generate_synthetic_prices(n=100, seed=5)
    with pytest.raises(ValueError):
        PairsTradingStrategy().generate_signals(prices)
