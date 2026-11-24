"""Tests for data generators and walk-forward validation."""

import numpy as np
import pandas as pd
import pytest

from quant_backtester.data import generate_cointegrated_pair, generate_synthetic_prices
from quant_backtester.engine import BacktestConfig, run_backtest
from quant_backtester.metrics import compute_metrics
from quant_backtester.strategies import SMACrossoverStrategy
from quant_backtester.walkforward import run_walk_forward, walk_forward_splits


def test_synthetic_prices_reproducible_and_positive():
    a = generate_synthetic_prices(n=100, seed=42)
    b = generate_synthetic_prices(n=100, seed=42)
    pd.testing.assert_frame_equal(a, b)
    assert (a > 0).all().all()
    assert len(a) == 100


def test_synthetic_prices_different_seeds_differ():
    a = generate_synthetic_prices(n=100, seed=1)
    b = generate_synthetic_prices(n=100, seed=2)
    assert not a.equals(b)


def test_walk_forward_splits_cover_and_do_not_overlap():
    splits = walk_forward_splits(n=1000, train_size=252, test_size=63)
    assert len(splits) == 11  # (1000-252-63)//63 + 1
    for train, test in splits:
        assert train.stop == test.start
        assert test.stop - test.start == 63
        assert train.stop - train.start == 252


def test_walk_forward_splits_reject_bad_sizes():
    with pytest.raises(ValueError):
        walk_forward_splits(n=100, train_size=0, test_size=10)


def test_walk_forward_runs_and_returns_oos_metrics():
    prices = generate_synthetic_prices(n=756, seed=42)
    wf = run_walk_forward(prices, lambda: SMACrossoverStrategy(20, 50), train_size=252, test_size=63)
    assert len(wf) == 8  # (756-252-63)//63 + 1
    assert set(wf.columns) >= {"fold", "sharpe_ratio", "total_return", "max_drawdown"}
    assert wf["max_drawdown"].le(0).all()
    # Guards against the signals/test-window misalignment bug: at least
    # some out-of-sample folds must actually trade.
    assert (wf["total_return"] != 0).sum() >= 3


def test_full_pipeline_sma_on_synthetic():
    prices = generate_synthetic_prices(n=504, tickers=("SPY",), seed=42)
    signals = SMACrossoverStrategy(20, 50).generate_signals(prices)
    result = run_backtest(prices, signals, config=BacktestConfig())
    metrics = compute_metrics(result.equity_curve, result.returns)
    assert result.final_equity > 0
    assert metrics["sharpe_ratio"] == metrics["sharpe_ratio"]  # not NaN
    assert -1.0 < metrics["max_drawdown"] <= 0.0
