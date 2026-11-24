"""Tests for the backtesting engine: no-lookahead, costs, equity."""

import numpy as np
import pandas as pd
import pytest

from quant_backtester.data import generate_synthetic_prices
from quant_backtester.engine import BacktestConfig, run_backtest, run_event_backtest


def _prices_with_jump() -> pd.Series:
    # Flat at 100, then a single +10% jump on day 5, then flat.
    prices = pd.Series(
        [100.0, 100.0, 100.0, 100.0, 100.0, 110.0, 110.0, 110.0],
        index=pd.bdate_range("2024-01-01", periods=8),
        name="SPY",
    )
    return prices


def test_no_lookahead_signal_cannot_profit_from_same_day_jump():
    """A signal that turns long *on* the jump day must not capture it."""
    prices = _prices_with_jump()
    # Signal is 1 only on the jump day (index 5) — a strategy that saw
    # the jump at the close. The engine must execute it the next bar,
    # when the price is already 110 and flat, so no profit.
    signals = pd.Series([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0], index=prices.index)
    result = run_backtest(prices, signals, config=BacktestConfig(commission_bps=0, slippage_bps=0))
    # No profit from the jump itself; equity should be ~initial capital.
    assert result.final_equity == pytest.approx(100_000.0, rel=1e-9)


def test_signal_before_jump_does_capture_it():
    prices = _prices_with_jump()
    signals = pd.Series([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0], index=prices.index)
    result = run_backtest(prices, signals, config=BacktestConfig(commission_bps=0, slippage_bps=0))
    # Long before the jump: +10% on $100k.
    assert result.final_equity == pytest.approx(110_000.0, rel=1e-6)


def test_costs_reduce_returns():
    prices = generate_synthetic_prices(n=252, seed=1)["SYN1"]
    signals = pd.Series(1.0, index=prices.index)
    free = run_backtest(prices, signals, config=BacktestConfig(commission_bps=0, slippage_bps=0))
    costly = run_backtest(prices, signals, config=BacktestConfig(commission_bps=50, slippage_bps=50))
    # Always-long has only one entry trade, so costs are small but present.
    assert costly.final_equity <= free.final_equity
    # A churning signal pays costs repeatedly.
    churn = pd.Series([1.0 if i % 2 == 0 else 0.0 for i in range(len(prices))], index=prices.index)
    churn_free = run_backtest(prices, churn, config=BacktestConfig(commission_bps=0, slippage_bps=0))
    churn_costly = run_backtest(prices, churn, config=BacktestConfig(commission_bps=20, slippage_bps=10))
    assert churn_costly.final_equity < churn_free.final_equity


def test_equity_starts_at_initial_capital():
    prices = generate_synthetic_prices(n=100, seed=2)["SYN1"]
    signals = pd.Series(1.0, index=prices.index)
    result = run_backtest(prices, signals)
    assert result.equity_curve.iloc[0] == pytest.approx(100_000.0)
    assert len(result.equity_curve) == len(prices)


def test_flat_strategy_stays_flat_minus_costs():
    prices = generate_synthetic_prices(n=100, seed=3)["SYN1"]
    signals = pd.Series(0.0, index=prices.index)
    result = run_backtest(prices, signals, config=BacktestConfig(commission_bps=5, slippage_bps=2))
    assert result.final_equity == pytest.approx(100_000.0)
    assert len(result.trades) == 0


def test_event_and_vectorized_agree():
    prices = generate_synthetic_prices(n=200, seed=4)["SYN1"]
    signals = pd.Series([1.0 if i % 3 == 0 else 0.0 for i in range(len(prices))], index=prices.index)
    config = BacktestConfig(commission_bps=5, slippage_bps=2)
    vec = run_backtest(prices, signals, config=config)
    evt = run_event_backtest(prices, signals, config=config)
    pd.testing.assert_series_equal(vec.equity_curve, evt.equity_curve, check_names=False)
    pd.testing.assert_series_equal(vec.returns, evt.returns, check_names=False)
