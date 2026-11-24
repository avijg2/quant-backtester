"""Tests for performance metrics: hand-checkable math."""

import numpy as np
import pandas as pd
import pytest

from quant_backtester.metrics import compute_metrics, max_drawdown


def test_max_drawdown_known():
    equity = pd.Series([100.0, 120.0, 90.0, 110.0, 80.0])
    # Peak 120 -> trough 80 = -33.33%
    assert max_drawdown(equity) == pytest.approx(-1 / 3, rel=1e-9)
    assert max_drawdown(pd.Series([100.0, 110.0, 120.0])) == pytest.approx(0.0)


def test_cagr_and_total_return():
    # +10% over exactly 252 days => CAGR 10%.
    equity = pd.Series(np.linspace(100, 110, 253))
    # Use returns derived from equity; CAGR uses n/252 years.
    m = compute_metrics(equity)
    assert m["total_return"] == pytest.approx(0.10, rel=1e-6)
    assert m["cagr"] == pytest.approx(0.10, rel=0.02)


def test_sharpe_zero_when_no_volatility():
    # Constant +0.1% daily returns: excess std is 0, Sharpe defined as 0.
    returns = pd.Series([0.001] * 100)
    equity = 100_000 * (1 + returns).cumprod()
    m = compute_metrics(equity, returns)
    assert m["sharpe_ratio"] == 0.0
    assert m["volatility"] == pytest.approx(0.0)


def test_sharpe_positive_for_steady_gains():
    rng = np.random.default_rng(0)
    returns = pd.Series(rng.normal(0.001, 0.005, 252))
    equity = 100_000 * (1 + returns).cumprod()
    m = compute_metrics(equity, returns)
    assert m["sharpe_ratio"] > 0
    assert m["volatility"] == pytest.approx(returns.std(ddof=1) * np.sqrt(252), rel=1e-6)


def test_sortino_exceeds_sharpe_for_positive_mean():
    # Downside deviation is smaller than total volatility when the mean
    # is positive, so Sortino must exceed Sharpe (and be O(1), not 0.03).
    rng = np.random.default_rng(0)
    returns = pd.Series(rng.normal(0.001, 0.005, 252))
    equity = 100_000 * (1 + returns).cumprod()
    m = compute_metrics(equity, returns)
    assert m["sortino_ratio"] > m["sharpe_ratio"] > 0
    assert m["sortino_ratio"] < 10.0


def test_win_rate_and_profit_factor():
    returns = pd.Series([0.02, -0.01, 0.02, -0.01, 0.02])
    equity = 100_000 * (1 + returns).cumprod()
    m = compute_metrics(equity, returns)
    assert m["win_rate"] == pytest.approx(0.6)
    # Gains 0.06, losses 0.02 => profit factor 3.
    assert m["profit_factor"] == pytest.approx(3.0)


def test_metrics_reject_too_short_equity():
    with pytest.raises(ValueError):
        compute_metrics(pd.Series([100.0]))
