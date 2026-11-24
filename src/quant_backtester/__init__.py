"""Quant Backtester: a clean, no-lookahead backtesting engine.

Modules
-------
data
    Price loading (yfinance with a local CSV cache) and synthetic data.
engine
    Vectorized backtester with transaction costs and slippage.
strategies
    SMA crossover and pairs-trading strategies.
metrics
    Sharpe, Sortino, max drawdown, Calmar, CAGR, win rate, profit factor.
walkforward
    Walk-forward validation splits and out-of-sample evaluation.
cli
    Command-line entry point.
"""

from .data import generate_cointegrated_pair, generate_synthetic_prices, load_prices
from .engine import BacktestConfig, BacktestResult, run_backtest
from .metrics import compute_metrics, max_drawdown
from .strategies import PairsTradingStrategy, SMACrossoverStrategy
from .walkforward import run_walk_forward, walk_forward_splits

__all__ = [
    "BacktestConfig",
    "BacktestResult",
    "PairsTradingStrategy",
    "SMACrossoverStrategy",
    "compute_metrics",
    "generate_cointegrated_pair",
    "generate_synthetic_prices",
    "load_prices",
    "max_drawdown",
    "run_backtest",
    "run_walk_forward",
    "walk_forward_splits",
]

__version__ = "0.1.0"
