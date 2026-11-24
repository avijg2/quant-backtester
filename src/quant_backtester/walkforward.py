"""Walk-forward validation.

In-sample optimisation on the whole history overfits. Walk-forward
validation instead trains / fits on a window and tests on the next,
out-of-sample window, then rolls forward. The reported metrics are
the out-of-sample ones only.
"""

from __future__ import annotations

import pandas as pd

from .engine import BacktestConfig, run_backtest
from .metrics import compute_metrics
from .strategies import Strategy


def walk_forward_splits(
    n: int,
    train_size: int,
    test_size: int,
    step: int | None = None,
) -> list[tuple[slice, slice]]:
    """Generate (train, test) index slices.

    Windows roll forward by ``step`` (default ``test_size``) until
    there is no full test window left.
    """
    if train_size <= 0 or test_size <= 0:
        raise ValueError("train_size and test_size must be positive")
    step = step or test_size
    splits: list[tuple[slice, slice]] = []
    start = 0
    while start + train_size + test_size <= n:
        train = slice(start, start + train_size)
        test = slice(start + train_size, start + train_size + test_size)
        splits.append((train, test))
        start += step
    return splits


def run_walk_forward(
    prices: pd.DataFrame,
    strategy_factory,
    train_size: int = 252,
    test_size: int = 63,
    config: BacktestConfig | None = None,
) -> pd.DataFrame:
    """Run walk-forward validation for a strategy factory.

    ``strategy_factory`` is a callable returning a fresh :class:`Strategy`
    (e.g. ``lambda: SMACrossoverStrategy(20, 50)``). It is called once
    per fold so no state leaks between folds. Signals are generated on
    the combined train+test window for each fold, but the backtest and
    metrics are computed on the test window only.
    """
    config = config or BacktestConfig()
    splits = walk_forward_splits(len(prices), train_size, test_size)
    if not splits:
        raise ValueError("Not enough data for even one walk-forward fold")
    rows = []
    for fold, (train_slice, test_slice) in enumerate(splits):
        strategy: Strategy = strategy_factory()
        window = prices.iloc[train_slice.start : test_slice.stop]
        signals = strategy.generate_signals(window)
        test_prices = prices.iloc[test_slice]
        # Align by date, not by absolute position: ``signals`` is indexed
        # from the start of the combined window, so .iloc[test_slice]
        # would be out of bounds for every fold after the first.
        test_signals = signals.loc[test_prices.index]
        result = run_backtest(test_prices, test_signals, config=config)
        metrics = compute_metrics(result.equity_curve, result.returns)
        rows.append(
            {
                "fold": fold,
                "test_start": str(test_prices.index[0].date()),
                "test_end": str(test_prices.index[-1].date()),
                "total_return": metrics["total_return"],
                "sharpe_ratio": metrics["sharpe_ratio"],
                "max_drawdown": metrics["max_drawdown"],
                "win_rate": metrics["win_rate"],
            }
        )
    return pd.DataFrame(rows)
