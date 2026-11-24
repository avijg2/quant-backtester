"""Performance metrics.

All returns are daily unless stated otherwise. Annualisation uses
252 trading days per year. Sharpe and Sortino use the daily risk-free
rate derived from ``risk_free_rate`` (annual).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def max_drawdown(equity_curve: pd.Series) -> float:
    """Maximum peak-to-trough drawdown, as a negative fraction (e.g. -0.20)."""
    equity = pd.Series(equity_curve).astype(float)
    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0
    return float(drawdown.min())


def compute_metrics(
    equity_curve: pd.Series,
    returns: pd.Series | None = None,
    risk_free_rate: float = 0.0,
    periods_per_year: int = TRADING_DAYS,
    trades: pd.DataFrame | None = None,
) -> dict[str, float]:
    """Compute the standard backtest metrics from an equity curve.

    Parameters
    ----------
    equity_curve:
        Portfolio value over time. The first value is initial capital.
    returns:
        Daily strategy returns. If omitted, derived from the equity curve.
    risk_free_rate:
        Annual risk-free rate, e.g. 0.02 for 2%.
    trades:
        Optional trade log; if given, turnover is the mean absolute
        daily position change implied by the trade count, otherwise 0.
    """
    equity = pd.Series(equity_curve).astype(float).dropna()
    if len(equity) < 2:
        raise ValueError("equity_curve needs at least 2 points")
    if returns is None:
        returns = equity.pct_change().fillna(0.0)
    returns = pd.Series(returns).astype(float).fillna(0.0)

    n = len(returns)
    initial = float(equity.iloc[0])
    final = float(equity.iloc[-1])
    total_return = final / initial - 1.0
    # CAGR over the realised number of trading days.
    years = max(n / periods_per_year, 1e-9)
    cagr = (final / initial) ** (1.0 / years) - 1.0 if final > 0 else -1.0

    daily_rf = risk_free_rate / periods_per_year
    excess = returns - daily_rf
    excess_std = float(excess.std(ddof=1)) if n > 1 else 0.0
    volatility = float(returns.std(ddof=1) * np.sqrt(periods_per_year)) if n > 1 else 0.0
    # Treat near-zero volatility as zero: floating-point noise in an
    # otherwise constant return series must not produce a huge Sharpe.
    sharpe = (
        float(excess.mean() / excess_std * np.sqrt(periods_per_year))
        if excess_std > 1e-12
        else 0.0
    )

    # Downside deviation: mean of squared negative excess returns over
    # *all* periods (upside days contribute 0), not just downside days.
    # Averaging only over downside days inflates the denominator and
    # understates Sortino by an order of magnitude.
    downside_sq = np.minimum(excess.values, 0.0) ** 2
    downside_dev = float(np.sqrt(downside_sq.mean()) * np.sqrt(periods_per_year))
    sortino = (
        float(excess.mean() * periods_per_year / downside_dev) if downside_dev > 0 else 0.0
    )

    mdd = max_drawdown(equity)
    calmar = float(cagr / abs(mdd)) if mdd != 0 else 0.0

    nonzero = returns[returns != 0]
    win_rate = float((nonzero > 0).mean()) if len(nonzero) else 0.0
    gains = float(returns[returns > 0].sum())
    losses = float(abs(returns[returns < 0].sum()))
    profit_factor = float(gains / losses) if losses > 0 else float("inf") if gains > 0 else 0.0

    turnover = 0.0
    if trades is not None and len(trades) and n:
        turnover = float(len(trades) / n)

    return {
        "total_return": float(total_return),
        "cagr": float(cagr),
        "volatility": float(volatility),
        "sharpe_ratio": float(sharpe),
        "sortino_ratio": float(sortino),
        "max_drawdown": float(mdd),
        "calmar_ratio": float(calmar),
        "win_rate": float(win_rate),
        "profit_factor": float(profit_factor),
        "turnover_trades_per_day": float(turnover),
        "final_equity": float(final),
        "n_days": float(n),
    }
