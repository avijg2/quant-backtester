"""Backtesting engine.

The core rule, enforced everywhere in this module, is **no lookahead**:

* A signal computed from prices up to and including close ``t`` may only
  change the position from the *next* bar onwards.
* In code that is ``positions = signals.shift(1)``. The return on day ``t``
  is therefore earned by the position decided at ``t - 1``.

Costs are charged on turnover: every unit change in position pays
``commission_bps + slippage_bps`` in basis points of traded notional.
Costs are applied on the bar where the position actually changes, i.e.
the bar after the signal.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class BacktestConfig:
    """Backtest assumptions. All costs are in basis points (1 bp = 0.01%)."""

    initial_capital: float = 100_000.0
    commission_bps: float = 5.0
    slippage_bps: float = 2.0
    risk_free_rate: float = 0.0

    @property
    def total_cost_bps(self) -> float:
        return self.commission_bps + self.slippage_bps


@dataclass
class BacktestResult:
    """Output of :func:`run_backtest`."""

    equity_curve: pd.Series
    positions: pd.DataFrame
    signals: pd.DataFrame
    returns: pd.Series
    trades: pd.DataFrame
    config: BacktestConfig = field(repr=False)

    @property
    def final_equity(self) -> float:
        return float(self.equity_curve.iloc[-1])

    @property
    def total_return(self) -> float:
        return float(self.equity_curve.iloc[-1] / self.equity_curve.iloc[0] - 1.0)


def _as_frame(prices: pd.Series | pd.DataFrame) -> pd.DataFrame:
    if isinstance(prices, pd.Series):
        return prices.to_frame(name=prices.name or "price")
    return prices.copy()


def run_backtest(
    prices: pd.Series | pd.DataFrame,
    signals: pd.Series | pd.DataFrame,
    config: BacktestConfig | None = None,
) -> BacktestResult:
    """Run a vectorized backtest.

    Parameters
    ----------
    prices:
        Close prices, indexed by date. Series for one asset, DataFrame
        for several (one column per asset).
    signals:
        Desired position per asset, same shape/index as ``prices``.
        Convention: ``1`` = fully long, ``0`` = flat, ``-1`` = fully
        short. Fractional values are allowed for partial sizing.
        Signals are *decisions at the close*; they take effect next bar.
    config:
        Capital and cost assumptions. Defaults to $100k, 5 bps
        commission, 2 bps slippage.

    Returns
    -------
    BacktestResult with the equity curve, realised positions, daily
    strategy returns, and a trade log (one row per position change).
    """
    config = config or BacktestConfig()
    price_df = _as_frame(prices)
    sig_df = _as_frame(signals)

    if not price_df.index.equals(sig_df.index):
        sig_df = sig_df.reindex(price_df.index).fillna(0.0)
    sig_df = sig_df.fillna(0.0)
    # A single-column Series signal has no meaningful column name;
    # align it to the price column so the two frames multiply correctly.
    if sig_df.shape[1] == 1 and price_df.shape[1] == 1:
        sig_df.columns = price_df.columns
    elif list(sig_df.columns) != list(price_df.columns):
        sig_df = sig_df.reindex(columns=price_df.columns).fillna(0.0)

    # --- No lookahead: positions lag signals by one bar. ---
    positions = sig_df.shift(1).fillna(0.0)

    # Daily simple returns per asset.
    asset_returns = price_df.pct_change().fillna(0.0)

    # Gross strategy return: equally weighted across assets held.
    # For a single asset this is just position * return.
    gross_returns = (positions * asset_returns).sum(axis=1)

    # Turnover: absolute change in position, summed across assets.
    # A flip from 1 to -1 is turnover 2, a flip from 0 to 1 is 1.
    turnover = positions.diff().abs().sum(axis=1).fillna(0.0)
    # First bar: entering from flat.
    turnover.iloc[0] = positions.iloc[0].abs().sum()
    cost_rate = config.total_cost_bps / 10_000.0
    costs = turnover * cost_rate

    net_returns = gross_returns - costs
    equity = config.initial_capital * (1.0 + net_returns).cumprod()
    equity.iloc[0] = config.initial_capital

    # Trade log: one row per asset per bar where the position changed.
    rows = []
    pos_change = positions.diff().fillna(positions.iloc[[0]])
    for date, row in pos_change.iterrows():
        for asset, delta in row.items():
            if delta != 0:
                rows.append(
                    {
                        "date": date,
                        "asset": asset,
                        "position_change": float(delta),
                        "new_position": float(positions.loc[date, asset]),
                        "price": float(price_df.loc[date, asset]),
                        "cost_bps": float(config.total_cost_bps),
                    }
                )
    trades = pd.DataFrame(rows, columns=["date", "asset", "position_change", "new_position", "price", "cost_bps"])

    return BacktestResult(
        equity_curve=equity,
        positions=positions,
        signals=sig_df,
        returns=net_returns,
        trades=trades,
        config=config,
    )


def run_event_backtest(
    prices: pd.Series | pd.DataFrame,
    signals: pd.Series | pd.DataFrame,
    config: BacktestConfig | None = None,
) -> BacktestResult:
    """Bar-by-bar replay of the same logic as :func:`run_backtest`.

    Provided as a cross-check: the event loop must agree with the
    vectorized path (within floating-point tolerance). It iterates bars
    in order and only ever uses the previous bar's signal to set the
    current position, which makes the no-lookahead rule explicit.
    """
    config = config or BacktestConfig()
    price_df = _as_frame(prices)
    sig_df = _as_frame(signals).reindex(price_df.index).fillna(0.0)
    if sig_df.shape[1] == 1 and price_df.shape[1] == 1:
        sig_df.columns = price_df.columns
    elif list(sig_df.columns) != list(price_df.columns):
        sig_df = sig_df.reindex(columns=price_df.columns).fillna(0.0)
    asset_returns = price_df.pct_change().fillna(0.0)

    positions = pd.DataFrame(0.0, index=price_df.index, columns=price_df.columns)
    net_returns = pd.Series(0.0, index=price_df.index, dtype=float)
    cost_rate = config.total_cost_bps / 10_000.0

    prev_position = pd.Series(0.0, index=price_df.columns)
    for i, date in enumerate(price_df.index):
        if i == 0:
            current = pd.Series(0.0, index=price_df.columns)
        else:
            # Only the previous bar's signal is known at this point.
            current = sig_df.iloc[i - 1]
        positions.loc[date] = current
        turnover = (current - prev_position).abs().sum()
        gross = float((current * asset_returns.iloc[i]).sum())
        net_returns.loc[date] = gross - turnover * cost_rate
        prev_position = current

    equity = config.initial_capital * (1.0 + net_returns).cumprod()
    equity.iloc[0] = config.initial_capital
    trades = pd.DataFrame(
        columns=["date", "asset", "position_change", "new_position", "price", "cost_bps"]
    )
    return BacktestResult(
        equity_curve=equity,
        positions=positions,
        signals=sig_df,
        returns=net_returns,
        trades=trades,
        config=config,
    )
