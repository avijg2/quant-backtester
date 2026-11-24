"""Trading strategies.

Every strategy returns *signals*: desired positions decided at the close
of bar ``t``. The engine shifts them by one bar before trading, so a
strategy can never profit from the same bar's price move it observed.

Both strategies are plain classes with explicit parameters, so they can
be grid-searched and walk-forward tested without hidden state.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, coint


class Strategy(ABC):
    """Base class: generate desired positions from prices."""

    @abstractmethod
    def generate_signals(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Return desired positions, same index/columns as ``prices``."""
        raise NotImplementedError


class SMACrossoverStrategy(Strategy):
    """Simple moving-average crossover, long-only.

    Long (1) when the short SMA is above the long SMA, flat (0)
    otherwise. SMAs are computed on closes up to and including bar
    ``t``; the engine executes the signal on bar ``t + 1``.
    """

    def __init__(self, short_window: int = 20, long_window: int = 50) -> None:
        if short_window >= long_window:
            raise ValueError("short_window must be < long_window")
        if short_window < 2:
            raise ValueError("short_window must be >= 2")
        self.short_window = short_window
        self.long_window = long_window

    def generate_signals(self, prices: pd.DataFrame) -> pd.DataFrame:
        if isinstance(prices, pd.Series):
            prices = prices.to_frame()
        short_sma = prices.rolling(self.short_window).mean()
        long_sma = prices.rolling(self.long_window).mean()
        signals = (short_sma > long_sma).astype(float)
        # No signal until both SMAs exist.
        signals = signals.where(long_sma.notna(), 0.0)
        return signals.fillna(0.0)


class PairsTradingStrategy(Strategy):
    """Pairs trading on a cointegrated pair, z-score entry/exit.

    For the two price series in ``prices`` (exactly two columns):

    * The hedge ratio ``beta`` is estimated by OLS on the trailing
      ``lookback`` bars only, so it never uses future data.
    * The spread is ``log(P2) - beta * log(P1)``, z-scored over the
      same trailing window.
    * Enter the spread when ``|z| > entry_z`` (long the spread when
      z is very negative, short when very positive) and exit when
      ``|z| < exit_z``.
    * Positions are returned per leg: leg 2 gets ``+/-1`` and leg 1
      gets ``-/+beta``, so the combined position is market-neutral
      in the spread sense.

    Use :meth:`test_cointegration` first to check the pair is actually
    cointegrated; trading a non-cointegrated pair is just betting on
    a random walk.
    """

    def __init__(self, entry_z: float = 2.0, exit_z: float = 0.5, lookback: int = 60) -> None:
        if entry_z <= exit_z:
            raise ValueError("entry_z must be > exit_z")
        if lookback < 20:
            raise ValueError("lookback must be >= 20")
        self.entry_z = entry_z
        self.exit_z = exit_z
        self.lookback = lookback

    @staticmethod
    def test_cointegration(prices: pd.DataFrame) -> dict:
        """Engle-Granger cointegration test and ADF test on the spread.

        Returns a dict with the cointegration p-value, the ADF
        statistic / p-value for the OLS spread, and the full-sample
        hedge ratio (for diagnostics only, not for trading).
        """
        if prices.shape[1] != 2:
            raise ValueError("test_cointegration needs exactly two price series")
        x = prices.iloc[:, 0].values
        y = prices.iloc[:, 1].values
        _, pvalue, _ = coint(x, y)
        # OLS hedge ratio on the full sample, diagnostics only.
        # Explicit OLS: y = a + b*x
        coef = np.polyfit(x, y, 1)
        beta = float(coef[0])
        spread = y - (coef[1] + beta * x)
        adf_stat, adf_p, *_ = adfuller(spread, result_object=False)
        return {
            "coint_pvalue": float(pvalue),
            "hedge_ratio_full_sample": beta,
            "adf_statistic": float(adf_stat),
            "adf_pvalue": float(adf_p),
            "is_cointegrated": bool(pvalue < 0.05),
        }

    def _rolling_beta_and_spread(self, prices: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
        """Trailing-window OLS hedge ratio and spread, no future data."""
        log_p = np.log(prices)
        x = log_p.iloc[:, 0]
        y = log_p.iloc[:, 1]
        beta = pd.Series(np.nan, index=prices.index, dtype=float)
        spread = pd.Series(np.nan, index=prices.index, dtype=float)
        for i in range(self.lookback, len(prices)):
            window_x = x.iloc[i - self.lookback : i]
            window_y = y.iloc[i - self.lookback : i]
            # OLS y = a + b*x on the trailing window.
            coef = np.polyfit(window_x.values, window_y.values, 1)
            b = float(coef[0])
            beta.iloc[i] = b
            spread.iloc[i] = y.iloc[i] - (coef[1] + b * x.iloc[i])
        return beta, spread

    def generate_signals(self, prices: pd.DataFrame) -> pd.DataFrame:
        if isinstance(prices, pd.Series):
            raise ValueError("PairsTradingStrategy needs two price series")
        if prices.shape[1] != 2:
            raise ValueError("PairsTradingStrategy needs exactly two price series")
        prices = prices.dropna()
        beta, spread = self._rolling_beta_and_spread(prices)
        spread_mean = spread.rolling(self.lookback).mean()
        spread_std = spread.rolling(self.lookback).std()
        zscore = (spread - spread_mean) / spread_std

        # Spread position: +1 long spread, -1 short spread, 0 flat.
        spread_position = pd.Series(0.0, index=prices.index)
        current = 0.0
        for i, date in enumerate(prices.index):
            z = zscore.iloc[i]
            if pd.isna(z):
                spread_position.iloc[i] = current
                continue
            if current == 0:
                if z < -self.entry_z:
                    current = 1.0
                elif z > self.entry_z:
                    current = -1.0
            else:
                if abs(z) < self.exit_z:
                    current = 0.0
            spread_position.iloc[i] = current

        signals = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        # Leg 2: the spread position itself. Leg 1: opposite, scaled by beta.
        signals.iloc[:, 1] = spread_position
        signals.iloc[:, 0] = -spread_position * beta.fillna(0.0)
        return signals.fillna(0.0)
