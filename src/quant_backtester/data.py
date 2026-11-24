"""Price data loading and synthetic data generation.

Real data comes from yfinance with ``auto_adjust=True`` so prices already
reflect dividends and splits. Downloads are cached as CSV files under
``cache_dir`` so repeated runs and tests do not hit the network every time.

Synthetic data is geometric Brownian motion (GBM) and a cointegrated pair.
It is used for tests, offline runs, and the example when the network is
unavailable. Every generator takes an explicit ``seed`` so results are
reproducible.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _cache_path(cache_dir: str | Path, tickers: list[str], start: str, end: str) -> Path:
    name = "_".join(tickers) + f"_{start}_{end}.csv"
    return Path(cache_dir) / name


def load_prices(
    tickers: list[str] | str,
    start: str = "2020-01-01",
    end: str | None = None,
    cache_dir: str | Path = "data",
    use_cache: bool = True,
) -> pd.DataFrame:
    """Load adjusted close prices for ``tickers`` from yfinance.

    Parameters
    ----------
    tickers:
        One ticker or a list, e.g. ``"SPY"`` or ``["AAPL", "MSFT"]``.
    start, end:
        Date range, ``YYYY-MM-DD``. ``end`` defaults to today.
    cache_dir:
        Directory for the local CSV cache.
    use_cache:
        If True and a cached CSV exists, load it instead of downloading.

    Returns
    -------
    DataFrame of adjusted close prices, indexed by date, one column
    per ticker. Rows with any missing price are dropped and the index
    is sorted. Raises ``ValueError`` if no usable data is returned.
    """
    if isinstance(tickers, str):
        tickers = [tickers]
    tickers = [t.upper() for t in tickers]
    end = end or pd.Timestamp.today().strftime("%Y-%m-%d")

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = _cache_path(cache_dir, tickers, start, end)

    if use_cache and path.exists():
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        if not df.empty:
            return df.sort_index()

    import yfinance as yf  # imported lazily so tests can run without it

    raw = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        threads=False,
    )
    if raw is None or raw.empty:
        raise ValueError(f"No data returned by yfinance for {tickers} {start} to {end}")

    # yfinance returns a MultiIndex column frame for multiple tickers.
    if isinstance(raw.columns, pd.MultiIndex):
        if "Close" in raw.columns.get_level_values(0):
            df = raw["Close"].copy()
        else:
            df = raw.xs("Close", level=0, axis=1).copy()
    elif "Close" in raw.columns:
        df = raw[["Close"]].copy()
        df.columns = tickers
    else:
        df = raw.copy()

    df = df.dropna(how="all").sort_index()
    # Forward-fill at most a few days (holidays / missing bars), then drop
    # anything still missing so the engine never sees NaN prices.
    df = df.ffill(limit=3).dropna()
    if df.empty:
        raise ValueError(f"No usable price data for {tickers} {start} to {end}")

    df.to_csv(path)
    return df


def generate_synthetic_prices(
    n: int = 756,
    tickers: tuple[str, ...] = ("SYN1",),
    seed: int = 42,
    start_price: float = 100.0,
    mu: float = 0.0003,
    sigma: float = 0.01,
    start: str = "2023-01-02",
) -> pd.DataFrame:
    """Generate GBM price series.

    ``mu`` and ``sigma`` are daily drift and volatility. Each ticker gets
    its own independent random walk, seeded for reproducibility.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start=start, periods=n)
    data: dict[str, np.ndarray] = {}
    for ticker in tickers:
        shocks = rng.normal(loc=mu, scale=sigma, size=n)
        log_returns = np.cumsum(shocks)
        data[ticker] = start_price * np.exp(log_returns)
    return pd.DataFrame(data, index=dates)


def generate_cointegrated_pair(
    n: int = 500,
    seed: int = 7,
    beta: float = 1.5,
    start: str = "2023-01-02",
) -> pd.DataFrame:
    """Generate a cointegrated pair (X, Y) with Y ≈ beta * X + noise.

    X is a random walk. Y is beta * X plus stationary AR(1) noise, so the
    spread ``Y - beta * X`` is stationary and the pair is cointegrated
    by construction. This is the ground truth used in the pairs tests.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start=start, periods=n)
    x_increments = rng.normal(scale=1.0, size=n)
    x = 100.0 + np.cumsum(x_increments)
    # Stationary AR(1) noise, phi = 0.7.
    noise = np.zeros(n)
    eps = rng.normal(scale=0.5, size=n)
    for i in range(1, n):
        noise[i] = 0.7 * noise[i - 1] + eps[i]
    y = beta * x + noise
    # Keep prices positive for log-price use in the strategy.
    y = y - y.min() + 50.0
    x = x - x.min() + 50.0
    return pd.DataFrame({"X": x, "Y": y}, index=dates)
