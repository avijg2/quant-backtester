# Quant Backtester

A clean, no-lookahead backtesting engine for quantitative finance research, in Python.

The point of this repo is not a magic strategy. It is a correct engine: signals are computed only from past data, trades execute on the next bar, costs are charged on every position change, and every result is checked out-of-sample with walk-forward validation. A strategy that only works in-sample, or only works if you ignore costs, will fail loudly here.

## What it does

- **Vectorized + event-replay backtester** (`engine.py`) with transaction costs and slippage, position sizing via signals, and a trade log. The vectorized and bar-by-bar paths are cross-checked against each other in tests.
- **Two strategies** (`strategies.py`):
  - SMA crossover momentum (long-only, 20/50 by default).
  - Pairs trading on a cointegrated pair: trailing-window OLS hedge ratio, z-score entry/exit, Engle-Granger cointegration test and ADF test on the spread.
- **Metrics** (`metrics.py`): Sharpe, Sortino, max drawdown, Calmar, CAGR, volatility, win rate, profit factor, turnover. Annualised on 252 trading days.
- **Walk-forward validation** (`walkforward.py`): 252-day train / 63-day test folds, reported out-of-sample only.
- **Data** (`data.py`): yfinance with a local CSV cache and auto-adjusted prices (dividends/splits), plus reproducible synthetic GBM prices and a synthetic cointegrated pair for tests and offline runs.

## Quick start — run in 2 minutes

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .

# SMA crossover on SPY, 2020 to present (real data via yfinance)
python run_example.py

# Pairs trading on a synthetic cointegrated pair (offline-safe)
python -m quant_backtester.cli --strategy pairs --use-synthetic --tickers AAPL MSFT --output-dir results/pairs

# Your own run
python -m quant_backtester.cli --strategy sma --tickers QQQ --start 2018-01-01 --short-window 20 --long-window 50
```

If yfinance is unreachable, the CLI falls back to synthetic data and prints that it did so. It never silently pretends synthetic data is real market data.

## Example results — actual observed runs, 23 Nov 2026

All runs below used $100,000 starting capital, 5 bps commission + 2 bps slippage = 7 bps per trade, and no risk-free rate. Numbers are from the runs in this repo, not invented.

### SMA crossover (20/50) on SPY — real data

`python run_example.py` — yfinance, SPY, 2020-01-02 to 2026-11-23, 1,733 bars.

| Metric | Value |
|---|---|
| Total return | 70.17% |
| CAGR | 8.04% |
| Volatility (annualised) | 12.17% |
| Sharpe ratio | 0.70 |
| Sortino ratio | 0.95 |
| Max drawdown | -29.38% |
| Calmar ratio | 0.27 |
| Win rate | 54.27% |
| Profit factor | 1.15 |
| Final equity | $170,172.71 |
| Trades | 28 |

Walk-forward (252-day train, 63-day test, 23 folds): mean out-of-sample Sharpe **0.59**, mean out-of-sample return **1.42%** per 63-day fold.

Honest comparison: buy-and-hold SPY over the same period returned **144.35%** (CAGR 13.87%, Sharpe 0.75, max drawdown -33.72%, final equity $244,346.85). The SMA strategy underperformed buy-and-hold on return in this strong bull market, but with lower volatility (12.17% vs 20.14%) and a smaller max drawdown. That is the point of reporting both: a backtest without a benchmark is marketing, not research.

### Pairs trading — synthetic cointegrated pair

`python -m quant_backtester.cli --strategy pairs --use-synthetic --tickers AAPL MSFT` — synthetic pair by construction (seed 7, true beta 1.5, 756 bars, 2023-01-02 to 2025-11-24). Cointegration test p-value **0.0000**, full-sample hedge ratio **1.498**.

| Metric | Value |
|---|---|
| Total return | 16.97% |
| CAGR | 5.37% |
| Volatility (annualised) | 4.39% |
| Sharpe ratio | 1.21 |
| Sortino ratio | 2.00 |
| Max drawdown | -2.14% |
| Calmar ratio | 2.51 |
| Win rate | 55.20% |
| Profit factor | 1.66 |
| Final equity | $116,974.28 |
| Trades | 161 |

Walk-forward (8 folds): mean out-of-sample Sharpe **1.70**, mean out-of-sample return **1.68%** per fold.

Note: the pairs example is synthetic on purpose — it is a cointegrated pair by construction, so it tests whether the strategy can recover a known spread. Do not read it as a claim about real AAPL/MSFT. The cointegration test in `strategies.py` is there precisely so you check a real pair before trading it.

Outputs are saved to `results/`: `equity_curve.csv`, `equity.png`, and `walk_forward.csv` (and `results/pairs/` for the pairs run). Those folders are gitignored; regenerate them with the commands above.

## No-lookahead rule

This is the one rule the engine enforces everywhere:

> A signal computed from prices up to and including close `t` may only change the position from the next bar onwards.

In code: `positions = signals.shift(1)`. The return on day `t` is earned by the position decided at `t - 1`. There is a dedicated test for this: a signal that turns long on the day of a +10% jump captures none of it, while a signal from the day before captures all of it.

The pairs hedge ratio is likewise estimated on a trailing window only, never on the full sample. The full-sample ratio is reported by `test_cointegration` for diagnostics only.

## Costs and assumptions

- Default: 5 bps commission + 2 bps slippage = 7 bps per unit of turnover. A flip from long to short (turnover 2) costs 14 bps.
- Positions are fractions of capital (1 = fully long, 0 = flat, -1 = fully short). No leverage, no margin, no borrowing costs modelled.
- Prices are adjusted closes from yfinance (`auto_adjust=True`), so dividends and splits are already reflected.
- No intraday data, no order-book or market-impact model. Slippage is a flat bps assumption, not a microstructure model.

## Walk-forward validation

In-sample optimisation overfits. `run_walk_forward` fits nothing on the test window: it generates signals on the train+test window, then reports metrics on the test window only, and rolls forward. The SMA walk-forward above shows the strategy is not consistently positive out-of-sample — 9 of 23 folds were negative — which is exactly the kind of result a single in-sample backtest would hide.

## Tests

```bash
pytest -q
```

26 tests, all passing (Nov 2026). Coverage includes: no-lookahead (jump-day test), costs reduce returns, event-loop vs vectorized agreement, Sharpe/Sortino/max-drawdown/CAGR hand-checked math, SMA signals, pairs cointegration on a synthetic cointegrated series and non-cointegration on independent walks, synthetic data reproducibility, walk-forward splits and out-of-sample trading, and a full SMA pipeline run.

CI runs `pytest` on Python 3.11 and 3.12 on every push (`.github/workflows/ci.yml`).

## File tree

```
quant-backtester/
├── pyproject.toml
├── requirements.txt
├── LICENSE (MIT)
├── README.md
├── run_example.py            # SMA on SPY, end to end
├── src/quant_backtester/
│   ├── __init__.py
│   ├── data.py               # yfinance + cache, synthetic GBM / cointegrated pair
│   ├── engine.py             # vectorized + event backtester, costs, trade log
│   ├── strategies.py         # SMA crossover, pairs trading, cointegration test
│   ├── metrics.py            # Sharpe, Sortino, drawdown, Calmar, CAGR, ...
│   ├── walkforward.py        # walk-forward splits and OOS evaluation
│   └── cli.py                # CLI: run a backtest, print metrics, save chart
├── tests/
│   ├── test_engine.py
│   ├── test_metrics.py
│   ├── test_strategies.py
│   └── test_data_walkforward.py
└── .github/workflows/ci.yml
```

## Limitations

- This is a research backtester, not a live-trading system. There is no broker connection, no live data feed, and no order execution.
- Past performance does not predict future returns. The SMA example above underperformed buy-and-hold.
- Survivorship bias: backtesting only on tickers that still exist (e.g. SPY) overstates what was knowable at the time. The engine does not correct for this.
- Flat slippage and fixed commission are approximations. Real costs depend on liquidity, size, and venue.
- The pairs example uses synthetic data by design; real pairs need a cointegration check, a regime check, and a borrow-cost model before they mean anything.
