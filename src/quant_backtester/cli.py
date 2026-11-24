"""Command-line interface.

Example:
    python -m quant_backtester.cli --strategy sma --tickers SPY --start 2020-01-01
    python run_example.py --use-synthetic

If yfinance is unreachable the CLI falls back to synthetic data and
says so explicitly, so an offline run never silently pretends to be
real market data.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from .data import generate_cointegrated_pair, generate_synthetic_prices, load_prices
from .engine import BacktestConfig, run_backtest
from .metrics import compute_metrics
from .strategies import PairsTradingStrategy, SMACrossoverStrategy
from .walkforward import run_walk_forward


def _print_metrics(title: str, metrics: dict[str, float]) -> None:
    print(f"\n{title}")
    print("-" * len(title))
    for key in [
        "total_return",
        "cagr",
        "volatility",
        "sharpe_ratio",
        "sortino_ratio",
        "max_drawdown",
        "calmar_ratio",
        "win_rate",
        "profit_factor",
    ]:
        value = metrics[key]
        if key in ("total_return", "cagr", "volatility", "max_drawdown", "win_rate"):
            print(f"  {key:18s} {value:8.2%}")
        else:
            print(f"  {key:18s} {value:8.2f}")
    print(f"  {'final_equity':18s} ${metrics['final_equity']:,.2f}")


def _save_chart(equity: pd.Series, output_dir: Path, name: str = "equity.png") -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / name
    fig, ax = plt.subplots(figsize=(10, 5))
    equity.plot(ax=ax)
    ax.set_title("Equity curve")
    ax.set_ylabel("Equity ($)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Quant backtester")
    parser.add_argument("--strategy", choices=["sma", "pairs"], default="sma")
    parser.add_argument("--tickers", nargs="+", default=["SPY"])
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default=None)
    parser.add_argument("--initial-capital", type=float, default=100_000.0)
    parser.add_argument("--commission-bps", type=float, default=5.0)
    parser.add_argument("--slippage-bps", type=float, default=2.0)
    parser.add_argument("--short-window", type=int, default=20)
    parser.add_argument("--long-window", type=int, default=50)
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--use-synthetic", action="store_true", help="Use synthetic data instead of yfinance")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    config = BacktestConfig(
        initial_capital=args.initial_capital,
        commission_bps=args.commission_bps,
        slippage_bps=args.slippage_bps,
    )
    output_dir = Path(args.output_dir)
    used_synthetic = args.use_synthetic

    if args.strategy == "pairs":
        tickers = args.tickers[:2] if len(args.tickers) >= 2 else ["AAPL", "MSFT"]
        if used_synthetic:
            prices = generate_cointegrated_pair(n=756, seed=args.seed)
            prices.columns = tickers
        else:
            try:
                prices = load_prices(tickers, start=args.start, end=args.end)
            except Exception as exc:  # noqa: BLE001 - fall back, but say so
                print(f"yfinance failed ({exc}); falling back to synthetic cointegrated data.")
                used_synthetic = True
                prices = generate_cointegrated_pair(n=756, seed=args.seed)
                prices.columns = tickers
        strategy = PairsTradingStrategy()
        coint = PairsTradingStrategy.test_cointegration(prices)
        print(f"Cointegration test: p-value={coint['coint_pvalue']:.4f} "
              f"(cointegrated={coint['is_cointegrated']}), "
              f"hedge ratio={coint['hedge_ratio_full_sample']:.3f}")
    else:
        tickers = args.tickers
        if used_synthetic:
            prices = generate_synthetic_prices(n=756, tickers=tuple(tickers), seed=args.seed)
        else:
            try:
                prices = load_prices(tickers, start=args.start, end=args.end)
            except Exception as exc:  # noqa: BLE001 - fall back, but say so
                print(f"yfinance failed ({exc}); falling back to synthetic data.")
                used_synthetic = True
                prices = generate_synthetic_prices(n=756, tickers=tuple(tickers), seed=args.seed)
        strategy = SMACrossoverStrategy(args.short_window, args.long_window)

    print(f"Data: {'SYNTHETIC (offline fallback)' if used_synthetic else 'yfinance'} "
          f"{prices.index[0].date()} to {prices.index[-1].date()}, {len(prices)} bars, "
          f"tickers={list(prices.columns)}")

    signals = strategy.generate_signals(prices)
    result = run_backtest(prices, signals, config=config)
    metrics = compute_metrics(result.equity_curve, result.returns, trades=result.trades)
    _print_metrics(f"{args.strategy.upper()} backtest — costs {config.total_cost_bps:.0f} bps/trade", metrics)
    print(f"  {'trades':18s} {len(result.trades)}")

    output_dir.mkdir(parents=True, exist_ok=True)
    equity_path = output_dir / "equity_curve.csv"
    result.equity_curve.to_csv(equity_path)
    chart_path = _save_chart(result.equity_curve, output_dir)
    print(f"\nSaved equity curve to {equity_path}")
    print(f"Saved chart to {chart_path}")

    # Walk-forward, out-of-sample only, if there is enough data.
    if len(prices) >= 252 + 63:
        try:
            if args.strategy == "pairs":
                factory = lambda: PairsTradingStrategy()  # noqa: E731
            else:
                factory = lambda: SMACrossoverStrategy(args.short_window, args.long_window)  # noqa: E731
            wf = run_walk_forward(prices, factory, train_size=252, test_size=63, config=config)
            print("\nWalk-forward (out-of-sample, 63-day test windows):")
            print(wf.to_string(index=False))
            print(f"  Mean OOS Sharpe: {wf['sharpe_ratio'].mean():.2f} | "
                  f"Mean OOS return: {wf['total_return'].mean():.2%} | "
                  f"Folds: {len(wf)}")
            wf.to_csv(output_dir / "walk_forward.csv", index=False)
        except Exception as exc:  # noqa: BLE001
            print(f"Walk-forward failed: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
