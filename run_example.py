"""Run the example backtests end to end.

SMA crossover on SPY (real data if yfinance is reachable, synthetic
otherwise), then pairs trading on a synthetic cointegrated pair.

Usage:
    python run_example.py
    python run_example.py --use-synthetic
"""

from quant_backtester.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["--strategy", "sma", "--tickers", "SPY", "--start", "2020-01-01"]))
