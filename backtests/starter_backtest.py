"""Starter backtest: end-to-end pipeline on BTC/USDT daily.

Demonstrates the full loop and BAKES IN the project guardrails (see CLAUDE.md):
  - Realistic fees + slippage are ALWAYS applied (never a zero-cost backtest).
  - Data is split into in-sample (train) and out-of-sample (test), reported
    SEPARATELY and clearly labelled.
  - The strategy is an explicit PLACEHOLDER, not a validated edge.

Run from the project root with the venv python:
    .venv\\Scripts\\python.exe backtests\\starter_backtest.py
"""

from __future__ import annotations

import os
import sys

# Make the project root importable when run as a script.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import vectorbt as vbt

from strategies.ma_crossover import generate_signals
from utils.costs import FEES, SLIPPAGE, INIT_CASH
from utils.crypto_loader import load_crypto

# ---- Config -----------------------------------------------------------------
SYMBOL = "BTC/USDT"
TIMEFRAME = "1d"
LIMIT = 1000          # ~2.7 years of daily candles
IS_FRACTION = 0.70    # first 70% = in-sample, last 30% = out-of-sample
FAST, SLOW = 20, 50


def run_segment(label: str, close):
    """Backtest one price segment with costs applied; print labelled stats."""
    entries, exits = generate_signals(close, fast=FAST, slow=SLOW)

    pf = vbt.Portfolio.from_signals(
        close,
        entries,
        exits,
        fees=FEES,           # guardrail: real fees, every time
        slippage=SLIPPAGE,   # guardrail: real slippage, every time
        init_cash=INIT_CASH,
        freq="1D",
    )

    print(f"\n=== {label} ===")
    print(f"  period      : {close.index[0].date()} -> {close.index[-1].date()}  ({len(close)} bars)")
    print(f"  costs       : fees={FEES:.3%}/trade  slippage={SLIPPAGE:.3%}/trade")

    trades = pf.trades
    n_trades = trades.count()
    win_rate = trades.win_rate() if n_trades > 0 else float("nan")

    print(f"  total return     : {pf.total_return():.2%}")
    print(f"  annualized return: {pf.annualized_return():.2%}")
    print(f"  sharpe ratio     : {pf.sharpe_ratio():.2f}")
    print(f"  max drawdown     : {pf.max_drawdown():.2%}")
    print(f"  win rate         : {win_rate:.2%}" if n_trades > 0 else "  win rate         : n/a (no trades)")
    print(f"  # trades         : {n_trades}")
    return pf


def main():
    print(f"Loading {SYMBOL} {TIMEFRAME} ({LIMIT} candles) ...")
    df = load_crypto(symbol=SYMBOL, timeframe=TIMEFRAME, limit=LIMIT)
    close = df["close"]

    split = int(len(close) * IS_FRACTION)
    is_close = close.iloc[:split]
    oos_close = close.iloc[split:]

    print(f"\nStrategy: MA crossover (fast={FAST}, slow={SLOW}) -- PLACEHOLDER, not a validated edge.")
    print(f"Split: {len(is_close)} in-sample bars / {len(oos_close)} out-of-sample bars "
          f"({IS_FRACTION:.0%}/{1 - IS_FRACTION:.0%}).")

    run_segment("IN-SAMPLE (train)", is_close)
    run_segment("OUT-OF-SAMPLE (test)", oos_close)

    print("\n" + "-" * 70)
    print("Reminder: OUT-OF-SAMPLE results -- not in-sample -- indicate whether a")
    print("strategy holds up on unseen data. This MA crossover is a placeholder and")
    print("is NOT validated. Do not treat any single-coin, single-period result as")
    print("'profitable'. See CLAUDE.md for the standing rules.")


if __name__ == "__main__":
    main()
