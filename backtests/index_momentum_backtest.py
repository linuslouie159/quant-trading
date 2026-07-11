"""Time-series momentum on stock indexes -- SPY, QQQ, IWM.

Backtests the lean time-series (absolute) momentum rule from
strategies/ts_momentum.py across THREE indexes, and BAKES IN the project
guardrails (see CLAUDE.md):
  - Realistic fees + slippage are ALWAYS applied (never a zero-cost backtest).
  - Each index is split into in-sample (train) and out-of-sample (test),
    reported SEPARATELY and clearly labelled.
  - A buy-and-hold benchmark is printed per segment -- momentum has to BEAT
    just holding the index to be worth anything.
  - Running across multiple indexes is itself a robustness check: an effect that
    works on one index but collapses on the others is a red flag, not an edge.

Run from the project root with the venv python:
    .venv\\Scripts\\python.exe backtests\\index_momentum_backtest.py
"""

from __future__ import annotations

import os
import sys

# Make the project root importable when run as a script.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import vectorbt as vbt

from strategies.ts_momentum import generate_signals
from utils.costs import FEES, SLIPPAGE, INIT_CASH
from utils.stock_loader import load_stock

# ---- Config -----------------------------------------------------------------
SYMBOLS = ["SPY", "QQQ", "IWM"]   # S&P 500, Nasdaq-100, Russell 2000 (ETF proxies)
START = "2005-01-01"
IS_FRACTION = 0.70                # first 70% = in-sample, last 30% = out-of-sample
LOOKBACK = 126                    # ~6 trading months of trailing momentum
TREND = 200                       # long regime-filter SMA window


def run_segment(label: str, close):
    """Backtest one price segment with costs applied; print labelled stats.

    Also prints a buy-and-hold benchmark on the same segment for comparison.
    """
    entries, exits = generate_signals(close, lookback=LOOKBACK, trend=TREND)

    pf = vbt.Portfolio.from_signals(
        close,
        entries,
        exits,
        fees=FEES,           # guardrail: real fees, every time
        slippage=SLIPPAGE,   # guardrail: real slippage, every time
        init_cash=INIT_CASH,
        freq="1D",
    )

    # Benchmark: buy on the first bar and hold the index for the whole segment.
    bh = vbt.Portfolio.from_holding(close, init_cash=INIT_CASH, freq="1D")

    print(f"\n=== {label} ===")
    print(f"  period      : {close.index[0].date()} -> {close.index[-1].date()}  ({len(close)} bars)")
    print(f"  costs       : fees={FEES:.3%}/trade  slippage={SLIPPAGE:.3%}/trade")

    trades = pf.trades
    n_trades = trades.count()
    win_rate = trades.win_rate() if n_trades > 0 else float("nan")

    print(f"  total return     : {pf.total_return():.2%}   (buy & hold: {bh.total_return():.2%})")
    print(f"  annualized return: {pf.annualized_return():.2%}")
    print(f"  sharpe ratio     : {pf.sharpe_ratio():.2f}   (buy & hold: {bh.sharpe_ratio():.2f})")
    print(f"  max drawdown     : {pf.max_drawdown():.2%}   (buy & hold: {bh.max_drawdown():.2%})")
    print(f"  win rate         : {win_rate:.2%}" if n_trades > 0 else "  win rate         : n/a (no trades)")
    print(f"  # trades         : {n_trades}")
    return pf


def run_symbol(symbol: str):
    """Load one index, split IS/OOS, and backtest both segments."""
    print("\n" + "=" * 70)
    print(f"INDEX: {symbol}")
    print("=" * 70)

    try:
        df = load_stock(symbol, start=START)
    except Exception as exc:  # network / data hiccup -> skip, don't crash the run
        print(f"  !! could not load {symbol}: {exc}")
        print(f"  !! skipping {symbol}.")
        return

    close = df["close"]
    split = int(len(close) * IS_FRACTION)
    is_close = close.iloc[:split]
    oos_close = close.iloc[split:]

    print(f"Strategy: time-series momentum (lookback={LOOKBACK}, trend={TREND}) "
          f"-- long-flat, NOT a validated edge.")
    print(f"Split: {len(is_close)} in-sample bars / {len(oos_close)} out-of-sample bars "
          f"({IS_FRACTION:.0%}/{1 - IS_FRACTION:.0%}).")

    run_segment("IN-SAMPLE (train)", is_close)
    run_segment("OUT-OF-SAMPLE (test)", oos_close)


def main():
    print(f"Loading indexes {SYMBOLS} from {START} ...")
    for symbol in SYMBOLS:
        run_symbol(symbol)

    print("\n" + "-" * 70)
    print("Reminder (see CLAUDE.md):")
    print("- OUT-OF-SAMPLE results -- not in-sample -- indicate whether this holds up.")
    print("- The strategy must beat its BUY & HOLD benchmark to be interesting; matching")
    print("  buy-and-hold while taking on cash-drag and trading costs is NOT an edge.")
    print("- Watch for it working on one index but collapsing on the others, a large")
    print("  in-sample / out-of-sample gap, or a Sharpe > 3 on daily data (overfit smell).")
    print("- This is a research hypothesis, not a validated, profitable strategy.")


if __name__ == "__main__":
    main()
