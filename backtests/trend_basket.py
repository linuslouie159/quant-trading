"""Multi-asset trend-following BASKET backtest (stocks/ETFs).

WHY THIS EXISTS (see CLAUDE.md)
------------------------------
A trend strategy that only works on one ticker or one period is overfit. This
script runs the SAME long-only trend rule (strategies/trend_filter.py) across a
basket of liquid ETFs, with realistic fees + slippage, and reports IN-SAMPLE vs
OUT-OF-SAMPLE results SEPARATELY for every asset plus a basket average. It also
prints buy & hold for each asset so we can see whether the trend rule actually
adds anything or is just riding a bull market.

Robustness is judged by BREADTH: does the edge show up out-of-sample across MOST
of the basket, or only on one or two names? One good OOS ticker is noise.

Run from the project root with the venv python:
    .venv\\Scripts\\python.exe backtests\\trend_basket.py
"""

from __future__ import annotations

import os
import sys

# Make the project root importable when run as a script.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import vectorbt as vbt

from strategies.trend_filter import generate_signals
from utils.costs import FEES, SLIPPAGE, INIT_CASH
from utils.stock_loader import load_stock

# ---- Config -----------------------------------------------------------------
# A spread of liquid ETFs across asset classes so the test isn't all one bet:
# broad US equity, tech, small-cap, long bonds, gold, energy.
BASKET = ["SPY", "QQQ", "IWM", "TLT", "GLD", "XLE"]
START = "2010-01-01"
INTERVAL = "1d"
IS_FRACTION = 0.70          # first 70% = in-sample, last 30% = out-of-sample
FAST, SLOW, TREND = 50, 100, 200


def _load_basket() -> pd.DataFrame:
    """Load each symbol's close, align on a shared date index -> one DataFrame."""
    closes = {}
    for sym in BASKET:
        try:
            df = load_stock(symbol=sym, start=START, interval=INTERVAL)
            closes[sym] = df["close"]
        except Exception as e:  # one bad ticker shouldn't sink the whole run
            print(f"  ! skipping {sym}: {e}")
    if not closes:
        raise RuntimeError("No symbols loaded; check network / yfinance.")
    # Inner-join on dates so all columns share the same bars (vbt needs alignment).
    close = pd.DataFrame(closes).dropna(how="any")
    return close


def _signals_frame(close: pd.DataFrame):
    """Build entries/exits DataFrames column-by-column (per-asset signals)."""
    entries = pd.DataFrame(index=close.index, columns=close.columns, dtype=bool)
    exits = pd.DataFrame(index=close.index, columns=close.columns, dtype=bool)
    for sym in close.columns:
        e, x = generate_signals(close[sym], fast=FAST, slow=SLOW, trend=TREND)
        entries[sym], exits[sym] = e, x
    return entries, exits


def run_segment(label: str, close: pd.DataFrame):
    """Backtest the whole basket for one segment; print per-asset + average stats."""
    entries, exits = _signals_frame(close)

    pf = vbt.Portfolio.from_signals(
        close,
        entries,
        exits,
        fees=FEES,           # guardrail: real fees, every time
        slippage=SLIPPAGE,   # guardrail: real slippage, every time
        init_cash=INIT_CASH,
        freq="1D",
    )
    # Buy & hold baseline (same costs) to judge whether the rule adds value.
    bh = vbt.Portfolio.from_holding(close, fees=FEES, slippage=SLIPPAGE,
                                    init_cash=INIT_CASH, freq="1D")

    print(f"\n=== {label} ===")
    print(f"  period : {close.index[0].date()} -> {close.index[-1].date()}  ({len(close)} bars)")
    print(f"  costs  : fees={FEES:.3%}/trade  slippage={SLIPPAGE:.3%}/trade")
    print(f"  {'sym':<5} {'strat_ret':>10} {'b&h_ret':>9} {'sharpe':>7} {'maxDD':>8} {'trades':>7}")

    strat_ret = pf.total_return()
    bh_ret = bh.total_return()
    sharpe = pf.sharpe_ratio()
    maxdd = pf.max_drawdown()
    n_trades = pf.trades.count()

    beat = 0
    for sym in close.columns:
        sr, br = strat_ret[sym], bh_ret[sym]
        beat += sr > br
        print(f"  {sym:<5} {sr:>10.2%} {br:>9.2%} {sharpe[sym]:>7.2f} "
              f"{maxdd[sym]:>8.2%} {int(n_trades[sym]):>7d}")

    n = len(close.columns)
    print(f"  {'-'*48}")
    print(f"  AVG   {strat_ret.mean():>10.2%} {bh_ret.mean():>9.2%} "
          f"{sharpe.mean():>7.2f} {maxdd.mean():>8.2%}")
    print(f"  breadth: strategy beat buy&hold on {beat}/{n} assets")
    return pf


def main():
    print(f"Loading basket {BASKET} ({INTERVAL}, from {START}) ...")
    close = _load_basket()
    print(f"Aligned basket: {len(close)} shared bars across {len(close.columns)} assets "
          f"({list(close.columns)}).")

    split = int(len(close) * IS_FRACTION)
    is_close = close.iloc[:split]
    oos_close = close.iloc[split:]

    print(f"\nStrategy: long-only trend filter (fast={FAST}, slow={SLOW}, trend={TREND}).")
    print(f"Split: {len(is_close)} in-sample bars / {len(oos_close)} out-of-sample bars "
          f"({IS_FRACTION:.0%}/{1 - IS_FRACTION:.0%}).")

    run_segment("IN-SAMPLE (train)", is_close)
    run_segment("OUT-OF-SAMPLE (test)", oos_close)

    print("\n" + "-" * 70)
    print("How to read this: trust OUT-OF-SAMPLE, not in-sample. The strategy is only")
    print("interesting if it beats buy&hold on a MAJORITY of the basket out-of-sample")
    print("(breadth), not just one lucky ticker. A big IS->OOS drop, or an edge that")
    print("lives on a single asset, is the overfitting signature. See CLAUDE.md.")


if __name__ == "__main__":
    main()
