"""OOS robustness sweep for time-series momentum on indexes.

PURPOSE / METHODOLOGY (read this -- see CLAUDE.md)
--------------------------------------------------
This sweeps the lookback x trend parameter grid on the OUT-OF-SAMPLE window and
prints EVERY cell. It is a ROBUSTNESS DIAGNOSTIC, not an optimizer:

  - We do NOT pick the "best" OOS parameters. Searching OOS for the best combo
    and reporting it would turn the test set into a training set -- textbook
    overfitting, and exactly what the project's standing rules forbid.
  - Instead we ask: is the edge STABLE across the whole grid? If most combos do
    similarly (e.g. all beat or all trail buy-and-hold by a similar margin), the
    effect is real and not a knife-edge. If only one lucky cell works while its
    neighbours collapse, it's noise.

Each cell shows the strategy's out-of-sample total return and Sharpe. A single
buy-and-hold benchmark per index is printed for reference. Fees + slippage are
applied to every cell (never a zero-cost backtest).

Run from the project root with the venv python:
    .venv\\Scripts\\python.exe backtests\\index_momentum_oos_sweep.py
"""

from __future__ import annotations

import os
import sys

# Make the project root importable when run as a script.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import vectorbt as vbt

from strategies.ts_momentum import generate_signals
from utils.costs import FEES, SLIPPAGE, INIT_CASH
from utils.stock_loader import load_stock

# ---- Config -----------------------------------------------------------------
SYMBOLS = ["SPY", "QQQ", "IWM"]
START = "2005-01-01"
IS_FRACTION = 0.70                 # OOS = the last 30% (same split as the main backtest)
LOOKBACKS = [63, 126, 189, 252]    # ~3, 6, 9, 12 trading months
TRENDS = [100, 150, 200, 250]      # long regime-filter SMA windows


def backtest(close, lookback, trend):
    """Run one parameter cell on `close`; return (total_return, sharpe, n_trades)."""
    entries, exits = generate_signals(close, lookback=lookback, trend=trend)
    pf = vbt.Portfolio.from_signals(
        close, entries, exits,
        fees=FEES, slippage=SLIPPAGE, init_cash=INIT_CASH, freq="1D",
    )
    return pf.total_return(), pf.sharpe_ratio(), pf.trades.count()


def sweep_symbol(symbol: str):
    print("\n" + "=" * 78)
    print(f"INDEX: {symbol}   --   OUT-OF-SAMPLE robustness sweep")
    print("=" * 78)

    try:
        df = load_stock(symbol, start=START)
    except Exception as exc:
        print(f"  !! could not load {symbol}: {exc} -- skipping.")
        return

    close = df["close"]
    split = int(len(close) * IS_FRACTION)
    oos = close.iloc[split:]

    bh = vbt.Portfolio.from_holding(oos, init_cash=INIT_CASH, freq="1D")
    print(f"OOS period : {oos.index[0].date()} -> {oos.index[-1].date()}  ({len(oos)} bars)")
    print(f"costs      : fees={FEES:.3%}/trade  slippage={SLIPPAGE:.3%}/trade")
    print(f"BUY & HOLD : total return {bh.total_return():+.2%}   sharpe {bh.sharpe_ratio():.2f}")

    # ---- Total-return grid --------------------------------------------------
    header = "lookback \\ trend |" + "".join(f"{t:>10}" for t in TRENDS)
    print("\n  total return (strategy, net of costs):")
    print("  " + header)
    print("  " + "-" * len(header))
    ret_grid = []
    for lb in LOOKBACKS:
        row = []
        cells = ""
        for tr in TRENDS:
            ret, _, _ = backtest(oos, lb, tr)
            row.append(ret)
            cells += f"{ret:>9.1%} "
        ret_grid.append(row)
        print(f"  {lb:>14} |{cells}")

    # ---- Sharpe grid --------------------------------------------------------
    print("\n  sharpe ratio (strategy):")
    print("  " + header)
    print("  " + "-" * len(header))
    sharpe_grid = []
    for lb in LOOKBACKS:
        row = []
        cells = ""
        for tr in TRENDS:
            _, sh, _ = backtest(oos, lb, tr)
            row.append(sh)
            cells += f"{sh:>9.2f} "
        sharpe_grid.append(row)
        print(f"  {lb:>14} |{cells}")

    # ---- Stability read-out -------------------------------------------------
    rets = np.array(ret_grid)
    shps = np.array(sharpe_grid)
    bh_ret = bh.total_return()
    frac_beat = (rets > bh_ret).mean()
    print(f"\n  STABILITY across {rets.size} parameter combos:")
    print(f"    total return : min {rets.min():+.1%}  median {np.median(rets):+.1%}  max {rets.max():+.1%}")
    print(f"    sharpe       : min {shps.min():.2f}  median {np.median(shps):.2f}  max {shps.max():.2f}")
    print(f"    beat buy&hold return in {frac_beat:.0%} of combos (B&H = {bh_ret:+.1%})")


def main():
    print("OUT-OF-SAMPLE ROBUSTNESS SWEEP -- diagnostic only, NOT an optimizer.")
    print("We show the whole grid to judge stability; we do NOT pick OOS 'winners'.")
    for symbol in SYMBOLS:
        sweep_symbol(symbol)

    print("\n" + "-" * 78)
    print("How to read this (see CLAUDE.md):")
    print("- STABLE grid (most cells similar, similar sign vs buy & hold) => the effect is")
    print("  real and robust to parameter choice.")
    print("- KNIFE-EDGE grid (one great cell, neighbours collapse) => noise / overfit smell.")
    print("- Picking the single best OOS cell would CONTAMINATE the test set; we don't.")
    print("- A real out-of-sample validation = choose params on in-sample, apply once to OOS")
    print("  (walk-forward). This sweep only tells us whether such a choice would be fragile.")


if __name__ == "__main__":
    main()
