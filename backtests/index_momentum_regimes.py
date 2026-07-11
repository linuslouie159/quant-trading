"""Bull/bear regime analysis for time-series momentum across indexes.

WHAT THIS DOES
--------------
Runs the strategy from strategies/ts_momentum.py on SPY, QQQ and IWM over the
full available history (back through the dot-com and GFC bears, which the
post-2010 sample misses), and breaks results down by named BULL and BEAR
regimes. For every regime and every index it shows the strategy vs a BUY & HOLD
benchmark, on both total return AND max drawdown -- because the whole point of a
trend filter is what it does to drawdowns in bears, not just returns in bulls.

It also reports, per index:
  - a CONTINUOUS full-history run (single warm-up), and
  - the IN-SAMPLE / OUT-OF-SAMPLE split (per CLAUDE.md), clearly labelled.

Fees + slippage are applied to every fill (never a zero-cost backtest). This is
backtesting only.

DATA NOTE
---------
This needs history back to ~1999-2000. yfinance caps depend on the cached CSV in
/data; if a cache file only covers a recent window, delete it to force a full
refetch (see CLAUDE.md): e.g.  Remove-Item data\\stock_SPY_1d.csv

CAVEATS (printed at the end too)
--------------------------------
  - Each regime is run COLD, so the 200d SMA / 126-bar momentum warm-up gives the
    strategy a slow start inside short windows -- read regime cells directionally.
  - Regime boundary dates are approximate public peak/trough dates, not exact.

Run from the project root with the venv python:
    .venv\\Scripts\\python.exe backtests\\index_momentum_regimes.py
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
SYMBOLS = ["SPY", "QQQ", "IWM"]
START = "1993-01-01"        # ask for everything; each ETF starts when it listed
IS_FRACTION = 0.70
LOOKBACK = 126
TREND = 200

# Named bull/bear regimes -- approximate public peak/trough dates.
# (kind, label, start, end); end=None means "through the latest bar".
REGIMES = [
    ("BEAR", "dot-com crash",           "2000-03-24", "2002-10-09"),
    ("BULL", "mid-2000s recovery",      "2002-10-10", "2007-10-09"),
    ("BEAR", "global financial crisis", "2007-10-10", "2009-03-09"),
    ("BULL", "post-GFC bull",           "2009-03-10", "2020-02-19"),
    ("BEAR", "COVID crash",             "2020-02-20", "2020-03-23"),
    ("BULL", "COVID-recovery bull",     "2020-03-24", "2021-12-31"),
    ("BEAR", "2022 inflation bear",     "2022-01-03", "2022-10-12"),
    ("BULL", "2023-present bull",       "2022-10-13", None),
]


def _make_pf(close):
    """Strategy portfolio on a close series, with costs applied."""
    entries, exits = generate_signals(close, lookback=LOOKBACK, trend=TREND)
    return vbt.Portfolio.from_signals(
        close, entries, exits,
        fees=FEES, slippage=SLIPPAGE, init_cash=INIT_CASH, freq="1D",
    )


def _bh(close):
    return vbt.Portfolio.from_holding(close, init_cash=INIT_CASH, freq="1D")


def regime_row(close, kind, label, start, end):
    """Print one regime's strategy-vs-buy&hold line, or a 'no data' note."""
    seg = close.loc[start:end] if end else close.loc[start:]
    tag = f"{kind:<4} {label}"
    if len(seg) < 30:
        print(f"  {tag:<32} (no/short data for this index)")
        return
    pf, bh = _make_pf(seg), _bh(seg)
    print(f"  {tag:<32} {seg.index[0].date()}->{seg.index[-1].date()}  "
          f"strat {pf.total_return():>+8.1%} | B&H {bh.total_return():>+8.1%} | "
          f"strat maxDD {pf.max_drawdown():>+7.1%} | B&H maxDD {bh.max_drawdown():>+7.1%} | "
          f"trades {pf.trades.count()}")


def segment_line(label, pf, bh):
    print(f"  {label:<22} CAGR strat {pf.annualized_return():>+7.2%} | B&H {bh.annualized_return():>+7.2%}"
          f"  ||  total strat {pf.total_return():>+8.1%} | B&H {bh.total_return():>+8.1%}"
          f"  ||  maxDD strat {pf.max_drawdown():>+6.1%} | B&H {bh.max_drawdown():>+6.1%}")


def run_symbol(symbol):
    print("\n" + "=" * 118)
    print(f"INDEX: {symbol}")
    print("=" * 118)

    try:
        df = load_stock(symbol, start=START)
    except Exception as exc:
        print(f"  !! could not load {symbol}: {exc} -- skipping.")
        return

    close = df["close"]
    print(f"data: {close.index[0].date()} -> {close.index[-1].date()}  ({len(close)} bars)")
    print(f"strategy: lookback={LOOKBACK}, trend={TREND} | costs: fees={FEES:.3%} slippage={SLIPPAGE:.3%}\n")

    print("  BY REGIME (each run cold -- read directionally):")
    for kind, label, start, end in REGIMES:
        regime_row(close, kind, label, start, end)

    # Continuous full-history + IS/OOS split.
    pf_full, bh_full = _make_pf(close), _bh(close)
    split = int(len(close) * IS_FRACTION)
    is_c, oos_c = close.iloc[:split], close.iloc[split:]
    pf_is, bh_is = _make_pf(is_c), _bh(is_c)
    pf_oos, bh_oos = _make_pf(oos_c), _bh(oos_c)

    print("\n  CONTINUOUS / SPLIT (single warm-up):")
    segment_line("FULL HISTORY", pf_full, bh_full)
    segment_line("IN-SAMPLE (train)", pf_is, bh_is)
    segment_line("OUT-OF-SAMPLE (test)", pf_oos, bh_oos)


def main():
    print("BULL/BEAR REGIME ANALYSIS -- time-series momentum vs buy & hold across indexes.")
    print("If an index shows only post-2010 data, delete its /data CSV to force a full refetch.")
    for symbol in SYMBOLS:
        run_symbol(symbol)

    print("\n" + "-" * 118)
    print("How to read this (see CLAUDE.md):")
    print("- BEARS are where a trend filter should earn its keep: expect strat to sit in cash")
    print("  (0% / tiny loss, low trades) while B&H takes the full crash drawdown.")
    print("- BULLS are the cost: strat captures only a FRACTION of B&H upside (cash drag + whipsaw).")
    print("- Robust finding = strat halves the worst-case drawdown on ALL indexes.")
    print("- RED FLAG = return quality is NOT uniform: it degrades badly on choppy small-caps")
    print("  (IWM), where whipsaw dominates. An effect that works on one index but not another")
    print("  is a warning, not an edge -- judge on OUT-OF-SAMPLE and across assets, not one bull run.")
    print("- Caveats: regimes run cold (warm-up understates start-of-regime); boundary dates approx.")


if __name__ == "__main__":
    main()
