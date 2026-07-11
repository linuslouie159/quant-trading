"""Dual-momentum (cross-sectional + absolute) BASKET backtest -- stocks/ETFs.

WHY THIS EXISTS (see CLAUDE.md)
------------------------------
Cross-sectional momentum is one of the few effects with decades of out-of-sample,
cross-market academic support -- a real candidate "edge" rather than a curve-fit
pattern. This implements the classic medium-risk, long-only form:

  - CROSS-SECTIONAL: each month, rank the basket by trailing `lookback` return and
    hold the TOP `top_n`, equally weighted.
  - ABSOLUTE FILTER: only hold a selected asset if ITS OWN trailing return is also
    positive; otherwise that slice goes to CASH. This is the "dual momentum" safety
    net -- it pulls the whole book toward cash in broad downturns, cutting drawdown.

This is a single TRUE portfolio (cash shared across assets, rebalanced monthly),
not 6 independent bets -- so the Sharpe/return are portfolio-level and honest.

GUARDRAILS (CLAUDE.md):
  - Realistic fees + slippage on every rebalance.
  - Results split into IN-SAMPLE vs OUT-OF-SAMPLE, reported SEPARATELY.
  - Compared against an equal-weight buy&hold of the SAME basket, so we can see
    whether the ranking edge adds value or just rides the market.
  - Monthly rebalance + top_n selection = low turnover, friendly to small accounts.

Run from the project root with the venv python:
    .venv\\Scripts\\python.exe backtests\\dual_momentum.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import vectorbt as vbt

from utils.costs import FEES, SLIPPAGE, INIT_CASH
from utils.stock_loader import load_stock

# ---- Config -----------------------------------------------------------------
BASKET = ["SPY", "QQQ", "IWM", "TLT", "GLD", "XLE"]
START = "2010-01-01"
INTERVAL = "1d"
IS_FRACTION = 0.70
LOOKBACK_DAYS = 126     # ~6 months trailing return for ranking (classic momentum)
TOP_N = 2               # hold the 2 strongest of 6 each month
REBALANCE = "ME"        # month-end rebalance (pandas 'ME' = month end)


def _load_basket() -> pd.DataFrame:
    closes = {}
    for sym in BASKET:
        try:
            closes[sym] = load_stock(symbol=sym, start=START, interval=INTERVAL)["close"]
        except Exception as e:
            print(f"  ! skipping {sym}: {e}")
    if not closes:
        raise RuntimeError("No symbols loaded; check network / yfinance.")
    return pd.DataFrame(closes).dropna(how="any")


def _build_weights_with_cash(close: pd.DataFrame) -> pd.DataFrame:
    """Build a target-weight matrix that emits orders ONLY on rebalance dates.

    On each monthly rebalance date we set an EXPLICIT target-weight row: rank by
    trailing `LOOKBACK_DAYS` return, take the top TOP_N, drop any whose own trailing
    return is <= 0 (absolute filter), equal-weight the survivors -- or an all-zero row
    meaning "go to cash".

    CRITICAL: every NON-rebalance day is left as NaN, NOT forward-filled. With
    `size_type="targetpercent"`, vbt reads NaN as "no order this bar, hold the current
    position", whereas a concrete number is re-targeted EVERY bar -- which would make
    the book trade daily to chase drift and pay fees ~75x more than intended. So the
    book only turns over on the ~monthly rebalance rows; in between it simply holds.
    """
    mom = close.pct_change(LOOKBACK_DAYS)
    weights = pd.DataFrame(np.nan, index=close.index, columns=close.columns)

    rebal_dates = close.resample(REBALANCE).last().index
    rebal_dates = [close.index[close.index.get_indexer([d], method="ffill")[0]]
                   for d in rebal_dates if d >= close.index[0]]

    for d in rebal_dates:
        row = mom.loc[d]
        decided = pd.Series(0.0, index=close.columns)   # default: cash (sell to flat)
        if not row.isna().all():
            ranked = row.dropna().sort_values(ascending=False)
            picks = ranked.head(TOP_N)
            picks = picks[picks > 0.0]
            if len(picks) > 0:
                decided[picks.index] = 1.0 / len(picks)
        weights.loc[d] = decided.values    # concrete target ONLY on rebalance days

    # Non-rebalance days stay NaN -> "hold, do not re-target" (this is the fix).
    return weights


def run_segment(label: str, close: pd.DataFrame):
    weights = _build_weights_with_cash(close)

    pf = vbt.Portfolio.from_orders(
        close,
        size=weights,
        size_type="targetpercent",
        direction="longonly",
        fees=FEES,                 # guardrail: real fees on every rebalance
        slippage=SLIPPAGE,         # guardrail: real slippage
        init_cash=INIT_CASH,
        cash_sharing=True,         # ONE portfolio, cash shared across assets
        group_by=True,             # treat the columns as a single grouped book
        call_seq="auto",           # sell-before-buy so rebalances don't reject on cash
        freq="1D",
    )

    # Equal-weight buy & hold of the SAME basket: target 1/n ONCE on the first bar,
    # NaN thereafter so it buys-and-holds instead of re-targeting (churning) daily.
    n = len(close.columns)
    bh_w = pd.DataFrame(np.nan, index=close.index, columns=close.columns)
    bh_w.iloc[0] = 1.0 / n
    bh = vbt.Portfolio.from_orders(
        close, size=bh_w, size_type="targetpercent", direction="longonly",
        fees=FEES, slippage=SLIPPAGE, init_cash=INIT_CASH,
        cash_sharing=True, group_by=True, call_seq="auto", freq="1D",
    )

    print(f"\n=== {label} ===")
    print(f"  period : {close.index[0].date()} -> {close.index[-1].date()}  ({len(close)} bars)")
    print(f"  costs  : fees={FEES:.3%}/trade  slippage={SLIPPAGE:.3%}/trade")
    print(f"  rule   : top-{TOP_N} of {n} by {LOOKBACK_DAYS}d return, abs-mom cash filter, monthly")
    print(f"  {'':<14}{'strategy':>12}{'eq-wt B&H':>12}")
    print(f"  {'total return':<14}{pf.total_return():>12.2%}{bh.total_return():>12.2%}")
    print(f"  {'ann. return':<14}{pf.annualized_return():>12.2%}{bh.annualized_return():>12.2%}")
    print(f"  {'sharpe':<14}{pf.sharpe_ratio():>12.2f}{bh.sharpe_ratio():>12.2f}")
    print(f"  {'sortino':<14}{pf.sortino_ratio():>12.2f}{bh.sortino_ratio():>12.2f}")
    print(f"  {'max drawdown':<14}{pf.max_drawdown():>12.2%}{bh.max_drawdown():>12.2%}")
    print(f"  {'# trades':<14}{int(pf.trades.count()):>12d}{int(bh.trades.count()):>12d}")
    return pf, bh


def main():
    print(f"Loading basket {BASKET} ({INTERVAL}, from {START}) ...")
    close = _load_basket()
    print(f"Aligned basket: {len(close)} shared bars across {len(close.columns)} assets.")

    split = int(len(close) * IS_FRACTION)
    is_close = close.iloc[:split]
    oos_close = close.iloc[split:]

    print(f"\nStrategy: dual momentum -- top-{TOP_N} cross-sectional + absolute filter, "
          f"{LOOKBACK_DAYS}d lookback, monthly rebalance.")
    print(f"Split: {len(is_close)} in-sample bars / {len(oos_close)} out-of-sample bars "
          f"({IS_FRACTION:.0%}/{1 - IS_FRACTION:.0%}).")

    run_segment("IN-SAMPLE (train)", is_close)
    run_segment("OUT-OF-SAMPLE (test)", oos_close)

    print("\n" + "-" * 70)
    print("How to read this: trust OUT-OF-SAMPLE. A real edge shows a Sharpe that")
    print("survives into the test segment AND beats equal-weight buy&hold there. A")
    print("big IS->OOS Sharpe drop means the lookback/top_n were curve-fit. Sharpe > 3")
    print("on daily data would be a red flag, not a triumph. See CLAUDE.md.")


if __name__ == "__main__":
    main()
