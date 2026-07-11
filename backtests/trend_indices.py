"""Trend-momentum on stock-index ETFs -- single-split AND rolling walk-forward.

WHY INDICES
-----------
Trend/momentum has far stronger published evidence on broad equity indices than on a
2-year crypto sample, and we have ~16 years of daily ETF data cached (2010-2026). More
history lets us do the honest thing properly: a ROLLING walk-forward, where we re-tune
on a trailing window and then trade the NEXT unseen window -- repeatedly. The strategy
never trades a bar whose parameters were chosen with knowledge of that bar.

TWO STUDIES (both honour CLAUDE.md: fees+slippage always, IS/OOS labelled separately):

  STUDY A -- single split (65/35), one tune, one OOS judge. Comparable to the crypto run.
  STUDY B -- rolling walk-forward: tune on a `TRAIN_BARS` window, trade the next
             `TEST_BARS` window with those locked params, roll forward, repeat. The
             concatenation of all the out-of-sample chunks is the walk-forward equity
             curve -- the closest thing to "how it would have done live."

$1,500 capital. Long-only, full-equity compounding ("aggressive" without leverage and
without any live-order code). BACKTEST / PAPER ONLY.

Run:
    .venv\\Scripts\\python.exe backtests\\trend_indices.py
"""

from __future__ import annotations

import itertools
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import vectorbt as vbt

from strategies.trend_momentum import generate_signals
from utils.costs import FEES, SLIPPAGE
from utils.stock_loader import load_stock

# ---- Config -----------------------------------------------------------------
INIT_CASH = 1500
IS_FRACTION = 0.65
ASSETS = ["SPY", "QQQ", "IWM", "GLD", "TLT", "XLE"]

# Rolling walk-forward window sizes (trading days). ~2y train, ~1y test, step = test.
TRAIN_BARS = 504
TEST_BARS = 252

GRID_BREAKOUT = [20, 30, 50]
GRID_EXIT_N = [10, 20]
GRID_TREND = [100, 150, 200]


def _backtest(close, params) -> vbt.Portfolio:
    entries, exits = generate_signals(close, **params)
    return vbt.Portfolio.from_signals(
        close, entries, exits,
        fees=FEES, slippage=SLIPPAGE, init_cash=INIT_CASH, freq="1D",
    )


def _valid_combos():
    for b, e, t in itertools.product(GRID_BREAKOUT, GRID_EXIT_N, GRID_TREND):
        if e < b and t >= b:
            yield dict(breakout=b, exit_n=e, trend=t)


def _tune(close):
    """Pick params maximising total return on the GIVEN window only."""
    best_params, best_ret = None, -np.inf
    for params in _valid_combos():
        ret = _backtest(close, params).total_return()
        if ret > best_ret:
            best_ret, best_params = ret, params
    return best_params


def _stats(label, pf, close):
    n = pf.trades.count()
    wr = pf.trades.win_rate() if n > 0 else float("nan")
    final = INIT_CASH * (1 + pf.total_return())
    print(f"  --- {label} ---")
    print(f"    period           : {close.index[0].date()} -> {close.index[-1].date()}  ({len(close)} bars)")
    print(f"    total return     : {pf.total_return():>8.2%}   (${INIT_CASH:,.0f} -> ${final:,.0f})")
    print(f"    annualized return: {pf.annualized_return():>8.2%}")
    print(f"    sharpe ratio     : {pf.sharpe_ratio():>8.2f}")
    print(f"    max drawdown     : {pf.max_drawdown():>8.2%}")
    print(f"    win rate         : {wr:>8.2%}" if n > 0 else "    win rate         :      n/a")
    print(f"    # trades         : {n:>8d}")
    return pf.total_return(), pf.sharpe_ratio(), n


def _bh(close):
    return close.iloc[-1] / close.iloc[0] - 1.0


# --------------------------------------------------------------------------- #
# STUDY B: rolling walk-forward                                               #
# --------------------------------------------------------------------------- #
def _walk_forward(close):
    """Re-tune on each trailing TRAIN_BARS window, trade the next TEST_BARS window.

    Returns the stitched out-of-sample daily-return series (params for each test chunk
    were chosen using ONLY data strictly before that chunk -- no look-ahead).
    """
    seg_returns = []
    start = 0
    n_segments = 0
    while start + TRAIN_BARS + TEST_BARS <= len(close):
        train = close.iloc[start : start + TRAIN_BARS]
        test = close.iloc[start + TRAIN_BARS : start + TRAIN_BARS + TEST_BARS]
        params = _tune(train)
        # Feed a small lead-in so indicators are warm at the test window's first bar,
        # then keep only the test-window returns.
        lead = max(GRID_TREND)
        ctx = close.iloc[max(0, start + TRAIN_BARS - lead) : start + TRAIN_BARS + TEST_BARS]
        pf = _backtest(ctx, params)
        seg_returns.append(pf.returns().loc[test.index[0] : test.index[-1]])
        start += TEST_BARS
        n_segments += 1
    if not seg_returns:
        return None, 0
    return pd.concat(seg_returns), n_segments


def _wf_summary(sym, wf_ret):
    """Compound the stitched OOS returns into total/annualized/sharpe/maxDD."""
    eq = (1 + wf_ret).cumprod()
    total = eq.iloc[-1] - 1.0
    years = len(wf_ret) / 252.0
    ann = (1 + total) ** (1 / years) - 1.0 if years > 0 and total > -1 else float("nan")
    sharpe = (wf_ret.mean() / wf_ret.std() * np.sqrt(252)) if wf_ret.std() > 0 else float("nan")
    dd = (eq / eq.cummax() - 1.0).min()
    final = INIT_CASH * (1 + total)
    print(f"  {sym:<6} WF total {total:>+8.2%} (${INIT_CASH:,.0f}->${final:,.0f})  "
          f"ann {ann:>+7.2%}  sharpe {sharpe:>5.2f}  maxDD {dd:>7.2%}  bars {len(wf_ret)}")
    return total, sharpe


def main():
    print("=" * 78)
    print("TREND-MOMENTUM ON INDEX ETFs  --  BACKTEST / PAPER ONLY (see CLAUDE.md)")
    print(f"Capital ${INIT_CASH:,.0f} | fees={FEES:.3%} slippage={SLIPPAGE:.3%} | long-only, full-equity")
    print("=" * 78)

    loaded = {}
    for sym in ASSETS:
        try:
            loaded[sym] = load_stock(symbol=sym).dropna()
        except Exception as e:
            print(f"  [skip] {sym}: {e}")

    # ---------------- STUDY A: single 65/35 split ---------------------------
    print("\n" + "#" * 78)
    print("# STUDY A -- single split (tune on first 65%, judge ONCE on last 35%)")
    print("#" * 78)
    a_results = []
    for sym, df in loaded.items():
        close = df["close"]
        split = int(len(close) * IS_FRACTION)
        is_close, oos_close = close.iloc[:split], close.iloc[split:]
        params = _tune(is_close)
        print(f"\n######## {sym} ########  best IN-SAMPLE params: {params}")
        print("=== IN-SAMPLE (train) ===")
        is_ret, _, _ = _stats("trend_momentum", _backtest(is_close, params), is_close)
        print(f"    buy & hold      : {_bh(is_close):>8.2%}")
        print("=== OUT-OF-SAMPLE (test) ===")
        oos_ret, oos_sh, oos_n = _stats("trend_momentum", _backtest(oos_close, params), oos_close)
        print(f"    buy & hold      : {_bh(oos_close):>8.2%}")
        print(f"  IS->OOS gap: {is_ret - oos_ret:>+.2%}   |  strat OOS {oos_ret:+.2%} vs B&H {_bh(oos_close):+.2%}")
        a_results.append((sym, oos_ret, oos_sh, oos_n, _bh(oos_close)))

    # ---------------- STUDY B: rolling walk-forward -------------------------
    print("\n" + "#" * 78)
    print(f"# STUDY B -- rolling walk-forward (train {TRAIN_BARS}d / test {TEST_BARS}d, step {TEST_BARS}d)")
    print("#   Each test chunk uses params tuned ONLY on data before it. Stitched = 'live-like'.")
    print("#" * 78)
    b_results = []
    for sym, df in loaded.items():
        wf_ret, n_seg = _walk_forward(df["close"])
        if wf_ret is None or len(wf_ret) == 0:
            print(f"  {sym:<6} not enough data for walk-forward")
            continue
        total, sharpe = _wf_summary(sym, wf_ret)
        b_results.append((sym, total, sharpe, n_seg))

    # ---------------- Honest verdict ----------------------------------------
    print("\n" + "=" * 78)
    print("VERDICT (per CLAUDE.md)")
    print("=" * 78)
    a_wins = sum(1 for _, r, *_ in a_results if r > 0)
    a_beat = sum(1 for _, r, _, _, bh in a_results if r > bh)
    b_wins = sum(1 for _, t, *_ in b_results if t > 0)
    print(f"  STUDY A single-split : profitable OOS on {a_wins}/{len(a_results)} ETFs; "
          f"beat buy&hold on {a_beat}/{len(a_results)}.")
    print(f"  STUDY B walk-forward : positive stitched OOS on {b_wins}/{len(b_results)} ETFs.")
    print()
    for sym, t, sh, n_seg in b_results:
        if sh > 3:
            print(f"  * {sym}: WF sharpe {sh:.2f} > 3 on daily data -- suspiciously high, distrust.")
    print("  Trend-following typically UNDERPERFORMS buy & hold on raw return in a long bull")
    print("  market but cuts drawdown -- judge it on risk-adjusted terms (sharpe, maxDD), not")
    print("  just total return. Even a positive walk-forward here is NOT a license to size $1,500")
    print("  aggressively: it's one rule, one asset class, daily bars, no regime-change stress")
    print("  test. BACKTEST/PAPER ONLY -- no live-order code is wired up (CLAUDE.md).")


if __name__ == "__main__":
    main()
