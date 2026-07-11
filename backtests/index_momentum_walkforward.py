"""Walk-forward validation of the BANDED time-series momentum fix.

WHAT THIS DOES
--------------
Tests whether the buffer-band (hysteresis) fix added to strategies/ts_momentum.py
actually helps -- and isn't just overfit to history -- using a ROLLING WALK-FORWARD:
re-tune parameters on a trailing TRAIN window, trade the NEXT unseen TEST window
with those locked params, roll forward, repeat. The strategy never trades a bar
whose parameters were chosen with knowledge of that bar. The stitched out-of-sample
chunks form a "live-like" equity curve.

It runs the walk-forward TWICE per index:
  - BASELINE : band forced to 0.0 (the original hard-threshold strategy).
  - BANDED   : band is a tunable parameter ({0, 1%, 2%, 3%}).
and prints them side by side so the whipsaw cut and bull-drag change are visible.

Honours CLAUDE.md: fees + slippage always applied; walk-forward (not in-sample)
is the basis for any claim; buy & hold shown for context; BACKTEST/PAPER ONLY.

Mirrors the proven harness in backtests/trend_indices.py.

Run:
    .venv\\Scripts\\python.exe backtests\\index_momentum_walkforward.py
"""

from __future__ import annotations

import itertools
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import vectorbt as vbt

from strategies.ts_momentum import generate_signals
from utils.costs import FEES, SLIPPAGE, INIT_CASH
from utils.stock_loader import load_stock

# ---- Config -----------------------------------------------------------------
SYMBOLS = ["SPY", "QQQ", "IWM"]
START = "1993-01-01"

# Rolling walk-forward window sizes (trading days). ~2y train, ~1y test, step = test.
TRAIN_BARS = 504
TEST_BARS = 252

GRID_LOOKBACK = [126, 189, 252]
GRID_TREND = [100, 150, 200]
GRID_BAND = [0.0, 0.01, 0.02, 0.03]


def _backtest(close, params) -> vbt.Portfolio:
    entries, exits = generate_signals(close, **params)
    return vbt.Portfolio.from_signals(
        close, entries, exits,
        fees=FEES, slippage=SLIPPAGE, init_cash=INIT_CASH, freq="1D",
    )


def _combos(bands):
    for lb, tr, bd in itertools.product(GRID_LOOKBACK, GRID_TREND, bands):
        yield dict(lookback=lb, trend=tr, band=bd)


def _tune(close, bands):
    """Pick params maximising total return on the GIVEN (train) window only."""
    best_params, best_ret = None, -np.inf
    for params in _combos(bands):
        ret = _backtest(close, params).total_return()
        if ret > best_ret:
            best_ret, best_params = ret, params
    return best_params


def _walk_forward(close, bands):
    """Re-tune on each trailing TRAIN_BARS window, trade the next TEST_BARS window.

    Returns (stitched_oos_daily_returns, n_segments, n_entries) where params for
    each test chunk were chosen using ONLY data strictly before that chunk.
    """
    seg_returns = []
    n_segments = 0
    n_entries = 0
    lead = max(GRID_TREND)
    start = 0
    while start + TRAIN_BARS + TEST_BARS <= len(close):
        train = close.iloc[start : start + TRAIN_BARS]
        test = close.iloc[start + TRAIN_BARS : start + TRAIN_BARS + TEST_BARS]
        params = _tune(train, bands)
        # Warm-up lead-in so indicators are warm at the test window's first bar.
        ctx = close.iloc[max(0, start + TRAIN_BARS - lead) : start + TRAIN_BARS + TEST_BARS]
        pf = _backtest(ctx, params)
        seg_returns.append(pf.returns().loc[test.index[0] : test.index[-1]])
        # Count entries that actually fall inside the traded test window.
        e, _ = generate_signals(ctx, **params)
        n_entries += int(e.loc[test.index[0] : test.index[-1]].sum())
        start += TEST_BARS
        n_segments += 1
    if not seg_returns:
        return None, 0, 0
    return pd.concat(seg_returns), n_segments, n_entries


def _summary(wf_ret):
    """Compound stitched OOS returns into total/annualized/sharpe/maxDD."""
    eq = (1 + wf_ret).cumprod()
    total = eq.iloc[-1] - 1.0
    years = len(wf_ret) / 252.0
    ann = (1 + total) ** (1 / years) - 1.0 if years > 0 and total > -1 else float("nan")
    sharpe = (wf_ret.mean() / wf_ret.std() * np.sqrt(252)) if wf_ret.std() > 0 else float("nan")
    dd = (eq / eq.cummax() - 1.0).min()
    return total, ann, sharpe, dd


def _bh_summary(close, wf_index):
    """Buy & hold over the same stitched walk-forward span, with costs."""
    span = close.loc[wf_index[0] : wf_index[-1]]
    bh = vbt.Portfolio.from_holding(span, fees=FEES, slippage=SLIPPAGE,
                                    init_cash=INIT_CASH, freq="1D")
    return bh.total_return(), bh.annualized_return(), bh.sharpe_ratio(), bh.max_drawdown()


def main():
    print("=" * 96)
    print("WALK-FORWARD: BANDED time-series momentum vs BASELINE (band=0)  --  BACKTEST/PAPER ONLY")
    print(f"train {TRAIN_BARS}d / test {TEST_BARS}d / step {TEST_BARS}d | "
          f"fees={FEES:.3%} slippage={SLIPPAGE:.3%} | init ${INIT_CASH:,.0f}")
    print("Params for each test chunk are tuned ONLY on the prior train window (no look-ahead).")
    print("=" * 96)

    rows = []
    for sym in SYMBOLS:
        try:
            close = load_stock(sym, start=START)["close"].dropna()
        except Exception as e:
            print(f"\n[skip] {sym}: {e}")
            continue

        base_ret, base_seg, base_n = _walk_forward(close, bands=[0.0])
        band_ret, band_seg, band_n = _walk_forward(close, bands=GRID_BAND)
        if base_ret is None or band_ret is None:
            print(f"\n{sym}: not enough data for walk-forward")
            continue

        bt, ba, bs, bd = _summary(base_ret)
        nt, na, ns, nd = _summary(band_ret)
        bh = _bh_summary(close, band_ret.index)

        print(f"\n######## {sym} ########  ({close.index[0].date()} -> {close.index[-1].date()}, "
              f"{band_seg} WF segments)")
        print(f"  {'variant':<16}{'OOS total':>11}{'ann':>9}{'sharpe':>8}{'maxDD':>9}{'WF entries':>12}")
        print(f"  {'-'*64}")
        print(f"  {'BASELINE band=0':<16}{bt:>+11.1%}{ba:>+9.2%}{bs:>8.2f}{bd:>+9.1%}{base_n:>12d}")
        print(f"  {'BANDED (tuned)':<16}{nt:>+11.1%}{na:>+9.2%}{ns:>8.2f}{nd:>+9.1%}{band_n:>12d}")
        print(f"  {'buy & hold':<16}{bh[0]:>+11.1%}{bh[1]:>+9.2%}{bh[2]:>8.2f}{bh[3]:>+9.1%}{'-':>12}")
        rows.append((sym, bt, bs, bd, base_n, nt, ns, nd, band_n, bh))

    # ---- Verdict -----------------------------------------------------------
    print("\n" + "=" * 96)
    print("VERDICT (per CLAUDE.md -- walk-forward, not in-sample)")
    print("=" * 96)
    for sym, bt, bs, bd, bn, nt, ns, nd, nn, bh in rows:
        whip = f"trades {bn}->{nn}"
        ret_chg = f"ret {bt:+.1%}->{nt:+.1%}"
        sh_chg = f"sharpe {bs:.2f}->{ns:.2f}"
        dd_chg = f"maxDD {bd:+.1%}->{nd:+.1%}"
        beat = "beats B&H" if nt > bh[0] else "trails B&H"
        print(f"  {sym}: {whip} | {ret_chg} | {sh_chg} | {dd_chg} | banded {beat} ({bh[0]:+.1%})")
        if ns > 3:
            print(f"      * {sym} sharpe {ns:.2f} > 3 on daily data -- suspiciously high, distrust.")
    print("\n  Reading: the band should CUT trade count (less whipsaw) and hold-or-improve")
    print("  sharpe/maxDD. Trend-following still typically trails buy & hold on RAW return in")
    print("  a long bull market -- judge on risk-adjusted terms. maxDD staying low confirms the")
    print("  bear-market protection is intact. One rule, daily bars, BACKTEST/PAPER ONLY.")


if __name__ == "__main__":
    main()
