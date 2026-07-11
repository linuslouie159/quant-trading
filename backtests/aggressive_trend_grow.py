"""Walk-forward backtest for the trend_momentum strategy -- "grow $1500" study.

WHY THIS DESIGN (read me)
-------------------------
The user asked to "grow $1500 aggressively" and to NOT build the strategy on past data
"as it will then only work on info you already know." You cannot backtest without
historical data, but you CAN avoid the trap they're describing -- letting the optimizer
peek at the data it's later judged on. So:

  1. We split each asset into IN-SAMPLE (train, first 65%) and OUT-OF-SAMPLE (test,
     last 35%). The OOS window is the proxy for "data you don't already know."
  2. We grid-search the 3 parameters ONLY on the in-sample window and lock the single
     best combo (by in-sample total return).
  3. We then run those locked parameters ONCE on the out-of-sample window. That number
     is the only one that means anything about unseen data.
  4. We repeat across SEVERAL crypto assets. A real edge survives on more than one coin.

Guardrails baked in per CLAUDE.md:
  - FEES + SLIPPAGE applied to every single backtest (never zero-cost).
  - IN-SAMPLE vs OUT-OF-SAMPLE reported separately and clearly labelled.
  - We do NOT call this strategy "validated"/"profitable" -- the script prints the
    honest caveats and flags overfitting (large IS/OOS gap, single-coin wins, etc.).

INIT_CASH is set to 1500 here to match the user's stated capital. "Aggressive" is
expressed as full-equity, long-only compounding -- NOT leverage and NOT real orders.
This is BACKTEST/PAPER ONLY (see CLAUDE.md: no live trading).

Run from the project root with the venv python:
    .venv\\Scripts\\python.exe backtests\\aggressive_trend_grow.py
"""

from __future__ import annotations

import itertools
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import vectorbt as vbt

from strategies.trend_momentum import generate_signals
from utils.costs import FEES, SLIPPAGE
from utils.crypto_loader import load_crypto

# ---- Config -----------------------------------------------------------------
INIT_CASH = 1500          # the user's stated starting capital
TIMEFRAME = "1d"
LIMIT = 1000              # ~2.7 years of daily candles per asset
IS_FRACTION = 0.65        # first 65% = in-sample (train), last 35% = out-of-sample
ASSETS = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]

# Parameter grid searched ON THE IN-SAMPLE WINDOW ONLY. Kept small on purpose:
# more combos = more chances to fit noise. Invalid combos (exit_n>=breakout or
# trend<breakout) are skipped by the strategy's own validation.
GRID_BREAKOUT = [20, 30, 50]
GRID_EXIT_N = [10, 20]
GRID_TREND = [100, 150, 200]


def _backtest(close, params) -> vbt.Portfolio:
    """Long-only, full-equity, costs always applied."""
    entries, exits = generate_signals(close, **params)
    return vbt.Portfolio.from_signals(
        close,
        entries,
        exits,
        fees=FEES,
        slippage=SLIPPAGE,
        init_cash=INIT_CASH,
        freq="1D",
    )


def _valid_combos():
    for b, e, t in itertools.product(GRID_BREAKOUT, GRID_EXIT_N, GRID_TREND):
        if e < b and t >= b:
            yield dict(breakout=b, exit_n=e, trend=t)


def _tune_in_sample(is_close):
    """Grid-search on in-sample ONLY; return (best_params, best_is_return)."""
    best_params, best_ret = None, -np.inf
    for params in _valid_combos():
        pf = _backtest(is_close, params)
        ret = pf.total_return()
        if ret > best_ret:
            best_ret, best_params = ret, params
    return best_params, best_ret


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


def _buy_hold_return(close):
    return close.iloc[-1] / close.iloc[0] - 1.0


def main():
    print("=" * 74)
    print("WALK-FORWARD TREND-MOMENTUM STUDY  --  BACKTEST / PAPER ONLY (see CLAUDE.md)")
    print(f"Capital: ${INIT_CASH:,.0f}  |  fees={FEES:.3%}/trade  slippage={SLIPPAGE:.3%}/trade")
    print(f"Params tuned on IN-SAMPLE only, then judged ONCE on OUT-OF-SAMPLE (unseen).")
    print("=" * 74)

    oos_returns = []
    for symbol in ASSETS:
        print(f"\n################  {symbol}  ################")
        try:
            df = load_crypto(symbol=symbol, timeframe=TIMEFRAME, limit=LIMIT)
        except Exception as e:
            print(f"  [skip] could not load {symbol}: {e}")
            continue

        close = df["close"]
        split = int(len(close) * IS_FRACTION)
        is_close, oos_close = close.iloc[:split], close.iloc[split:]

        best_params, best_is_ret = _tune_in_sample(is_close)
        print(f"  best IN-SAMPLE params (chosen WITHOUT seeing test data): {best_params}")

        print("\n=== IN-SAMPLE (train) ===")
        is_ret, _, is_n = _stats("trend_momentum", _backtest(is_close, best_params), is_close)
        print(f"    buy & hold (same window): {_buy_hold_return(is_close):>8.2%}")

        print("\n=== OUT-OF-SAMPLE (test) ===")
        oos_ret, oos_sh, oos_n = _stats("trend_momentum", _backtest(oos_close, best_params), oos_close)
        print(f"    buy & hold (same window): {_buy_hold_return(oos_close):>8.2%}")

        gap = is_ret - oos_ret
        print(f"\n  IS->OOS return gap: {gap:>+.2%}  (large positive gap = overfitting signal)")
        oos_returns.append((symbol, oos_ret, oos_sh, oos_n))

    # ---- Honest verdict ------------------------------------------------------
    print("\n" + "=" * 74)
    print("OUT-OF-SAMPLE SUMMARY (the only numbers that speak to unseen data)")
    print("=" * 74)
    if not oos_returns:
        print("  No assets loaded (offline?). Re-run with a network connection.")
        return

    wins = 0
    for sym, ret, sh, n in oos_returns:
        flag = "profitable OOS" if ret > 0 else "LOSES OOS"
        if ret > 0:
            wins += 1
        print(f"  {sym:<10} OOS return {ret:>+8.2%}  sharpe {sh:>5.2f}  trades {n:>3d}   -> {flag}")

    print("\n--- Overfitting / robustness flags ---")
    print(f"  * Profitable out-of-sample on {wins}/{len(oos_returns)} assets.")
    if wins <= len(oos_returns) // 2:
        print("  * WEAK: fails on half or more of the assets out-of-sample. Treat as NOT an edge.")
    for sym, ret, sh, n in oos_returns:
        if n < 30:
            print(f"  * {sym}: only {n} OOS trades -- too few to trust any win-rate/return.")
        if sh > 3:
            print(f"  * {sym}: OOS Sharpe {sh:.2f} > 3 on daily data -- suspiciously high, distrust.")

    print("\n--- Verdict (per CLAUDE.md) ---")
    print("  This is NOT a validated, 'profitable', or 'working' strategy. The out-of-sample")
    print("  numbers above are a single walk-forward slice on three coins -- not multi-period")
    print("  walk-forward, not multi-year regime testing. 'Aggressive' here means full-equity")
    print("  long-only compounding in a BACKTEST; it is NOT advice to risk $1,500 of real money,")
    print("  and no live-trading code is wired up. Sizing aggressively on this would be exactly")
    print("  the overfitting trap the brief warned about.")


if __name__ == "__main__":
    main()
