"""ETH/USDT overfitting test of DonchianBreakoutPro on the VectorBT port.

Runs the UNCHANGED strategy + params (tuned on BTC) against ETH-USDT 4h perp data,
2023 -> now, per-year + full-period. This is the VBT half of a two-engine comparison;
the FAITHFUL verdict comes from the Jesse run. We already proved VBT's 4h fills impose a
systematic entry penalty on this intrabar strategy, so VBT's ABSOLUTE numbers are biased
low -- the value here is whether the Jesse-vs-VBT divergence repeats on a second coin.

Run:
    .venv\\Scripts\\python.exe backtests\\eth_crosscheck.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategies.donchian_breakout import run_backtest, DEFAULT_PARAMS
from utils.crypto_loader import load_perp_range

FEE_RATE = 0.0004   # Binance USDT-M futures taker
SLIPPAGE = 0.0005
INIT_CASH = 10_000
LEVERAGE = 2.0

# Symbol is overridable via CLI: `... eth_crosscheck.py SOL/USDT`. Defaults to ETH.
SYMBOL = sys.argv[1] if len(sys.argv) > 1 else "ETH/USDT"
TIMEFRAME = "4h"


def _segment_stats(pf, label):
    trades = pf.trades
    n = trades.count()
    win = trades.win_rate() if n > 0 else float("nan")
    print(f"\n=== {label} ===")
    print(f"  total return     : {pf.total_return():.2%}")
    try:
        print(f"  annualized return: {pf.annualized_return():.2%}")
    except Exception:
        print("  annualized return: n/a")
    print(f"  sharpe ratio     : {pf.sharpe_ratio():.2f}")
    print(f"  max drawdown     : {pf.max_drawdown():.2%}")
    print(f"  win rate         : {win:.2%}" if n > 0 else "  win rate         : n/a (no trades)")
    print(f"  # trades         : {n}")
    return dict(total_return=pf.total_return(), sharpe=pf.sharpe_ratio(),
               max_dd=pf.max_drawdown(), n_trades=int(n))


def main():
    print(f"Loading {SYMBOL} {TIMEFRAME} perp (2022-09 warm-up -> now) ...")
    df = load_perp_range(symbol=SYMBOL, timeframe=TIMEFRAME, start="2022-09-01", end=None)
    df = df[~df.index.duplicated(keep="first")].sort_index()
    print(f"Loaded {len(df)} candles: {df.index[0]} -> {df.index[-1]}")

    print(f"\nStrategy: DonchianBreakoutPro on {SYMBOL} (UNCHANGED BTC-tuned params).")
    print(f"Params : {DEFAULT_PARAMS}")
    print(f"Costs  : fee={FEE_RATE:.3%}/trade  slippage={SLIPPAGE:.3%}/trade  leverage={LEVERAGE}x")
    print("NOTE: VBT 4h fills bias these numbers LOW vs Jesse; read the verdict off Jesse.")

    observed = {}
    for year in (2023, 2024, 2025, 2026):
        sub = df[df.index.year == year]
        if len(sub) < DEFAULT_PARAMS["entry_period"] + DEFAULT_PARAMS["atr_period"] + 5:
            continue
        pf_y = run_backtest(sub, init_cash=INIT_CASH, fee_rate=FEE_RATE,
                            slippage=SLIPPAGE, leverage=LEVERAGE, params=DEFAULT_PARAMS)
        label = f"{year}" if year != 2026 else "2026 (YTD)"
        observed[year] = _segment_stats(pf_y, label)

    # Full continuous period.
    df_full = df[df.index.year >= 2023]
    pf_full = run_backtest(df_full, init_cash=INIT_CASH, fee_rate=FEE_RATE,
                           slippage=SLIPPAGE, leverage=LEVERAGE, params=DEFAULT_PARAMS)
    print("\n" + "=" * 70)
    _segment_stats(pf_full, "FULL PERIOD (2023 -> now, continuous)")

    print("\n" + "-" * 70)
    print("VBT ETH summary (biased low by 4h fills; Jesse is the faithful verdict):")
    for year, o in observed.items():
        print(f"  {year}: total {o['total_return']:+.1%}  sharpe {o['sharpe']:+.2f}  "
              f"maxDD {o['max_dd']:.1%}  trades {o['n_trades']}")


if __name__ == "__main__":
    main()
