"""Cross-check the Jesse 'DonchianBreakoutPro' strategy on VectorBT.

Loads BTC/USDT 4h data (2023 -> now), runs the full-fidelity VectorBT port, and reports
performance PER YEAR (2023, 2024, 2025, 2026-YTD) plus a full-period run. Finishes with a
CLAIMED (Jesse) vs OBSERVED (VectorBT) comparison and overfitting/divergence commentary.

Run from the project root with the venv python:
    .venv\\Scripts\\python.exe backtests\\donchian_crosscheck.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategies.donchian_breakout import run_backtest, DEFAULT_PARAMS
from utils.crypto_loader import load_crypto

# Binance USDT-M futures taker fee ~= 0.04%; matches Jesse's futures config order of
# magnitude so the comparison is fair. Slippage kept modest for a liquid 4h market.
FEE_RATE = 0.0004
SLIPPAGE = 0.0005
INIT_CASH = 10_000
LEVERAGE = 2.0

SYMBOL = "BTC/USDT"
TIMEFRAME = "4h"
# 4h bars: ~2190/yr. 2023-01 -> 2026-06 is ~3.5yr -> ~7700 bars; pad for warm-up.
LIMIT = 8200

# Jesse docstring's CLAIMED numbers, for side-by-side comparison.
JESSE_CLAIMED = {
    2023: dict(total_return=0.792, sharpe=2.00, max_dd=-0.138),
    2024: dict(total_return=0.429, sharpe=1.21, max_dd=-0.214),
}


def _segment_stats(pf, label):
    """Print labelled stats for a Portfolio (already date-sliced)."""
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
    print(f"Loading {SYMBOL} {TIMEFRAME} ({LIMIT} candles) ...")
    df = load_crypto(symbol=SYMBOL, timeframe=TIMEFRAME, limit=LIMIT)
    df = df[~df.index.duplicated(keep="first")].sort_index()
    print(f"Loaded {len(df)} candles: {df.index[0]} -> {df.index[-1]}")

    print(f"\nStrategy: DonchianBreakoutPro (Jesse port) -- CROSS-CHECK, not independent validation.")
    print(f"Params : {DEFAULT_PARAMS}")
    print(f"Costs  : fee={FEE_RATE:.3%}/trade  slippage={SLIPPAGE:.3%}/trade  leverage={LEVERAGE}x")

    # Run once over the FULL series (state/indicators need continuous history); then
    # slice the resulting portfolio per calendar year for reporting.
    pf_full = run_backtest(
        df, init_cash=INIT_CASH, fee_rate=FEE_RATE, slippage=SLIPPAGE,
        leverage=LEVERAGE, params=DEFAULT_PARAMS,
    )

    observed = {}
    for year in (2023, 2024, 2025, 2026):
        mask = pf_full.wrapper.index.year == year
        if not mask.any():
            continue
        # Re-run on the sliced price so per-year returns start from a clean equity base.
        sub = df[df.index.year == year]
        if len(sub) < DEFAULT_PARAMS["entry_period"] + DEFAULT_PARAMS["atr_period"] + 5:
            continue
        pf_y = run_backtest(
            sub, init_cash=INIT_CASH, fee_rate=FEE_RATE, slippage=SLIPPAGE,
            leverage=LEVERAGE, params=DEFAULT_PARAMS,
        )
        label = f"{year}" if year != 2026 else "2026 (YTD)"
        observed[year] = _segment_stats(pf_y, label)

    print("\n" + "=" * 70)
    _segment_stats(pf_full, "FULL PERIOD (2023 -> now, continuous)")

    # ---- Claimed vs observed -------------------------------------------- #
    print("\n" + "-" * 70)
    print("CLAIMED (Jesse) vs OBSERVED (VectorBT cross-check):")
    print(f"  {'year':<6}{'metric':<14}{'Jesse claim':>14}{'VectorBT':>14}")
    for year, claim in JESSE_CLAIMED.items():
        obs = observed.get(year)
        if not obs:
            continue
        for metric, name in [("total_return", "total return"), ("sharpe", "sharpe"), ("max_dd", "max drawdown")]:
            cv = claim[metric]
            ov = obs[metric]
            cfmt = f"{cv:.2%}" if metric != "sharpe" else f"{cv:.2f}"
            ofmt = f"{ov:.2%}" if metric != "sharpe" else f"{ov:.2f}"
            print(f"  {year:<6}{name:<14}{cfmt:>14}{ofmt:>14}")

    print("\n" + "-" * 70)
    print("Interpretation (per CLAUDE.md):")
    print("- This is a CROSS-CHECK on a second engine, NOT independent proof of an edge.")
    print("- Expect SOME divergence from Jesse: bar-resolution stops, reproduced risk/margin")
    print("  math, and once-per-bar pyramid timing all differ from Jesse's internals.")
    print("- A LARGE gap (e.g. VectorBT far below the claimed +79%/+43%) is a red flag for")
    print("  overfitting or engine-specific/look-ahead artifacts in the original result.")
    print("- 6 tuned parameters on a single coin/timeframe is inherently overfitting-prone;")
    print("  agreement on 2023/2024 alone does NOT make this 'validated' or 'profitable'.")


if __name__ == "__main__":
    main()
