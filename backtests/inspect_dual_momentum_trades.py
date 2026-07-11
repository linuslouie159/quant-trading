"""Inspect the ACTUAL simulated trades behind backtests/dual_momentum.py.

These are SIMULATED fills on historical data -- no real orders were placed (see
CLAUDE.md). This script rebuilds the OUT-OF-SAMPLE portfolio with the identical
logic and prints (a) the monthly target-weight decisions, (b) every executed order,
and (c) the closed round-trip trades. Run:

    .venv\\Scripts\\python.exe backtests\\inspect_dual_momentum_trades.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

import backtests.dual_momentum as dm

pd.set_option("display.width", 160)
pd.set_option("display.max_rows", 80)


def main():
    close = dm._load_basket()
    split = int(len(close) * dm.IS_FRACTION)
    oos = close.iloc[split:]

    weights = dm._build_weights_with_cash(oos)

    # Show only the rows where the target allocation CHANGES (i.e. rebalance actions).
    changed = weights.ne(weights.shift()).any(axis=1)
    decisions = weights[changed]
    print("=== OUT-OF-SAMPLE target-weight DECISIONS (only rows where allocation changes) ===")
    print(f"(rule: top-{dm.TOP_N} of {len(close.columns)} by {dm.LOOKBACK_DAYS}d return, "
          f"cash if trailing return <= 0)\n")
    print(decisions.round(3).to_string())

    pf, _ = dm.run_segment("OUT-OF-SAMPLE (rebuilt for inspection)", oos)

    # --- Executed orders (every fill the simulator made) --------------------
    orders = pf.orders.records_readable
    print(f"\n=== EXECUTED ORDERS (simulated fills): {len(orders)} total ===")
    cols = [c for c in ["Timestamp", "Column", "Size", "Price", "Fees", "Side"] if c in orders.columns]
    print(orders[cols].to_string(index=False))

    # --- Closed round-trip trades with P&L ----------------------------------
    trades = pf.trades.records_readable
    print(f"\n=== CLOSED ROUND-TRIP TRADES: {len(trades)} total ===")
    tcols = [c for c in ["Column", "Entry Timestamp", "Exit Timestamp", "Size",
                         "Avg Entry Price", "Avg Exit Price", "PnL", "Return"]
             if c in trades.columns]
    print(trades[tcols].to_string(index=False))


if __name__ == "__main__":
    main()
