"""Export the VectorBT port's 2024 BTC/USDT 4h order + trade list for cross-engine
comparison against Jesse's trade-by-trade log.

Emits TWO views:
  1. ORDERS  -- every fill (entry, pyramid add, exit), with timestamp/price/size/side.
     This is the one that exposes pyramiding and exit timing directly.
  2. TRADES  -- netted positions (vectorbt's trade records) for a higher-level view.

Run:
    .venv\\Scripts\\python.exe backtests\\export_2024_trades.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from strategies.donchian_breakout import run_backtest, DEFAULT_PARAMS
from utils.crypto_loader import load_crypto

FEE_RATE = 0.0004
SLIPPAGE = 0.0005
INIT_CASH = 10_000
LEVERAGE = 2.0

pd.set_option("display.max_rows", 400)
pd.set_option("display.width", 200)


def main():
    df = load_crypto(symbol="BTC/USDT", timeframe="4h", limit=8200)
    df = df[~df.index.duplicated(keep="first")].sort_index()

    # Run on 2024 only so the equity base + indexing start clean for the comparison.
    sub = df[df.index.year == 2024]
    print(f"2024 candles: {len(sub)}  ({sub.index[0]} -> {sub.index[-1]})")

    pf = run_backtest(sub, init_cash=INIT_CASH, fee_rate=FEE_RATE,
                      slippage=SLIPPAGE, leverage=LEVERAGE, params=DEFAULT_PARAMS)

    # ---- ORDERS (every fill) ---- #
    orders = pf.orders.records_readable.copy()
    # Attach the bar timestamp for each order via its index position.
    idx = pf.wrapper.index
    if "Timestamp" not in orders.columns and "Index" in orders.columns:
        orders["Timestamp"] = orders["Index"].map(lambda i: idx[int(i)] if pd.notna(i) else None)
    print("\n================ VECTORBT ORDERS (2024) ================")
    keep = [c for c in ["Timestamp", "Side", "Size", "Price", "Fees"] if c in orders.columns]
    print(orders[keep].to_string(index=False) if len(orders) else "  (no orders)")
    print(f"\ntotal orders (fills incl. pyramid adds + exits): {len(orders)}")

    # ---- TRADES (netted positions) ---- #
    trades = pf.trades.records_readable.copy()
    print("\n================ VECTORBT TRADES (2024, netted) ================")
    tkeep = [c for c in ["Entry Timestamp", "Exit Timestamp", "Direction", "Size",
                         "Avg Entry Price", "Avg Exit Price", "PnL", "Return"]
            if c in trades.columns]
    print(trades[tkeep].to_string(index=False) if len(trades) else "  (no trades)")
    print(f"\nnetted trades: {len(trades)}   "
          f"2024 total return: {pf.total_return():.2%}   sharpe: {pf.sharpe_ratio():.2f}")


if __name__ == "__main__":
    main()
