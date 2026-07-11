"""Reconcile the VBT port's FIRST 2024 trade against Jesse's logged trade #1.

Jesse trade #1 (2024-01-08 16:00 UTC, Binance Perpetual Futures):
  unit1 buy 0.22554757 @ 44230.2 (bar open)
  unit2 buy 0.11592557 @ 45435.0 (next 4h bar open)
  stop  sell 0.34147314 @ 43433.45
  size $15,243   PNL -$423.76 (-5.56%)

We feed the SAME perp feed, build indicators on a continuous window ending at 2024,
and print the VBT orders around 2024-01-08 to line up entry price / unit sizes / exit.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from strategies.donchian_breakout import run_backtest, DEFAULT_PARAMS
from utils.crypto_loader import load_perp_range

FEE_RATE = 0.0004   # Binance USDT-M futures taker
SLIPPAGE = 0.0      # set 0 to isolate logic; Jesse log has no extra slippage on these
INIT_CASH = 10_000
LEVERAGE = 2.0

pd.set_option("display.width", 200)
pd.set_option("display.max_rows", 60)

JESSE = """JESSE trade #1:
  unit1 buy  0.22554757 @ 44230.2
  unit2 buy  0.11592557 @ 45435.0
  exit  sell 0.34147314 @ 43433.45   size $15,243  PNL -$423.76 (-5.56%)"""


def main():
    # Continuous history from 2022-09 so Donchian/ATR are warm well before 2024-01-08.
    df = load_perp_range(symbol="BTC/USDT", timeframe="4h",
                         start="2022-09-01", end="2024-03-01")
    df = df[~df.index.duplicated(keep="first")].sort_index()
    print(f"perp candles: {len(df)}  {df.index[0]} -> {df.index[-1]}")

    # Show the raw bar at the entry to confirm feed/open price.
    bar = df.loc["2024-01-08 16:00:00+00:00"] if "2024-01-08 16:00:00+00:00" in df.index else None
    if bar is not None:
        print(f"\n2024-01-08 16:00 bar: open={bar['open']:.1f} high={bar['high']:.1f} "
              f"low={bar['low']:.1f} close={bar['close']:.1f}")

    pf = run_backtest(df, init_cash=INIT_CASH, fee_rate=FEE_RATE,
                      slippage=SLIPPAGE, leverage=LEVERAGE, params=DEFAULT_PARAMS)

    orders = pf.orders.records_readable.copy()
    idx = pf.wrapper.index
    if "Index" in orders.columns:
        orders["Timestamp"] = orders["Index"].map(lambda i: idx[int(i)] if pd.notna(i) else None)
    # Window around the first trade.
    win = orders[(orders["Timestamp"] >= "2024-01-07") & (orders["Timestamp"] <= "2024-01-20")]
    keep = [c for c in ["Timestamp", "Side", "Size", "Price", "Fees"] if c in win.columns]
    print("\n================ VBT ORDERS around 2024-01-08 ================")
    print(win[keep].to_string(index=False) if len(win) else "  (no orders in window)")

    print("\n" + JESSE)


if __name__ == "__main__":
    main()
