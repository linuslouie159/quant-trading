"""Daily paper-trading runner for the trend_momentum strategy on Alpaca.

WHAT IT DOES (once per run -- intended to run once per trading day, after the close)
-----------------------------------------------------------------------------------
  1. Pull recent daily bars for the ETF universe from Alpaca market data.
  2. Compute long/flat signals with strategies.trend_momentum.generate_signals
     (the SAME rule that was walk-forward tested -- no second strategy to drift).
  3. Decide a TARGET portfolio: equal-split available equity across every ETF whose
     latest bar is signalled long ("full-equity aggressive": if only one is long it
     gets ~100%; if k are long they each get ~1/k). Long-only, no leverage.
  4. Reconcile target vs. current Alpaca paper positions and submit market orders to
     close exits and open/resize entries.

SAFETY (see CLAUDE.md)
----------------------
  * PAPER ONLY. utils.alpaca_client refuses any non-paper endpoint.
  * DRY-RUN BY DEFAULT: prints the orders it WOULD place and exits. You must pass
    --live-paper to actually submit (even though those are still simulated orders).
  * This forward-tests on genuinely unseen data, which is the honest answer to "don't
    build on past data" -- but it is NOT validation. One forward sample, one regime.

Usage (from project root, venv python):
    .venv\\Scripts\\python.exe utils\\alpaca_client.py        # connectivity check
    .venv\\Scripts\\python.exe live\\paper_trader.py           # DRY RUN (no orders)
    .venv\\Scripts\\python.exe live\\paper_trader.py --live-paper   # submit paper orders
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import DataFeed
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

from strategies.trend_momentum import generate_signals, exit_levels
from utils.alpaca_client import get_trading_client, get_data_client

# Universe + strategy params. These params were the most common walk-forward winners;
# they are FIXED here (no re-tuning live -- tuning on the data you then trade is the
# exact lookahead trap we've been avoiding).
UNIVERSE = ["SPY", "QQQ", "IWM", "GLD", "TLT", "XLE"]
PARAMS = dict(breakout=30, exit_n=20, trend=150)

# Bars of history to pull -- enough to warm up the `trend` SMA with margin.
LOOKBACK_DAYS = 400


def fetch_daily_closes(data_client) -> pd.DataFrame:
    """Return a DataFrame of daily close prices, one column per symbol.

    Free Paper Only accounts are entitled to IEX data, not the paid SIP feed, so we
    request feed=IEX explicitly. IEX also delays the most recent data ~15 min, so we
    end the window 1 day before now to stay clear of not-yet-available bars.
    """
    end = datetime.now(timezone.utc) - timedelta(days=1)
    start = end - timedelta(days=LOOKBACK_DAYS)
    req = StockBarsRequest(
        symbol_or_symbols=UNIVERSE,
        timeframe=TimeFrame.Day,
        start=start,
        end=end,
        feed=DataFeed.IEX,
    )
    bars = data_client.get_stock_bars(req).df  # MultiIndex (symbol, timestamp)
    closes = bars["close"].unstack(level=0)    # -> columns = symbols
    return closes.sort_index()


def desired_long_symbols(closes: pd.DataFrame) -> list[str]:
    """Symbols whose LATEST bar is in a long state per trend_momentum.

    A symbol is 'long' if its most recent entry signal is more recent than its most
    recent exit signal (i.e. the position the rule would currently be holding).
    """
    longs = []
    for sym in UNIVERSE:
        if sym not in closes.columns:
            print(f"  [warn] no data for {sym}; skipping")
            continue
        close = closes[sym].dropna()
        if len(close) < PARAMS["trend"] + 5:
            print(f"  [warn] {sym}: only {len(close)} bars, need >{PARAMS['trend']}; skipping")
            continue
        entries, exits = generate_signals(close, **PARAMS)
        last_entry = entries[entries].index.max() if entries.any() else None
        last_exit = exits[exits].index.max() if exits.any() else None
        if last_entry is None:
            continue
        if last_exit is None or last_entry > last_exit:
            longs.append(sym)
    return longs


def build_targets(equity: float, longs: list[str]) -> dict[str, float]:
    """Equal-split equity across the long symbols (full-equity aggressive)."""
    if not longs:
        return {}
    alloc = equity / len(longs)
    return {sym: alloc for sym in longs}


def report_exit_levels(closes: pd.DataFrame, held: set[str] | None = None) -> list[dict]:
    """Print the live exit thresholds for each symbol; return the rows.

    For trend_momentum the exit is the HIGHER of the prior exit_n-bar low (Donchian
    trailing stop) and the trend SMA -- price closing below the binding level sells
    the position on the next run. Held symbols are flagged; sorted by least cushion
    first (closest to an exit at the top). Read-only -- never touches orders.
    """
    held = held or set()
    min_bars = PARAMS["trend"] + 5
    rows = []
    for sym in UNIVERSE:
        if sym not in closes.columns:
            continue
        close = closes[sym].dropna()
        if len(close) < min_bars:
            continue
        lv = exit_levels(close, **PARAMS)
        lv["symbol"] = sym
        lv["held"] = sym in held
        rows.append(lv)
    rows.sort(key=lambda r: r["cushion"])  # closest to exit first

    print("\n  EXIT LEVELS (price closes below stop -> sell next run; stop = max(Donchian low, SMA)):")
    print(f"    {'sym':<5}{'held':>5}{'last':>10}{'SMA':>10}{'Donch low':>11}"
          f"{'stop':>10}{'cushion':>9}")
    for r in rows:
        print(f"    {r['symbol']:<5}{('yes' if r['held'] else '-'):>5}"
              f"{r['last_close']:>10.2f}{r['sma']:>10.2f}{r['donchian_low']:>11.2f}"
              f"{r['band_exit']:>10.2f}{r['pct_above_band_exit']:>+8.1%}")
    return rows


def current_positions(trading_client) -> dict[str, float]:
    """Map symbol -> current market value held (paper)."""
    out = {}
    for p in trading_client.get_all_positions():
        out[p.symbol] = float(p.market_value)
    return out


def symbols_with_open_orders(trading_client) -> set[str]:
    """Symbols that already have an unfilled (open) order.

    Critical for unattended runs: an order placed pre-market sits OPEN until the
    session opens. Without this guard the next run would see "held $0" for that
    symbol and submit ANOTHER order, stacking duplicates every day until fill.
    """
    from alpaca.trading.requests import GetOrdersRequest
    from alpaca.trading.enums import QueryOrderStatus

    open_orders = trading_client.get_orders(
        GetOrdersRequest(status=QueryOrderStatus.OPEN)
    )
    return {o.symbol for o in open_orders}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--live-paper", action="store_true",
                    help="actually SUBMIT paper orders (default: dry run, prints only)")
    ap.add_argument("--exits", action="store_true",
                    help="print the live exit levels for each symbol and exit "
                         "(read-only; never trades)")
    args = ap.parse_args()

    tc = get_trading_client()
    dc = get_data_client()

    acct = tc.get_account()
    equity = float(acct.portfolio_value)
    print("=" * 70)
    print("ALPACA PAPER TREND-MOMENTUM RUNNER  --  PAPER ONLY (CLAUDE.md)")
    print(f"  account status : {acct.status}   equity: ${equity:,.2f}")
    print(f"  mode           : {'LIVE-PAPER (submitting orders)' if args.live_paper else 'DRY RUN (no orders)'}")
    print(f"  params (FIXED) : {PARAMS}")
    print("=" * 70)

    closes = fetch_daily_closes(dc)
    held = current_positions(tc)

    # --exits: report-only command. Print live exit levels and stop (no trading).
    if args.exits:
        report_exit_levels(closes, held=set(held))
        return

    longs = desired_long_symbols(closes)
    print(f"\n  signalled LONG today: {longs or '(none -- stay in cash)'}")

    targets = build_targets(equity, longs)
    pending = symbols_with_open_orders(tc)
    if pending:
        print(f"  pending (unfilled) orders, will NOT re-submit: {sorted(pending)}")
    symbols = sorted(set(targets) | set(held))

    print("\n  reconciliation (target $ vs held $):")
    actions = []  # (symbol, side, notional)
    for sym in symbols:
        tgt = targets.get(sym, 0.0)
        cur = held.get(sym, 0.0)
        diff = tgt - cur
        note = "  [pending order -> skip]" if sym in pending else ""
        print(f"    {sym:<5} target ${tgt:>9,.2f}  held ${cur:>9,.2f}  delta ${diff:>+9,.2f}{note}")
        # Skip any symbol with an unfilled order so we don't stack duplicates.
        if sym in pending:
            continue
        # Exit fully if no longer a target.
        if tgt == 0.0 and cur > 0.0:
            actions.append((sym, "close", cur))
        elif abs(diff) > max(1.0, 0.02 * max(tgt, 1.0)):  # ignore tiny rebalances (<2%)
            side = "buy" if diff > 0 else "sell"
            actions.append((sym, side, abs(diff)))

    if not actions:
        print("\n  no actions -- portfolio already matches target.")
    else:
        print("\n  planned orders:")
        for sym, side, notional in actions:
            print(f"    {side.upper():<5} {sym:<5} ~${notional:,.2f}")

        if not args.live_paper:
            print("\n  DRY RUN -- no orders submitted. Re-run with --live-paper to execute (paper).")
        else:
            print("\n  submitting paper orders ...")
            for sym, side, notional in actions:
                if side == "close":
                    tc.close_position(sym)
                    print(f"    closed {sym}")
                    continue
                order = MarketOrderRequest(
                    symbol=sym,
                    notional=round(notional, 2),  # fractional notional order
                    side=OrderSide.BUY if side == "buy" else OrderSide.SELL,
                    time_in_force=TimeInForce.DAY,
                )
                resp = tc.submit_order(order)
                print(f"    {side} {sym} ${notional:,.2f} -> order id {resp.id}")
            print("\n  done. Verify fills in the Alpaca paper dashboard.")

    # Always log the live exit levels so each daily run leaves a record of how much
    # cushion every held position has before its exit triggers.
    report_exit_levels(closes, held=set(held))


if __name__ == "__main__":
    main()
