"""Daily paper-trading runner for the BANDED ts_momentum strategy on Alpaca.

WHAT IT DOES (once per run -- intended to run once per trading day, after the close)
-----------------------------------------------------------------------------------
  1. Pull recent daily bars for the index ETF universe from Alpaca market data.
  2. Compute long/flat signals with strategies.ts_momentum.generate_signals -- the
     SAME banded time-series-momentum rule (band=0.03) that was walk-forward tested
     and validated across 8 indexes. NO live re-tuning (that would be the lookahead
     trap we've avoided throughout).
  3. Decide a TARGET portfolio: equal-split available equity across every ETF whose
     latest bar is signalled long. Long-only, no leverage. If none are long -> cash.
  4. Reconcile target vs. current Alpaca paper positions and submit market orders.

WHY THIS, WHY NOW (see the analysis history)
--------------------------------------------
This strategy is a VALIDATED DEFENSIVE OVERLAY, not an index-beater: across 8 indexes
(incl. 5 never used for tuning) it reliably cut max drawdown ~13pp vs buy & hold while
capturing ~44% of the upside. Paper trading here forward-tests EXECUTION & OPERATIONAL
correctness on genuinely unseen data -- it is NOT additional return validation (one
forward sample, one regime proves little). Judge it on whether fills, sizing, and
day-to-day reconciliation behave -- not on a few weeks of P&L.

SAFETY (see CLAUDE.md)
----------------------
  * PAPER ONLY. utils.alpaca_client refuses any non-paper endpoint.
  * DRY-RUN BY DEFAULT: prints the orders it WOULD place and exits. You must pass
    --live-paper to actually submit (still simulated paper orders).

ACCOUNT ISOLATION
-----------------
This runner uses a DEDICATED Alpaca paper account, configured in .env.tsmom (keys
prefixed TSMOM_ALPACA_*), loaded by utils/alpaca_client_tsmom.py. It does NOT read
the main .env, so it cannot touch the separate $1,500 paper account that belongs to
another already-automated strategy.

Usage (from project root, venv python):
    .venv\\Scripts\\python.exe utils\\alpaca_client_tsmom.py               # connectivity check (this account)
    .venv\\Scripts\\python.exe live\\ts_momentum_paper_trader.py           # DRY RUN (no orders)
    .venv\\Scripts\\python.exe live\\ts_momentum_paper_trader.py --live-paper   # submit paper orders
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

from strategies.ts_momentum import generate_signals, exit_levels
# ISOLATED client: reads ONLY .env.tsmom (TSMOM_ALPACA_* keys), never the main .env
# / $1,500 account tied to the other automated strategy.
from utils.alpaca_client_tsmom import get_trading_client, get_data_client

# Index ETF universe -- the validated set (SPY/QQQ/IWM tuning + held-out generalizers).
# ^GSPC/EFA/EEM/etc. are indices/intl that Alpaca may not trade as ETFs; we stick to
# US ETFs that Alpaca can actually fill. Equal-weight, long-only.
UNIVERSE = ["SPY", "QQQ", "IWM", "DIA", "MDY"]

# Strategy params FIXED at the validated defaults (band=0.03 chosen by walk-forward).
# Do NOT re-tune live -- tuning on data you then trade is the lookahead trap.
PARAMS = dict(lookback=126, trend=200, band=0.03)

# Bars of history to pull -- enough to warm up the `trend` SMA + momentum with margin.
LOOKBACK_DAYS = 500


def fetch_daily_closes(data_client) -> pd.DataFrame:
    """Return a DataFrame of daily close prices, one column per symbol.

    Free Paper accounts get IEX data (not paid SIP), so request feed=IEX. IEX delays
    the most recent data ~15 min, so end the window 1 day before now to stay clear of
    not-yet-available bars.
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


def desired_long_symbols(closes: pd.DataFrame, entries_today_only: bool = False) -> list[str]:
    """Symbols the strategy wants long.

    Two modes:
      * default (validated logic): a symbol is 'long' if its most recent entry
        signal is more recent than its most recent exit -- i.e. the position the
        backtested rule would currently be HOLDING (state-based).
      * entries_today_only=True: a symbol qualifies ONLY if a fresh entry fired on
        the most recent bar. This is an event-driven entry model -- stricter, fewer
        orders, and it intentionally diverges from the validated hold-the-state
        backtest (miss the entry bar and you miss the position entirely).
    """
    min_bars = max(PARAMS["lookback"], PARAMS["trend"]) + 5
    longs = []
    for sym in UNIVERSE:
        if sym not in closes.columns:
            print(f"  [warn] no data for {sym}; skipping")
            continue
        close = closes[sym].dropna()
        if len(close) < min_bars:
            print(f"  [warn] {sym}: only {len(close)} bars, need >{min_bars}; skipping")
            continue
        entries, exits = generate_signals(close, **PARAMS)
        last_bar = close.index.max()

        if entries_today_only:
            # Fire only if the entry signal is True on the most recent bar.
            if bool(entries.get(last_bar, False)):
                longs.append(sym)
            continue

        last_entry = entries[entries].index.max() if entries.any() else None
        last_exit = exits[exits].index.max() if exits.any() else None
        if last_entry is None:
            continue
        if last_exit is None or last_entry > last_exit:
            longs.append(sym)
    return longs


def build_targets(equity: float, longs: list[str]) -> dict[str, float]:
    """Equal-split equity across the long symbols. None long -> empty (all cash)."""
    if not longs:
        return {}
    alloc = equity / len(longs)
    return {sym: alloc for sym in longs}


def report_exit_levels(closes: pd.DataFrame, held: set[str] | None = None) -> list[dict]:
    """Print the live exit thresholds for each symbol; return the rows.

    For every symbol with enough data, shows the price-level exit (SMA*(1-band)),
    the % cushion above it, trailing momentum, and whether the momentum exit has
    tripped. Held symbols are flagged. Sorted by least cushion first (closest to an
    exit at the top). Read-only -- never touches orders.
    """
    held = held or set()
    min_bars = max(PARAMS["lookback"], PARAMS["trend"]) + 5
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

    print("\n  EXIT LEVELS (price closes below band_exit OR momentum<=0 -> sell next run):")
    print(f"    {'sym':<5}{'held':>5}{'last':>10}{'200d SMA':>11}"
          f"{'band exit':>11}{'cushion':>9}{'momentum':>10}{'mom exit':>10}")
    for r in rows:
        print(f"    {r['symbol']:<5}{('yes' if r['held'] else '-'):>5}"
              f"{r['last_close']:>10.2f}{r['sma']:>11.2f}{r['band_exit']:>11.2f}"
              f"{r['pct_above_band_exit']:>+8.1%}{r['momentum']:>+10.1%}"
              f"{('YES' if r['momentum_exit'] else 'no'):>10}")
    return rows


def current_positions(trading_client) -> dict[str, float]:
    """Map symbol -> current market value held (paper)."""
    out = {}
    for p in trading_client.get_all_positions():
        out[p.symbol] = float(p.market_value)
    return out


def symbols_with_open_orders(trading_client) -> set[str]:
    """Symbols that already have an unfilled (open) order.

    Guards unattended runs: an order placed pre-market sits OPEN until the session
    opens; without this guard the next run sees "held $0" and stacks a duplicate.
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
    ap.add_argument("--entries-today-only", action="store_true",
                    help="only buy symbols whose entry fired on the most recent bar "
                         "(event-driven; diverges from the hold-the-state backtest)")
    ap.add_argument("--exits", action="store_true",
                    help="print the live exit levels for each symbol and exit "
                         "(read-only; never trades)")
    args = ap.parse_args()

    tc = get_trading_client()
    dc = get_data_client()

    acct = tc.get_account()
    equity = float(acct.portfolio_value)
    print("=" * 72)
    print("ALPACA PAPER TS-MOMENTUM RUNNER  --  PAPER ONLY (CLAUDE.md)")
    print(f"  account status : {acct.status}   equity: ${equity:,.2f}")
    print(f"  mode           : {'LIVE-PAPER (submitting orders)' if args.live_paper else 'DRY RUN (no orders)'}")
    print(f"  universe       : {UNIVERSE}")
    print(f"  params (FIXED) : {PARAMS}")
    entry_mode = "ENTRIES-TODAY-ONLY (event-driven)" if args.entries_today_only \
        else "HOLD-STATE (validated backtest logic)"
    print(f"  entry mode     : {entry_mode}")
    print("=" * 72)

    closes = fetch_daily_closes(dc)
    held = current_positions(tc)

    # --exits: report-only command. Print live exit levels and stop (no trading).
    if args.exits:
        report_exit_levels(closes, held=set(held))
        return

    longs = desired_long_symbols(closes, entries_today_only=args.entries_today_only)
    label = "entered TODAY" if args.entries_today_only else "signalled LONG today"
    print(f"\n  {label}: {longs or '(none -- no action)'}")

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
