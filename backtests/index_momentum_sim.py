"""Paper-trade simulation + explained trade log for time-series momentum.

WHAT THIS DOES
--------------
Runs the strategy from strategies/ts_momentum.py on an index as a PAPER-TRADE
simulation (no real orders, ever -- see CLAUDE.md), then:
  1. Prints the simulated ORDER fills (buys/sells), each with the fill price and
     the fee + slippage that was charged.
  2. Prints a sampled TRADE LOG (entry -> exit round-trips) with a plain-English
     explanation of WHY the strategy was in the market for each one and how it did.
  3. Reports IN-SAMPLE vs OUT-OF-SAMPLE summaries separately and clearly labelled.

Fees + slippage are applied to every fill (never a zero-cost backtest). This is
backtesting / paper trading only -- it does not connect to any exchange or broker.

Run from the project root with the venv python:
    .venv\\Scripts\\python.exe backtests\\index_momentum_sim.py
    .venv\\Scripts\\python.exe backtests\\index_momentum_sim.py QQQ
"""

from __future__ import annotations

import os
import sys

# Make the project root importable when run as a script.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import vectorbt as vbt

from strategies.ts_momentum import generate_signals
from utils.costs import FEES, SLIPPAGE, INIT_CASH
from utils.stock_loader import load_stock

# ---- Config -----------------------------------------------------------------
DEFAULT_SYMBOL = "SPY"
START = "2005-01-01"
IS_FRACTION = 0.70
LOOKBACK = 126
TREND = 200
SAMPLE_TRADES = 6     # how many trades to explain in detail (evenly sampled)


def build_portfolio(close):
    """Generate signals and run the paper-trade simulation with costs applied."""
    entries, exits = generate_signals(close, lookback=LOOKBACK, trend=TREND)
    pf = vbt.Portfolio.from_signals(
        close, entries, exits,
        fees=FEES, slippage=SLIPPAGE, init_cash=INIT_CASH, freq="1D",
    )
    return pf, entries, exits


def print_segment_summary(label, pf):
    print(f"\n=== {label} ===")
    print(f"  period      : {pf.close.index[0].date()} -> {pf.close.index[-1].date()}  ({len(pf.close)} bars)")
    print(f"  costs       : fees={FEES:.3%}/trade  slippage={SLIPPAGE:.3%}/trade")
    n = pf.trades.count()
    wr = pf.trades.win_rate() if n > 0 else float("nan")
    print(f"  total return     : {pf.total_return():+.2%}")
    print(f"  sharpe ratio     : {pf.sharpe_ratio():.2f}")
    print(f"  max drawdown     : {pf.max_drawdown():.2%}")
    print(f"  win rate         : {wr:.2%}" if n > 0 else "  win rate         : n/a")
    print(f"  # round-trip trades : {n}")


def print_order_fills(pf, max_rows=12):
    """Show the simulated order fills: side, size, fill price, fee charged."""
    orders = pf.orders.records_readable
    print(f"\n  --- SIMULATED ORDER FILLS (first {min(max_rows, len(orders))} of {len(orders)}) ---")
    print("  Each fill includes the {:.2%} fee; slippage of {:.2%} is already baked into the price."
          .format(FEES, SLIPPAGE))
    print(f"  {'date':<12}{'side':<6}{'shares':>10}{'fill price':>13}{'fee $':>10}{'notional $':>13}")
    for _, o in orders.head(max_rows).iterrows():
        notional = o["Size"] * o["Price"]
        print(f"  {str(o['Timestamp'].date()):<12}{o['Side']:<6}{o['Size']:>10.2f}"
              f"{o['Price']:>13.2f}{o['Fees']:>10.2f}{notional:>13.2f}")


def explain_trades(pf, close):
    """Print a sampled, plain-English explanation of individual round-trip trades."""
    trades = pf.trades.records_readable
    if len(trades) == 0:
        print("\n  (no trades to explain)")
        return

    # Evenly sample up to SAMPLE_TRADES trades across the whole history.
    n = len(trades)
    k = min(SAMPLE_TRADES, n)
    idx = [round(i * (n - 1) / (k - 1)) for i in range(k)] if k > 1 else [0]

    # Precompute the strategy inputs so we can explain the "why" at entry time.
    prior = close.shift(1)
    momentum = prior / prior.shift(LOOKBACK) - 1.0
    trend_ma = close.rolling(TREND).mean()

    print(f"\n  --- TRADE LOG (explaining {k} of {n} round-trips) ---")
    for rank, i in enumerate(idx, 1):
        t = trades.iloc[i]
        entry_dt = t["Entry Timestamp"]
        exit_dt = t["Exit Timestamp"]
        held_days = (exit_dt - entry_dt).days
        mom_at_entry = momentum.get(entry_dt, float("nan"))
        px_at_entry = close.get(entry_dt, float("nan"))
        sma_at_entry = trend_ma.get(entry_dt, float("nan"))
        won = t["PnL"] > 0

        print(f"\n  Trade #{i} ({rank}/{k}):  {'WIN ' if won else 'LOSS'}")
        print(f"    entered : {entry_dt.date()} @ ${t['Avg Entry Price']:.2f}   "
              f"(bought {t['Size']:.2f} shares, entry fee ${t['Entry Fees']:.2f})")
        print(f"    exited  : {exit_dt.date()} @ ${t['Avg Exit Price']:.2f}   "
              f"(exit fee ${t['Exit Fees']:.2f})")
        print(f"    held    : {held_days} calendar days")
        print(f"    result  : PnL ${t['PnL']:+.2f}   return {t['Return']:+.2%} (net of costs)")
        # Why the strategy was long here:
        mom_txt = f"{mom_at_entry:+.1%}" if pd.notna(mom_at_entry) else "n/a"
        regime = "above" if pd.notna(sma_at_entry) and px_at_entry > sma_at_entry else "below"
        print(f"    why long: {LOOKBACK}-bar momentum was {mom_txt} (positive) and price "
              f"${px_at_entry:.2f} was {regime} the {TREND}-day SMA ${sma_at_entry:.2f}.")
        print(f"    why exit: momentum turned non-positive and/or price closed back below "
              f"the {TREND}-day SMA, flipping the regime to flat (cash).")


def main():
    symbol = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_SYMBOL
    print(f"PAPER-TRADE SIMULATION (no real orders) -- time-series momentum on {symbol}")
    print(f"Strategy: lookback={LOOKBACK}, trend={TREND}  |  starting cash ${INIT_CASH:,.0f}")

    df = load_stock(symbol, start=START)
    close = df["close"]

    split = int(len(close) * IS_FRACTION)
    is_close, oos_close = close.iloc[:split], close.iloc[split:]

    # Full-history simulation for the explained order/trade log.
    pf_full, _, _ = build_portfolio(close)
    print_order_fills(pf_full)
    explain_trades(pf_full, close)

    # IS / OOS summaries, separated and labelled (per CLAUDE.md).
    pf_is, _, _ = build_portfolio(is_close)
    pf_oos, _, _ = build_portfolio(oos_close)
    print("\n" + "=" * 70)
    print("PERFORMANCE BY SEGMENT (in-sample is where overfitting hides):")
    print_segment_summary("IN-SAMPLE (train)", pf_is)
    print_segment_summary("OUT-OF-SAMPLE (test)", pf_oos)

    print("\n" + "-" * 70)
    print("Note: paper-trade simulation only -- no real orders were placed. Fees +")
    print("slippage are charged on every fill. Judge the strategy on OUT-OF-SAMPLE,")
    print("not in-sample, and remember it has historically trailed buy-and-hold on")
    print("raw return while cutting drawdowns. See CLAUDE.md.")


if __name__ == "__main__":
    main()
