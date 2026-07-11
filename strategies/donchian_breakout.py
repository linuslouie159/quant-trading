"""DonchianBreakoutPro -- VectorBT CROSS-CHECK port of a Jesse strategy.

WHY THIS EXISTS
---------------
The original strategy is written for the Jesse framework and its docstring claims
strong, RST-"validated" results (+79.2% train 2023 / +42.9% test 2024 on BTC-USDT 4h).
Per CLAUDE.md we do NOT take such numbers at face value. This module re-implements the
SAME logic on a completely different engine (VectorBT's custom order-function
simulation) so we can independently confirm or refute those numbers. Agreement across
two engines is weak evidence FOR; a large gap is evidence of overfitting, look-ahead, or
engine-specific artifacts. Either way this is a CROSS-CHECK, not a fresh validation of an
edge.

FAITHFULNESS / KNOWN DIVERGENCES (see CLAUDE.md)
------------------------------------------------
- Stops are bar-resolution: a stop/trail is "hit" when the 4h bar's high/low pierces the
  level, filled at the stop price. Jesse fills intrabar at finer resolution. This is the
  closest faithful match within VectorBT's bar model, but fills will differ slightly.
- `risk_to_qty` margin math is reproduced from Jesse's formula, not Jesse internals.
- Pyramid adds are evaluated once per 4h bar.
- One order per bar: if multiple actions are eligible on the same bar we PRIORITIZE
  stop-exit > pyramid-add > new-entry (exits first = conservative).

STRATEGY LOGIC (mapped from the Jesse source)
---------------------------------------------
- ENTRY  : close breaks the prior `entry_period`-bar high (long) / low (short).
- SIZE   : risk_to_qty -> each unit sized so (entry-stop) distance == risk_per_unit_pct
           of current equity. Initial stop = atr_stop_mult * ATR.
- PYRAMID: add a unit each time price advances add_step_atr*ATR in favor, up to max_units.
- EXIT   : trailing stop = opposite exit_period Donchian band, ratcheted (never loosened);
           also the initial ATR stop. No fixed take-profit.
- Long AND short; futures, 2x leverage handled at the portfolio level.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numba import njit

import vectorbt as vbt
from vectorbt.portfolio import nb as pf_nb
from vectorbt.portfolio.enums import Direction, SizeType

# Default params = Jesse docstring "candidate #10".
DEFAULT_PARAMS = dict(
    entry_period=33,
    exit_period=13,
    atr_stop_mult=1.5,
    add_step_atr=2.0,
    max_units=2,
    risk_per_unit_pct=2.0,
    atr_period=14,
)


# --------------------------------------------------------------------------- #
# Indicators (precomputed in pandas/numpy; NOT inside numba)                   #
# --------------------------------------------------------------------------- #
def compute_indicators(df: pd.DataFrame, entry_period: int, exit_period: int, atr_period: int):
    """Return numpy arrays aligned to df rows.

    All channel values are the PRIOR-bar Donchian (shifted by 1) so a breakout is
    compared against bands that exclude the current bar -- this avoids look-ahead, the
    same intent as Jesse indexing `upperband[-2]`.
    """
    open_, high, low, close = df["open"], df["high"], df["low"], df["close"]

    # Prior N-bar Donchian for ENTRY (exclude current bar via shift(1)).
    entry_upper = high.rolling(entry_period).max().shift(1)
    entry_lower = low.rolling(entry_period).min().shift(1)

    # Prior N-bar Donchian for EXIT trailing band.
    exit_upper = high.rolling(exit_period).max().shift(1)
    exit_lower = low.rolling(exit_period).min().shift(1)

    # Wilder ATR.
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    atr = tr.ewm(alpha=1 / atr_period, adjust=False, min_periods=atr_period).mean()

    # Jesse fill convention: a breakout is DETECTED on the close of bar t-1
    # (close[t-1] vs the Donchian band that excludes bar t-1), then the market order
    # fills at the OPEN of bar t. So the signal active at bar i is the prior bar's
    # breakout, and we fill at open[i]. ATR/stop also reference the prior bar (atr[i-1]).
    broke_up = (close > entry_upper).astype(bool)        # breakout on this bar's close
    broke_down = (close < entry_lower).astype(bool)
    long_sig = broke_up.shift(1, fill_value=False)   # -> act at next bar's open
    short_sig = broke_down.shift(1, fill_value=False)
    atr_prev = atr.shift(1)                       # ATR known at signal time

    def arr(s):
        return s.to_numpy(dtype=np.float64)

    return (
        arr(entry_upper), arr(entry_lower),
        arr(exit_upper), arr(exit_lower),
        arr(atr),
        arr(high), arr(low), arr(close),
        arr(open_),
        long_sig.to_numpy(dtype=np.bool_),
        short_sig.to_numpy(dtype=np.bool_),
        arr(atr_prev),
    )


@njit(cache=True)
def _risk_to_qty(capital, risk_pct, entry, stop, fee_rate, leverage):
    """EXACT port of Jesse utils.risk_to_qty (verified against jesse/utils.py source):

        risk_per_qty = |entry - stop|
        size = (risk_pct/100 * capital) / risk_per_qty * entry      # risk-based notional
        size = min(size, capital * leverage)                        # capped at buying power
        size = size * (1 - fee_rate * 3)                            # Jesse's fee haircut
        qty  = size / entry

    The min() cap is the key mechanic the previous version MISSED: for tight-ish ATR
    stops the risk formula wants more than the account can buy, so the position is
    capped at capital*leverage notional. On trade #1 this cap (1x notional, $10k) is
    exactly what produced Jesse's 0.2255 BTC unit -- not the risk formula.
    """
    risk_per_qty = abs(entry - stop)
    if risk_per_qty <= 0.0 or entry <= 0.0 or capital <= 0.0:
        return 0.0
    size = (risk_pct / 100.0) * capital / risk_per_qty * entry
    cap = capital * leverage
    if size > cap:
        size = cap
    size = size * (1.0 - fee_rate * 3.0)
    if size <= 0.0:
        return 0.0
    return size / entry


# --------------------------------------------------------------------------- #
# Simulation state + numba order functions                                    #
# --------------------------------------------------------------------------- #
# State layout (one column -> one BTC series, so a length-N vector per field):
#   units[col]          : pyramid units currently on (0 = flat)
#   stop_price[col]     : current (ratcheted) stop level
#   last_add_price[col] : price at which the last unit was added
#   dir[col]            : +1 long, -1 short, 0 flat

# State propagation note (verified against vectorbt/portfolio/nb.py simulate_nb):
#   pre_group_out  = pre_group_func_nb(ctx, *pre_sim_out, *pre_group_args)
#   pre_segment_out = pre_segment_func_nb(ctx, *pre_group_out, *pre_segment_args)
#   order          = order_func_nb(ctx, *pre_segment_out, *order_args)
# The default group func would DROP pre_sim_out, so we allocate persistent state in
# pre_group_func_nb and pass it through pre_segment_func_nb to the order func.
@njit(cache=True)
def _pre_group_func_nb(c):
    n_cols = c.target_shape[1]
    units = np.zeros(n_cols, dtype=np.float64)
    stop_price = np.full(n_cols, np.nan, dtype=np.float64)
    last_add_price = np.full(n_cols, np.nan, dtype=np.float64)
    pos_dir = np.zeros(n_cols, dtype=np.float64)
    return (units, stop_price, last_add_price, pos_dir)


@njit(cache=True)
def _pre_segment_func_nb(c, units, stop_price, last_add_price, pos_dir, close):
    # Value each column at the current close so equity/risk sizing is consistent,
    # then forward the persistent state arrays to the order func.
    for col in range(c.from_col, c.to_col):
        c.last_val_price[col] = close[c.i]
    return (units, stop_price, last_add_price, pos_dir)


@njit(cache=True)
def _order_func_nb(
    c,
    units, stop_price, last_add_price, pos_dir,  # from pre_segment_out
    entry_upper, entry_lower, exit_upper, exit_lower, atr, high, low, close,  # order_args
    open_, long_sig, short_sig, atr_prev,
    atr_stop_mult, add_step_atr, max_units, risk_pct, fee_rate, slippage, leverage,
):
    col = c.col
    i = c.i
    o = open_[i]               # entries/adds fill at the bar OPEN (Jesse market order)
    a_prev = atr_prev[i]       # ATR known at signal time (prior bar)

    if np.isnan(a_prev) or np.isnan(exit_upper[i]):
        return pf_nb.order_nothing_nb()  # warm-up: indicators not ready

    pos = c.position_now
    d = pos_dir[col]

    # ---------- 1) STOP / TRAIL EXIT (highest priority) ------------------- #
    # Exits still resolve against THIS bar's high/low (intrabar pierce), filled at stop.
    if pos != 0.0 and d != 0.0:
        if d > 0.0:  # long: ratchet stop up to exit_lower band; hit if bar low pierces
            new_stop = stop_price[col]
            if exit_lower[i] > new_stop:
                new_stop = exit_lower[i]
            stop_price[col] = new_stop
            if low[i] <= new_stop:
                units[col] = 0.0
                last_add_price[col] = np.nan
                pos_dir[col] = 0.0
                fill = new_stop if new_stop < o else o  # gap-safe
                return pf_nb.close_position_nb(price=fill, fees=fee_rate, slippage=slippage)
        else:  # short: ratchet stop down to exit_upper band; hit if bar high pierces
            new_stop = stop_price[col]
            if exit_upper[i] < new_stop:
                new_stop = exit_upper[i]
            stop_price[col] = new_stop
            if high[i] >= new_stop:
                units[col] = 0.0
                last_add_price[col] = np.nan
                pos_dir[col] = 0.0
                fill = new_stop if new_stop > o else o
                return pf_nb.close_position_nb(price=fill, fees=fee_rate, slippage=slippage)

    # ---------- 2) PYRAMID ADD (fills at bar open) ------------------------ #
    if pos != 0.0 and d != 0.0 and units[col] < max_units:
        step = add_step_atr * a_prev
        if d > 0.0 and o >= last_add_price[col] + step:
            stop = o - atr_stop_mult * a_prev
            qty = _risk_to_qty(c.value_now, risk_pct, o, stop, fee_rate, leverage)
            if qty > 0.0:
                units[col] += 1.0
                last_add_price[col] = o
                return pf_nb.order_nb(
                    size=qty, price=o, size_type=SizeType.Amount,
                    direction=Direction.Both, fees=fee_rate, slippage=slippage,
                )
        if d < 0.0 and o <= last_add_price[col] - step:
            stop = o + atr_stop_mult * a_prev
            qty = _risk_to_qty(c.value_now, risk_pct, o, stop, fee_rate, leverage)
            if qty > 0.0:
                units[col] += 1.0
                last_add_price[col] = o
                return pf_nb.order_nb(
                    size=-qty, price=o, size_type=SizeType.Amount,
                    direction=Direction.Both, fees=fee_rate, slippage=slippage,
                )

    # ---------- 3) NEW ENTRY (only when flat; prior-bar breakout, fill at open) -- #
    if pos == 0.0:
        if long_sig[i]:
            stop = o - atr_stop_mult * a_prev
            qty = _risk_to_qty(c.value_now, risk_pct, o, stop, fee_rate, leverage)
            if qty > 0.0:
                units[col] = 1.0
                last_add_price[col] = o
                stop_price[col] = stop
                pos_dir[col] = 1.0
                return pf_nb.order_nb(
                    size=qty, price=o, size_type=SizeType.Amount,
                    direction=Direction.Both, fees=fee_rate, slippage=slippage,
                )
        elif short_sig[i]:
            stop = o + atr_stop_mult * a_prev
            qty = _risk_to_qty(c.value_now, risk_pct, o, stop, fee_rate, leverage)
            if qty > 0.0:
                units[col] = 1.0
                last_add_price[col] = o
                stop_price[col] = stop
                pos_dir[col] = -1.0
                return pf_nb.order_nb(
                    size=-qty, price=o, size_type=SizeType.Amount,
                    direction=Direction.Both, fees=fee_rate, slippage=slippage,
                )

    return pf_nb.order_nothing_nb()


# --------------------------------------------------------------------------- #
# Public entry point                                                          #
# --------------------------------------------------------------------------- #
def run_backtest(df: pd.DataFrame, *, init_cash: float, fee_rate: float, slippage: float,
                 leverage: float = 2.0, params: dict | None = None):
    """Run the full DonchianBreakoutPro simulation over `df` (OHLCV, datetime index).

    Returns a vectorbt Portfolio whose returns are measured against the ACTUAL deposited
    `init_cash` (e.g. 10_000) -- NOT an inflated base -- so total_return is honest.

    Leverage is modelled the way Jesse does it: inside `_risk_to_qty`, position notional
    is capped at `capital * leverage` (2x buying power), and returns are measured against
    the real deposited `init_cash`. Entries and pyramid adds fill at the bar OPEN (Jesse
    market-on-signal-open); stops resolve intrabar against high/low.
    """
    p = {**DEFAULT_PARAMS, **(params or {})}
    ind = compute_indicators(df, p["entry_period"], p["exit_period"], p["atr_period"])
    # Pass an INDEXED close Series so the portfolio wrapper keeps the datetime index
    # (used for per-year slicing). The numba order func indexes the numpy arrays in
    # `ind` positionally by bar.
    close_series = df["close"].astype(np.float64)

    pf = vbt.Portfolio.from_order_func(
        close_series,
        _order_func_nb,
        # ---- order_args: per-bar arrays (incl. open/signals/atr_prev), then scalars ----
        *ind,
        np.float64(p["atr_stop_mult"]),
        np.float64(p["add_step_atr"]),
        np.float64(p["max_units"]),
        np.float64(p["risk_per_unit_pct"]),
        np.float64(fee_rate),
        np.float64(slippage),
        np.float64(leverage),
        pre_group_func_nb=_pre_group_func_nb,
        pre_segment_func_nb=_pre_segment_func_nb,
        pre_segment_args=(ind[7],),  # numpy close array; appended AFTER state arrays
        init_cash=init_cash,  # returns measured against real deposited capital
        freq="4h",
    )
    return pf
