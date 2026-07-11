"""Trend + momentum breakout -- a compact trend-following strategy.

WHAT THIS IS
------------
A long-only trend strategy with a SMALL parameter set, designed to be tuned ONLY on
in-sample data and then judged on out-of-sample data it never saw (see the matching
backtest in backtests/aggressive_trend_grow.py). The point (per CLAUDE.md) is to avoid
"only working on info you already know": the optimizer never touches the test window.

RULE
----
- Regime filter: only trade long when close > `trend` SMA (stay out of downtrends --
  where trend-following bleeds).
- Trigger: ENTER long when close makes a new `breakout`-bar high (Donchian-style upside
  breakout) WHILE the regime filter is on. This is the "momentum/trend kicks in" event.
- EXIT when close drops below the `exit_n`-bar low (Donchian trailing stop) OR price
  closes back below the `trend` SMA -- whichever comes first.
- Long-only, no leverage, no shorting, no pyramiding. Three free parameters
  (breakout, exit_n, trend) to keep overfitting risk low relative to trade count.

Aggressiveness at $1500 scale is handled at the PORTFOLIO/sizing level in the backtest
(position sizing), NOT by stuffing the rule with parameters. A leaner rule generalizes
to unseen data better -- which is exactly what "don't build on past data" asks for.

Contract (shared by all strategy modules):
    generate_signals(close, **params) -> (entries, exits)
where entries/exits are boolean pandas Series aligned to `close`.
"""

from __future__ import annotations

import pandas as pd


def generate_signals(
    close: pd.Series,
    breakout: int = 20,
    exit_n: int = 10,
    trend: int = 100,
):
    """Long on an upside breakout inside an uptrend; exit on a downside breakout.

    All channels are computed on PRIOR bars (shift(1)) so the current bar's own value
    can't leak into its own signal -- no look-ahead.

    Args:
        close: price series (datetime-indexed).
        breakout: lookback for the entry (new N-bar high) channel.
        exit_n:   lookback for the exit (new N-bar low) channel; should be < breakout.
        trend:    long regime-filter SMA window; should be >= breakout.

    Returns:
        (entries, exits): boolean Series aligned to `close`.
    """
    if exit_n >= breakout:
        raise ValueError(f"exit_n ({exit_n}) should be < breakout ({breakout})")
    if trend < breakout:
        raise ValueError(f"trend ({trend}) should be >= breakout ({breakout})")

    # Prior-bar Donchian channels (exclude the current bar via shift(1)).
    upper = close.rolling(breakout).max().shift(1)   # highest close of prior N bars
    lower = close.rolling(exit_n).min().shift(1)      # lowest close of prior N bars
    trend_ma = close.rolling(trend).mean()

    regime_on = close > trend_ma                      # uptrend regime filter
    broke_up = close > upper                          # new N-bar high -> momentum entry
    broke_down = close < lower                        # new N-bar low  -> trailing exit

    entries = broke_up & regime_on
    exits = broke_down | ~regime_on

    return entries.astype(bool), exits.astype(bool)


def exit_levels(
    close,
    breakout: int = 20,
    exit_n: int = 10,
    trend: int = 100,
) -> dict:
    """Compute the CURRENT exit thresholds for a held long, off the latest bar.

    A long is exited (on the next daily evaluation) when EITHER:
      * price closes below the prior `exit_n`-bar low  [Donchian trailing-stop exit], or
      * price closes below the `trend` SMA              [regime-filter exit].

    This mirrors generate_signals' exit rule (exits = broke_down | ~regime_on); it
    just reports the live levels rather than the boolean transitions. The exit price
    that matters is the HIGHER of the two stops (whichever price would hit first on
    the way down). Read-only; computes nothing about orders.

    Returns a dict for the most recent bar:
        last_close, sma, donchian_low (prior exit_n-bar low), band_exit (the binding
        stop = max(donchian_low, sma)), pct_above_band_exit, momentum_exit (False --
        this strategy has no momentum gate; kept for a uniform shape with ts_momentum),
        cushion (fractional distance to the binding stop).
    """
    c = close.dropna()
    last = float(c.iloc[-1])
    sma = float(c.rolling(trend).mean().iloc[-1])
    # Prior exit_n-bar low EXCLUDING the current bar (shift(1)), matching the signal.
    donchian_low = float(c.rolling(exit_n).min().shift(1).iloc[-1])

    # Either stop triggers the exit, so the position survives only while price is above
    # BOTH -> the binding (closer) stop is the higher of the two.
    binding = max(donchian_low, sma)
    pct_above = (last - binding) / last if last else float("nan")

    return {
        "last_close": last,
        "sma": sma,
        "donchian_low": donchian_low,
        "band_exit": binding,          # name reused so notify._exit_cushion_block is generic
        "pct_above_band_exit": pct_above,
        "momentum": float("nan"),      # no momentum gate in this strategy
        "momentum_exit": False,
        "cushion": pct_above,
    }
