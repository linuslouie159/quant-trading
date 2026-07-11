"""Long-only trend-following filter -- a basket robustness strategy.

WHAT THIS IS
------------
A deliberately SIMPLE, low-turnover trend rule meant to be tested across a BASKET
of assets at once, not curve-fit to one. The whole point (per CLAUDE.md) is to see
whether a plain momentum rule survives fees AND survives out-of-sample across MANY
assets -- not to find the best parameters on one ticker.

RULE
----
- Regime filter: only hold when close > long SMA (`trend` window). This keeps you
  out during sustained downtrends, which is where trend-following earns its keep.
- Trigger: go long when the fast SMA crosses ABOVE the slow SMA *while* the regime
  filter is on. Exit when price closes below the long SMA OR the fast SMA crosses
  back below the slow SMA -- whichever comes first.
- Long-only. No leverage, no shorting, no pyramiding -- minimal free parameters
  (3 windows) to keep overfitting risk low relative to trade count.

Long-only + a slow trend window = low turnover, which matters at $1000 scale where
per-trade fees would eat a high-frequency strategy alive.

Contract (shared by all strategy modules):
    generate_signals(close, **params) -> (entries, exits)
where entries/exits are boolean pandas Series aligned to `close`.
"""

from __future__ import annotations

import pandas as pd


def generate_signals(
    close: pd.Series,
    fast: int = 50,
    slow: int = 100,
    trend: int = 200,
):
    """Long while in an uptrend regime; exit when the trend breaks.

    Args:
        close: price series (datetime-indexed).
        fast:  fast SMA window (trigger).
        slow:  slow SMA window (trigger); must be > fast.
        trend: long regime-filter SMA window; must be >= slow.

    Returns:
        (entries, exits): boolean Series aligned to `close`.
    """
    if fast >= slow:
        raise ValueError(f"fast ({fast}) must be < slow ({slow})")
    if trend < slow:
        raise ValueError(f"trend ({trend}) must be >= slow ({slow})")

    fast_ma = close.rolling(fast).mean()
    slow_ma = close.rolling(slow).mean()
    trend_ma = close.rolling(trend).mean()

    regime_on = close > trend_ma          # uptrend regime filter
    above = fast_ma > slow_ma             # fast/slow crossover state
    prev_above = above.shift(1, fill_value=False)

    crossed_up = above & ~prev_above
    crossed_down = ~above & prev_above

    # Enter on a fresh fast/slow cross-up that happens inside an uptrend regime.
    entries = crossed_up & regime_on

    # Exit on a fast/slow cross-down OR when price loses the long trend filter.
    exits = crossed_down | ~regime_on

    return entries.astype(bool), exits.astype(bool)
