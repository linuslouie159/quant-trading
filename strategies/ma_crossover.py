"""Moving-average crossover -- PLACEHOLDER STRATEGY (pipeline demo only).

This is NOT a validated or profitable edge. It exists to exercise the full
backtest pipeline end to end (load -> signals -> costs -> IS/OOS split -> stats).
A simple SMA crossover is one of the most over-fit, widely-known patterns in
trading; treat any good-looking result here with heavy skepticism. See CLAUDE.md.

Contract (shared by all strategy modules):
    generate_signals(close, **params) -> (entries, exits)
where entries/exits are boolean pandas Series aligned to `close`.
"""

from __future__ import annotations

import pandas as pd


def generate_signals(close: pd.Series, fast: int = 20, slow: int = 50):
    """Long when the fast SMA crosses ABOVE the slow SMA; exit on the reverse cross.

    Args:
        close: price series (datetime-indexed).
        fast: fast SMA window.
        slow: slow SMA window (must be > fast).

    Returns:
        (entries, exits): boolean Series aligned to `close`.
    """
    if fast >= slow:
        raise ValueError(f"fast ({fast}) must be < slow ({slow})")

    fast_ma = close.rolling(fast).mean()
    slow_ma = close.rolling(slow).mean()

    above = fast_ma > slow_ma
    prev_above = above.shift(1, fill_value=False)

    entries = above & ~prev_above   # crossed up this bar
    exits = ~above & prev_above      # crossed down this bar

    # Bars before `slow` is reached are NaN comparisons -> False; ensure bool dtype.
    return entries.astype(bool), exits.astype(bool)
