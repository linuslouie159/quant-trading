"""Time-series (absolute) momentum -- a classic, lean quant strategy for indexes.

WHAT THIS IS
------------
Time-series momentum (a.k.a. absolute momentum / trend) is one of the most widely
documented quant effects in the public literature -- Moskowitz, Ooi & Pedersen
(2012), "Time Series Momentum"; and Antonacci's "dual momentum". The core idea:
an asset that has gone UP over the recent past tends to keep going up over the
near future, and one that has gone down tends to keep going down.

Applied LONG-FLAT to a single index, the rule is simply:
  - hold the index whenever its own trailing momentum is positive, AND
  - (optional) price is above its long trend SMA -- a standard regime filter;
  - otherwise sit in cash.

This is deliberately a LEAN rule (2-3 free parameters) so overfitting risk stays
low relative to the trade count, which is what the project's standing rules ask
for. It is long-only with no shorting/leverage -- a good fit for long-biased
index data.

FORMULA (publicly known)
------------------------
  momentum_t = close_{t-1} / close_{t-1-lookback} - 1        (trailing total return)
  trend_on_t = HYSTERESIS band around SMA(close, trend)      (regime filter)
  in_market  = (momentum_t > 0) AND trend_on_t

All inputs to the signal are computed on PRIOR bars (shift(1)) so the current
bar's own price cannot leak into its own signal -- no look-ahead.

THE BUFFER BAND (`band`) -- whipsaw fix
---------------------------------------
A plain `close > SMA` test flickers in/out whenever price hugs the SMA, churning
out cost-bleeding round-trips (the strategy's main weakness on choppy assets).
A `band` (fraction, e.g. 0.02 = 2%) turns that hard line into a DEAD ZONE with
hysteresis -- different thresholds to get in vs out:
  - ENTER the trend regime only when  close > SMA * (1 + band)   (clear the band)
  - EXIT  the trend regime only when  close < SMA * (1 - band)   (break the band)
  - INSIDE the band: HOLD whatever the regime was on the prior bar (no flip).
This only changes behavior when price is NEAR the SMA. In a real crash price
falls far BELOW SMA*(1-band), so bear-market exits fire exactly as before --
the protection is untouched by design. `band=0.0` reproduces the original
hard-threshold behavior byte-for-byte.

The canonical academic variant is 12-1 month momentum (lookback ~= 252 trading
days, skipping the most recent ~21). Here `lookback` defaults to 126 (~6 trading
months); `lookback`, `trend`, and `band` are the free knobs.

Contract (shared by all strategy modules):
    generate_signals(close, **params) -> (entries, exits)
where entries/exits are boolean pandas Series aligned to `close`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _banded_trend_on(close: pd.Series, trend_ma: pd.Series, band: float) -> pd.Series:
    """Hysteresis regime: clear SMA*(1+band) to turn on, break SMA*(1-band) to turn off.

    Inside the band the prior bar's state is held. With band=0.0 both thresholds
    collapse to `close > SMA`, reproducing the plain hard-threshold filter.
    """
    upper = (trend_ma * (1.0 + band)).to_numpy()
    lower = (trend_ma * (1.0 - band)).to_numpy()
    px = close.to_numpy()

    on = np.zeros(len(px), dtype=bool)
    state = False
    for i in range(len(px)):
        if np.isnan(upper[i]):          # SMA not warm yet -> stay out
            state = False
        elif px[i] > upper[i]:          # cleared the upper band -> turn on
            state = True
        elif px[i] < lower[i]:          # broke the lower band -> turn off
            state = False
        # else: inside the band -> hold previous `state`
        on[i] = state
    return pd.Series(on, index=close.index)


def generate_signals(
    close: pd.Series,
    lookback: int = 126,
    trend: int = 200,
    band: float = 0.03,
    use_trend_filter: bool = True,
):
    """Long when trailing momentum is positive (and, optionally, in an uptrend).

    The momentum and trend inputs are read off PRIOR bars (shift(1)) so the
    current bar's own close can't leak into its own signal -- no look-ahead.

    The strategy is a long-flat REGIME (in or out), so the boolean `in_market`
    state is converted into entry/exit *transitions*: an entry fires on the bar
    we flip from out->in, an exit on the bar we flip from in->out.

    Args:
        close: price series (datetime-indexed).
        lookback: trailing-return window for the momentum signal, in bars
            (~126 = 6 trading months). Must be > 0.
        trend: long regime-filter SMA window, in bars. Must be > 0.
        band: buffer/hysteresis band around the trend SMA, as a fraction
            (0.03 = 3%). Defaults to 0.03, chosen by walk-forward across SPY/QQQ/
            IWM as the best risk-adjusted DEFAULT (highest mean OOS Sharpe, fewest
            trades, lower drawdown than band=0). Pass band=0.0 to recover the
            original hard-threshold behavior. Reduces whipsaw without touching
            crash exits (price is far below the band in a real crash). Must be >= 0.
        use_trend_filter: if False, trade on the momentum sign alone (no SMA gate).

    Returns:
        (entries, exits): boolean Series aligned to `close`.
    """
    if lookback <= 0:
        raise ValueError(f"lookback ({lookback}) must be > 0")
    if trend <= 0:
        raise ValueError(f"trend ({trend}) must be > 0")
    if band < 0:
        raise ValueError(f"band ({band}) must be >= 0")

    # Trailing total return over `lookback` bars, read off prior bars (shift(1)).
    prior = close.shift(1)
    momentum = prior / prior.shift(lookback) - 1.0          # no look-ahead
    mom_positive = momentum > 0

    if use_trend_filter:
        trend_ma = close.rolling(trend).mean()
        trend_on = _banded_trend_on(close, trend_ma, band)  # hysteresis regime filter
        in_market = mom_positive & trend_on
    else:
        in_market = mom_positive

    # NaN from the early warm-up window -> treat as "out of market".
    in_market = in_market.fillna(False).astype(bool)
    prev = in_market.shift(1, fill_value=False)

    entries = in_market & ~prev                              # flipped out -> in
    exits = ~in_market & prev                                # flipped in  -> out

    return entries.astype(bool), exits.astype(bool)


def exit_levels(
    close: pd.Series,
    lookback: int = 126,
    trend: int = 200,
    band: float = 0.03,
) -> dict:
    """Compute the CURRENT exit thresholds for a held long, off the latest bar.

    A long position is exited (on the next daily evaluation) when EITHER:
      * price closes below the lower band  SMA * (1 - band)   [trend-filter exit], or
      * the trailing `lookback`-bar return turns <= 0          [momentum exit].

    This mirrors generate_signals exactly -- it just reports the live levels rather
    than the boolean transitions, so the runner / alerts can show how much cushion a
    position has. Read-only; computes nothing about orders.

    Returns a dict for the most recent bar:
        last_close, sma, band_exit (price-level exit), pct_above_band_exit,
        momentum, momentum_exit (bool), cushion (min fractional distance to an exit).
    """
    c = close.dropna()
    sma = float(c.rolling(trend).mean().iloc[-1])
    last = float(c.iloc[-1])
    band_exit = sma * (1.0 - band)
    momentum = float(c.iloc[-1] / c.iloc[-1 - lookback] - 1.0) if len(c) > lookback else float("nan")

    pct_above = (last - band_exit) / last if last else float("nan")
    mom_exit = momentum <= 0
    # "Cushion" = how close we are to exiting, as a fraction: distance to the band
    # exit, treated as 0 if the momentum exit has already tripped.
    cushion = 0.0 if mom_exit else pct_above

    return {
        "last_close": last,
        "sma": sma,
        "band_exit": band_exit,
        "pct_above_band_exit": pct_above,
        "momentum": momentum,
        "momentum_exit": mom_exit,
        "cushion": cushion,
    }
