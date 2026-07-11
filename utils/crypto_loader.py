"""Crypto OHLCV loader via ccxt, with exchange fallback and CSV caching.

Pulls daily (or any-timeframe) candles for a symbol and caches them to /data as CSV
so we don't re-download on every run. If the primary exchange (Binance) is
geo-blocked or unreachable, it automatically falls back to the next exchange in the
list (Kraken, then Coinbase).
"""

from __future__ import annotations

import os
import time

import ccxt
import pandas as pd

# /data lives at the project root, one level up from /utils.
_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

# How long a cache file is considered "fresh" (seconds). 12h by default.
_CACHE_TTL = 12 * 60 * 60


def _cache_path(exchange: str, symbol: str, timeframe: str) -> str:
    safe = symbol.replace("/", "-")
    return os.path.join(_DATA_DIR, f"{exchange}_{safe}_{timeframe}.csv")


def _is_fresh(path: str) -> bool:
    return os.path.exists(path) and (time.time() - os.path.getmtime(path)) < _CACHE_TTL


def _fetch_from_exchange(exchange_id: str, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
    """Fetch `limit` candles from one exchange, paginating backwards via `since`."""
    ex = getattr(ccxt, exchange_id)({"enableRateLimit": True})
    ex.load_markets()

    if symbol not in ex.markets:
        raise ValueError(f"{exchange_id} does not list {symbol}")

    tf_ms = ex.parse_timeframe(timeframe) * 1000
    since = ex.milliseconds() - limit * tf_ms

    rows: list[list] = []
    while len(rows) < limit:
        batch = ex.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=1000)
        if not batch:
            break
        rows += batch
        since = batch[-1][0] + tf_ms
        if len(batch) < 1000:
            break  # reached the present
        time.sleep(ex.rateLimit / 1000)

    if not rows:
        raise RuntimeError(f"{exchange_id} returned no candles for {symbol}")

    df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df = df.drop_duplicates(subset="timestamp").sort_values("timestamp")
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.set_index("datetime").drop(columns="timestamp")
    return df.tail(limit)


def load_crypto(
    symbol: str = "BTC/USDT",
    timeframe: str = "1d",
    limit: int = 1000,
    exchanges: tuple[str, ...] = ("binance", "kraken", "coinbase"),
    use_cache: bool = True,
) -> pd.DataFrame:
    """Load OHLCV crypto candles, cached to /data.

    Args:
        symbol: e.g. "BTC/USDT", "ETH/USDT".
        timeframe: e.g. "1d", "4h", "1h".
        limit: number of most-recent candles to return.
        exchanges: ordered fallback list; the first reachable one wins.
        use_cache: if True, reuse a fresh CSV instead of hitting the network.

    Returns:
        DataFrame indexed by UTC datetime with columns open/high/low/close/volume.
    """
    os.makedirs(_DATA_DIR, exist_ok=True)

    # Reuse any fresh cache across the candidate exchanges.
    if use_cache:
        for ex_id in exchanges:
            path = _cache_path(ex_id, symbol, timeframe)
            if _is_fresh(path):
                print(f"[crypto_loader] cache hit -> {os.path.basename(path)}")
                return pd.read_csv(path, index_col="datetime", parse_dates=True)

    last_err: Exception | None = None
    for ex_id in exchanges:
        try:
            print(f"[crypto_loader] fetching {symbol} {timeframe} from {ex_id} ...")
            df = _fetch_from_exchange(ex_id, symbol, timeframe, limit)
            path = _cache_path(ex_id, symbol, timeframe)
            df.to_csv(path)
            print(f"[crypto_loader] cached {len(df)} candles -> {os.path.basename(path)}")
            return df
        except Exception as e:  # network / geo-block / unlisted symbol
            last_err = e
            print(f"[crypto_loader] {ex_id} failed ({e}); trying next exchange ...")

    raise RuntimeError(
        f"All exchanges failed for {symbol} {timeframe}. Last error: {last_err}. "
        "If you are geo-blocked from crypto exchanges, try the stock loader instead."
    )


def load_perp_range(
    symbol: str = "BTC/USDT",
    timeframe: str = "4h",
    start: str = "2022-09-01",
    end: str | None = None,
    exchange_id: str = "binanceusdm",
    use_cache: bool = True,
) -> pd.DataFrame:
    """Load OHLCV from a Binance USDT-M PERPETUAL feed over an explicit date range.

    Built for the Jesse cross-check: Jesse's trade log is on 'Binance Perpetual Futures',
    so the cross-check must use the SAME feed (ccxt 'binanceusdm'), not Binance spot.
    Forward-paginates from `start` to `end` so it reliably reaches back to 2022/2023.

    Returns a DataFrame indexed by UTC datetime with open/high/low/close/volume.
    """
    os.makedirs(_DATA_DIR, exist_ok=True)
    safe = symbol.replace("/", "-")
    path = os.path.join(_DATA_DIR, f"{exchange_id}_{safe}_{timeframe}_range.csv")
    if use_cache and _is_fresh(path):
        print(f"[crypto_loader] cache hit -> {os.path.basename(path)}")
        return pd.read_csv(path, index_col="datetime", parse_dates=True)

    ex = getattr(ccxt, exchange_id)({"enableRateLimit": True, "options": {"defaultType": "future"}})
    ex.load_markets()
    tf_ms = ex.parse_timeframe(timeframe) * 1000
    since = ex.parse8601(f"{start}T00:00:00Z")
    end_ms = ex.milliseconds() if end is None else ex.parse8601(f"{end}T00:00:00Z")

    rows: list[list] = []
    print(f"[crypto_loader] fetching {symbol} {timeframe} from {exchange_id} (perp) {start}->{end or 'now'} ...")
    page = 1000  # binanceusdm caps OHLCV limit at 1000
    while since < end_ms:
        batch = ex.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=page)
        if not batch:
            break
        rows += batch
        last_ts = batch[-1][0]
        since = last_ts + tf_ms
        # Stop only when we've passed `end` or the feed stops advancing.
        if last_ts >= end_ms or len(batch) < 2:
            break
        time.sleep(ex.rateLimit / 1000)

    if not rows:
        raise RuntimeError(f"{exchange_id} returned no candles for {symbol}")

    df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df = df.drop_duplicates(subset="timestamp").sort_values("timestamp")
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.set_index("datetime").drop(columns="timestamp")
    df = df[df.index < pd.to_datetime(end_ms, unit="ms", utc=True)] if end else df
    df.to_csv(path)
    print(f"[crypto_loader] cached {len(df)} perp candles -> {os.path.basename(path)}")
    return df


if __name__ == "__main__":
    print(load_crypto().tail())
