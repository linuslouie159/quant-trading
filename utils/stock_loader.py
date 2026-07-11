"""Stock OHLCV loader via yfinance, with CSV caching.

Mirrors crypto_loader's interface and output shape (lowercase
open/high/low/close/volume, datetime index) so strategies are source-agnostic.
"""

from __future__ import annotations

import os
import time

import pandas as pd
import yfinance as yf

_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
_CACHE_TTL = 12 * 60 * 60


def _cache_path(symbol: str, interval: str) -> str:
    safe = symbol.replace("/", "-").replace("^", "")
    return os.path.join(_DATA_DIR, f"stock_{safe}_{interval}.csv")


def _is_fresh(path: str) -> bool:
    return os.path.exists(path) and (time.time() - os.path.getmtime(path)) < _CACHE_TTL


def load_stock(
    symbol: str = "SPY",
    start: str | None = "2015-01-01",
    end: str | None = None,
    interval: str = "1d",
    use_cache: bool = True,
) -> pd.DataFrame:
    """Load OHLCV stock/ETF candles via yfinance, cached to /data.

    Args:
        symbol: ticker, e.g. "SPY", "AAPL", "^GSPC".
        start, end: ISO date strings; end=None means up to today.
        interval: e.g. "1d", "1h", "1wk".
        use_cache: reuse a fresh CSV instead of re-downloading.

    Returns:
        DataFrame indexed by datetime with columns open/high/low/close/volume.
    """
    os.makedirs(_DATA_DIR, exist_ok=True)
    path = _cache_path(symbol, interval)

    if use_cache and _is_fresh(path):
        print(f"[stock_loader] cache hit -> {os.path.basename(path)}")
        return pd.read_csv(path, index_col="datetime", parse_dates=True)

    print(f"[stock_loader] downloading {symbol} {interval} from yfinance ...")
    df = yf.download(symbol, start=start, end=end, interval=interval, auto_adjust=True, progress=False)
    if df is None or df.empty:
        raise RuntimeError(f"yfinance returned no data for {symbol}")

    # yfinance may return a MultiIndex column frame for single tickers; flatten it.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]]
    df.index.name = "datetime"
    df.to_csv(path)
    print(f"[stock_loader] cached {len(df)} rows -> {os.path.basename(path)}")
    return df


if __name__ == "__main__":
    print(load_stock().tail())
