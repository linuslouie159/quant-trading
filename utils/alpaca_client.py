"""Alpaca PAPER-trading client -- hard-guarded against live trading.

SAFETY (see CLAUDE.md: "No live trading with real money")
---------------------------------------------------------
This module will ONLY connect to Alpaca's paper endpoint. If the configured base URL
is anything other than the paper host, `get_trading_client()` raises -- it cannot be
pointed at a real-money account by accident. There is intentionally no live-trading
code path here.

Keys are read from environment variables (loaded from a gitignored .env), never
hardcoded:
    ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL (optional, must be paper).
"""

from __future__ import annotations

import os

from dotenv import load_dotenv

# alpaca-py
from alpaca.trading.client import TradingClient
from alpaca.data.historical import StockHistoricalDataClient

load_dotenv()  # read .env at the project root, if present

PAPER_HOST = "paper-api.alpaca.markets"


def _require_paper_url(url: str) -> None:
    """Refuse anything that isn't the Alpaca paper endpoint."""
    if PAPER_HOST not in url:
        raise RuntimeError(
            f"REFUSING to connect: ALPACA_BASE_URL={url!r} is not the paper endpoint "
            f"({PAPER_HOST}). This project is paper-only (see CLAUDE.md). "
            "Live trading is not supported here."
        )


def _keys() -> tuple[str, str]:
    key = os.getenv("ALPACA_API_KEY")
    secret = os.getenv("ALPACA_SECRET_KEY")
    if not key or not secret:
        raise RuntimeError(
            "Missing ALPACA_API_KEY / ALPACA_SECRET_KEY. Copy .env.example to .env "
            "and paste your Alpaca PAPER keys (do not commit .env)."
        )
    return key, secret


def get_trading_client() -> TradingClient:
    """Return a paper-only TradingClient. Raises if pointed at a live endpoint."""
    url = os.getenv("ALPACA_BASE_URL", f"https://{PAPER_HOST}")
    _require_paper_url(url)
    key, secret = _keys()
    # paper=True is the belt-and-suspenders guard alongside the URL check above.
    return TradingClient(key, secret, paper=True)


def get_data_client() -> StockHistoricalDataClient:
    """Return a market-data client (data is read-only; no trading risk)."""
    key, secret = _keys()
    return StockHistoricalDataClient(key, secret)


if __name__ == "__main__":
    # Quick connectivity self-test: prints account status, never trades.
    tc = get_trading_client()
    acct = tc.get_account()
    print("Connected to Alpaca PAPER account.")
    print(f"  status        : {acct.status}")
    print(f"  cash          : ${float(acct.cash):,.2f}")
    print(f"  portfolio val : ${float(acct.portfolio_value):,.2f}")
    print(f"  buying power  : ${float(acct.buying_power):,.2f}")
