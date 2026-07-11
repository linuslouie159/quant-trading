"""Alpaca PAPER-trading client for the ts_momentum strategy -- ISOLATED account.

WHY A SEPARATE CLIENT
---------------------
The main utils/alpaca_client.py reads ALPACA_* env vars (the $1,500 paper account
tied to a DIFFERENT, already-automated strategy). This module is deliberately kept
apart: it loads keys ONLY from .env.tsmom and reads the TSMOM_ALPACA_* variables, so
the ts_momentum runner can never accidentally connect to the other strategy's account.

SAFETY (see CLAUDE.md: "No live trading with real money")
---------------------------------------------------------
Paper endpoint ONLY. If the configured base URL is anything other than the paper
host, the client refuses to connect. There is no live-trading code path here.

Keys are read from a gitignored .env.tsmom, never hardcoded:
    TSMOM_ALPACA_API_KEY, TSMOM_ALPACA_SECRET_KEY, TSMOM_ALPACA_BASE_URL (optional).
"""

from __future__ import annotations

import os

from dotenv import load_dotenv

# alpaca-py
from alpaca.trading.client import TradingClient
from alpaca.data.historical import StockHistoricalDataClient

# Load ONLY the ts_momentum env file -- not the main .env. This is the isolation
# boundary: the TSMOM_* keys live here and nowhere else.
_ENV_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env.tsmom"
)
load_dotenv(_ENV_PATH)

PAPER_HOST = "paper-api.alpaca.markets"


def _require_paper_url(url: str) -> None:
    """Refuse anything that isn't the Alpaca paper endpoint."""
    if PAPER_HOST not in url:
        raise RuntimeError(
            f"REFUSING to connect: TSMOM_ALPACA_BASE_URL={url!r} is not the paper "
            f"endpoint ({PAPER_HOST}). This project is paper-only (see CLAUDE.md). "
            "Live trading is not supported here."
        )


def _keys() -> tuple[str, str]:
    key = os.getenv("TSMOM_ALPACA_API_KEY")
    secret = os.getenv("TSMOM_ALPACA_SECRET_KEY")
    if not key or not secret:
        raise RuntimeError(
            "Missing TSMOM_ALPACA_API_KEY / TSMOM_ALPACA_SECRET_KEY. Copy "
            ".env.tsmom.example to .env.tsmom and paste the PAPER keys for your "
            "DEDICATED ts_momentum account (do not commit .env.tsmom)."
        )
    return key, secret


# Tripwire: the OTHER strategy's account is a $1,500 paper account. If the keys in
# .env.tsmom ever resolve to an account at (or near) that equity, we almost certainly
# pasted the wrong keys -- refuse to proceed rather than risk trading the wrong book.
# Set TSMOM_BLOCK_EQUITY to the other account's known balance (default 1500).
_BLOCK_EQUITY = float(os.getenv("TSMOM_BLOCK_EQUITY", "1500"))
_BLOCK_TOL = 0.01  # within 1% of the blocked figure counts as a match


def _assert_not_other_account(trading_client) -> None:
    """Abort if the connected account's equity matches the other strategy's book."""
    equity = float(trading_client.get_account().portfolio_value)
    if abs(equity - _BLOCK_EQUITY) <= _BLOCK_TOL * max(_BLOCK_EQUITY, 1.0):
        raise RuntimeError(
            f"SAFETY TRIPWIRE: connected account equity ${equity:,.2f} matches the "
            f"blocked ${_BLOCK_EQUITY:,.2f} figure -- this looks like the OTHER "
            "strategy's account, not the dedicated ts_momentum account. Refusing to "
            "proceed. Check the keys in .env.tsmom. (If the new account legitimately "
            "holds this amount, set TSMOM_BLOCK_EQUITY in .env.tsmom to a different "
            "value.)"
        )


def get_trading_client() -> TradingClient:
    """Return a paper-only TradingClient for the ts_momentum account.

    Guards: paper-endpoint URL check, paper=True, and an equity tripwire that refuses
    to return a client pointed at the other strategy's $1,500 account.
    """
    url = os.getenv("TSMOM_ALPACA_BASE_URL", f"https://{PAPER_HOST}")
    _require_paper_url(url)
    key, secret = _keys()
    # paper=True is the belt-and-suspenders guard alongside the URL check above.
    client = TradingClient(key, secret, paper=True)
    _assert_not_other_account(client)
    return client


def get_data_client() -> StockHistoricalDataClient:
    """Return a market-data client (data is read-only; no trading risk)."""
    key, secret = _keys()
    return StockHistoricalDataClient(key, secret)


if __name__ == "__main__":
    # Quick connectivity self-test for the ts_momentum account: never trades.
    tc = get_trading_client()
    acct = tc.get_account()
    print("Connected to Alpaca PAPER account (ts_momentum, isolated).")
    print(f"  status        : {acct.status}")
    print(f"  cash          : ${float(acct.cash):,.2f}")
    print(f"  portfolio val : ${float(acct.portfolio_value):,.2f}")
    print(f"  buying power  : ${float(acct.buying_power):,.2f}")
