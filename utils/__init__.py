"""Shared helpers: data loaders and cost model.

The data loaders are imported LAZILY (via module __getattr__) so that importing
`utils` -- or a submodule like `utils.alpaca_client` -- does NOT pull in the heavy
optional dependencies the loaders need (ccxt for crypto, yfinance for stocks).
The live/paper bot (Raspberry Pi) installs only requirements-live.txt and never
calls the loaders; forcing those imports here would crash it on startup. Backtests
on the desktop still use `from utils import load_crypto / load_stock` unchanged --
the symbol is resolved on first access.
"""

from .costs import FEES, SLIPPAGE, INIT_CASH

__all__ = ["FEES", "SLIPPAGE", "INIT_CASH", "load_crypto", "load_stock"]


def __getattr__(name):
    # PEP 562 module-level lazy attribute access.
    if name == "load_crypto":
        from .crypto_loader import load_crypto
        return load_crypto
    if name == "load_stock":
        from .stock_loader import load_stock
        return load_stock
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
