"""Central trading-cost assumptions.

This is the single source of truth for fees and slippage so that NO backtest can
silently run at zero cost. Import FEES and SLIPPAGE from here in every backtest.

Defaults are deliberately realistic (roughly Binance spot taker-fee territory):
  - FEES     = 0.1%  per trade  (entry and exit are each charged)
  - SLIPPAGE = 0.05% per trade  (price moves against you on fill)

Tune these per venue / asset / order-type. A maker-only strategy on a cheap
exchange might use lower fees; a thinly-traded altcoin should use HIGHER slippage.
Never set both to 0 to make a strategy "look good" -- see CLAUDE.md.
"""

FEES = 0.001       # 0.1% per trade
SLIPPAGE = 0.0005  # 0.05% per trade

# Convenience default starting capital for backtests.
INIT_CASH = 10_000
