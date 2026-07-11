# VectorBT Backtesting Project

A local, cost-aware, overfitting-resistant base for backtesting crypto and stock
trading strategies with [VectorBT](https://vectorbt.dev/). Guardrails (always model
costs, always split in-sample / out-of-sample) are baked in — see [CLAUDE.md](CLAUDE.md).

## Setup

Already done if you ran the setup, but to reproduce on a fresh machine (Windows, Python 3.12):

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

> The project calls the venv's interpreter directly (`.venv\Scripts\python.exe`)
> because a bare `python` on Windows is often just the Microsoft Store stub.

## Run a backtest

```powershell
.venv\Scripts\python.exe backtests\starter_backtest.py
```

This loads BTC/USDT daily data, runs a placeholder MA-crossover strategy with
realistic fees + slippage, and prints **in-sample** and **out-of-sample** stats
separately (total return, annualized return, Sharpe, max drawdown, win rate, # trades).

## Point it at a different coin

Edit the config block at the top of [backtests/starter_backtest.py](backtests/starter_backtest.py):

```python
SYMBOL = "ETH/USDT"   # any symbol your exchange lists
TIMEFRAME = "4h"      # "1d", "4h", "1h", ...
```

The crypto loader (`utils/crypto_loader.py`) tries Binance, then Kraken, then
Coinbase, and caches results to `/data`. For stocks/ETFs instead:

```python
from utils.stock_loader import load_stock
df = load_stock("AAPL", start="2018-01-01")
```

## Swap in your own strategy

1. Add a module under `strategies/`, e.g. `strategies/my_strategy.py`, exposing:

   ```python
   def generate_signals(close, **params):
       # ... compute boolean entries / exits aligned to `close`
       return entries, exits
   ```

2. In your backtest script, import it instead of `ma_crossover`:

   ```python
   from strategies.my_strategy import generate_signals
   ```

Everything downstream (costs, IS/OOS split, stats) stays the same.

## Project layout

| Path           | Purpose                                              |
|----------------|------------------------------------------------------|
| `utils/`       | Data loaders + `costs.py` (central fees/slippage)    |
| `strategies/`  | Strategy modules (`generate_signals` contract)       |
| `backtests/`   | Runnable backtest scripts                            |
| `data/`        | Cached OHLCV CSVs (gitignored)                       |
| `CLAUDE.md`    | Standing guardrails for the project                  |

## Guardrails (see CLAUDE.md)

- Fees + slippage modelled in **every** backtest — never zero-cost.
- In-sample vs out-of-sample always separated and labelled.
- A strategy is never "validated" on in-sample results alone.
- Overfitting risks are flagged proactively.
- **Backtesting / paper trading only** — no live real-money trading code unless explicitly requested.
