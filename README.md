# quant-trading — momentum backtester + live paper-trading bot

A cost-aware, overfitting-resistant framework for backtesting crypto and stock trading
strategies — with a companion bot that runs the validated strategy on a **paper account**
and sends daily signals over Telegram. Built for people who want to test a trading idea
*honestly*: fees and slippage in every run, in-sample and out-of-sample always separated.

**Built with:** Python 3.12 · [VectorBT](https://vectorbt.dev/) · pandas · ccxt (exchange data) · a Telegram bot for live alerts.

**What I learned / decisions:** The hard part of a backtester isn't the maths — it's not
lying to yourself. Every guardrail here exists because it's easy to get a beautiful equity
curve that means nothing: zero-cost fills, or "validating" on the same data you tuned on.
So costs are modelled centrally in `utils/costs.py` and can never be skipped, the
in-sample / out-of-sample split is enforced and labelled on every result, and strategies
plug in through a single `generate_signals()` contract so swapping ideas doesn't touch the
plumbing. See [CLAUDE.md](CLAUDE.md) for the full set of standing guardrails.

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
