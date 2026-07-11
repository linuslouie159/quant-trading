# CLAUDE.md — Standing rules for this VectorBT backtesting project

This project is for **backtesting and paper trading only**. Follow these rules in
every piece of work, without being reminded.

## Backtesting integrity (non-negotiable)
1. **Always model fees and slippage in every backtest.** Import `FEES` and
   `SLIPPAGE` from `utils/costs.py` and pass them to `Portfolio.from_signals`
   (or equivalent). **Never report zero-cost results.** If you genuinely need a
   frictionless sanity check, label it explicitly as "ZERO-COST SANITY CHECK —
   not a real result."
2. **Always separate in-sample (train) from out-of-sample (test) results, and
   label them clearly** (`=== IN-SAMPLE ===` / `=== OUT-OF-SAMPLE ===`). Report
   stats for each segment separately.
3. **Never describe a strategy as "validated", "profitable", or "working" based on
   in-sample results alone.** In-sample is where overfitting hides. Only
   out-of-sample (and ideally walk-forward / multi-asset) evidence supports such
   claims — and even then, hedge.

## Overfitting vigilance
Proactively flag overfitting risk when you see:
- Too many free parameters relative to the number of trades.
- Suspiciously high returns or Sharpe (e.g. Sharpe > 3 on daily data).
- Results that only work on **one coin** or **one time period** and collapse
  elsewhere.
- A large in-sample / out-of-sample performance gap.
Say so plainly in your summary, even when not asked.

## Safety
- **No live trading with real money.** Do not write, wire up, or enable code that
  places real orders on a live account unless the user **explicitly** asks for it
  in that message. ccxt/exchange API keys, "live"/"production" trade execution,
  and withdrawal calls are off-limits by default.

## Environment / how to run
- Python interpreter: **`py -3.12`** launcher; project uses a venv at `.venv`.
  Call the venv directly — bare `python` on PATH is the Windows Store stub.
- Install deps:    `.venv\Scripts\python.exe -m pip install -r requirements.txt`
- Run a backtest:  `.venv\Scripts\python.exe backtests\starter_backtest.py`
- Cached price CSVs live in `/data` (gitignored); delete a file to force a refetch.

## Layout
- `utils/`      — data loaders (`crypto_loader.py`, `stock_loader.py`) + `costs.py`.
- `strategies/` — each module exposes `generate_signals(close, ...) -> (entries, exits)`.
- `backtests/`  — runnable scripts that wire loaders + strategies + costs + IS/OOS split.
- `data/`       — cached OHLCV CSVs.
