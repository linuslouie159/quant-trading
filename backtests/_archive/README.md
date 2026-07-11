# backtests/_archive

Retired one-off scripts, kept for reference (not part of the active research loop).

These four were tools for a **one-time validation**: porting the Jesse
`DonchianBreakoutPro` strategy to VectorBT (`strategies/donchian_breakout.py`) and
reconciling the VBT port against Jesse's trade-by-trade log. That validation is
finished, so the scripts live here rather than cluttering `backtests/`.

| Script | Purpose |
|---|---|
| `donchian_crosscheck.py` | Per-year BTC/USDT 4h run; CLAIMED (Jesse) vs OBSERVED (VBT) comparison. |
| `eth_crosscheck.py` | Re-runs the BTC-tuned params on ETH/SOL — overfitting check. |
| `reconcile_trade1.py` | Lines up the VBT port's first 2024 trade against Jesse trade #1. |
| `export_2024_trades.py` | Dumps the VBT order + trade list for 2024 cross-engine diffing. |

To run one, invoke it from the project root (the `backtests\...` paths in their
docstrings now point here, i.e. `backtests\_archive\...`):

    .venv\Scripts\python.exe backtests\_archive\donchian_crosscheck.py
