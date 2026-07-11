"""Edge test: strategy return vs simple BUY-AND-HOLD, per coin per year.

The definitive question for a directional crypto strategy: did it beat just holding
the coin? If it underperforms buy-and-hold while taking leverage + fees + deep
drawdowns, there is no edge -- the "returns" are just (worse) market beta.

Buy-and-hold is computed straight from the cached perp 4h closes (first close of the
year -> last close of the year). Strategy numbers are the FAITHFUL Jesse results
(ETH, SOL) and the docstring's CLAIMED BTC numbers, hard-coded for the comparison.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.crypto_loader import load_perp_range

# Faithful Jesse strategy returns (% of $10k base, additive per-year contributions).
# ETH/SOL from this session's Jesse runs; BTC 2023/24 are the docstring's CLAIMED figures.
STRAT = {
    "ETH/USDT": {2023: 7.5, 2024: 70.8, 2025: 103.0, 2026: -10.9},
    "SOL/USDT": {2023: 152.0, 2024: -58.0, 2025: -40.0, 2026: 31.0},
    "BTC/USDT": {2023: 79.2, 2024: 42.9},  # claimed (train/test); 2025/26 not claimed
}


def buy_hold_by_year(symbol: str) -> dict[int, float]:
    df = load_perp_range(symbol=symbol, timeframe="4h", start="2022-09-01", end=None)
    df = df[~df.index.duplicated(keep="first")].sort_index()
    out = {}
    for year in (2023, 2024, 2025, 2026):
        sub = df[df.index.year == year]
        if len(sub) < 2:
            continue
        first, last = sub["close"].iloc[0], sub["close"].iloc[-1]
        out[year] = (last / first - 1.0) * 100.0
    return out


def main():
    print(f"{'coin':<10}{'year':<6}{'strategy':>12}{'buy&hold':>12}{'edge (strat-BH)':>18}")
    print("-" * 58)
    for symbol, strat_years in STRAT.items():
        bh = buy_hold_by_year(symbol)
        for year in sorted(strat_years):
            s = strat_years[year]
            b = bh.get(year)
            if b is None:
                continue
            edge = s - b
            tag = "  <-- beat hold" if edge > 0 else ""
            print(f"{symbol:<10}{year:<6}{s:>11.1f}%{b:>11.1f}%{edge:>16.1f}%{tag}")
        print()


if __name__ == "__main__":
    main()
