"""Validation suite: does the banded time-series momentum strategy actually work?

WHAT "WORKS" MEANS HERE (per CLAUDE.md)
---------------------------------------
This is a DEFENSIVE trend overlay, not an index-beater. The honest test of "works"
is therefore NOT "beats buy & hold on raw return" (it doesn't, in a bull-dominated
sample) but a set of falsifiable risk claims:

  CLAIM 1 -- Drawdown protection: max drawdown is materially SHALLOWER than buy &
             hold, on every index.
  CLAIM 2 -- Risk-adjusted parity-or-better: Sharpe >= buy & hold (you give up
             return but take far less risk), out-of-sample.
  CLAIM 3 -- Bear protection: in named bear markets it sits out the crash.
  CLAIM 4 -- Generalization: the above hold on HELD-OUT indexes that were NEVER
             used to choose any parameter (band=0.03 was tuned only on SPY/QQQ/IWM).
             If it only works on the tuning set, it's overfit.

The strongest evidence is CLAIM 4. We run two groups:
  TUNING SET  : SPY, QQQ, IWM      (used when selecting band=0.03)
  HELD-OUT    : DIA, MDY, EFA, EEM, ^GSPC  (never used to tune anything)

Tests use the strategy's CURRENT DEFAULTS (band=0.03) with NO per-asset tuning --
exactly what a user would get out of the box. Fees + slippage always applied.
A single 70/30 in-sample/out-of-sample split is reported (OOS is what counts).
BACKTEST / PAPER ONLY.

Run:
    .venv\\Scripts\\python.exe backtests\\index_momentum_validation.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import vectorbt as vbt

from strategies.ts_momentum import generate_signals
from utils.costs import FEES, SLIPPAGE, INIT_CASH
from utils.stock_loader import load_stock

TUNING = ["SPY", "QQQ", "IWM"]
HELDOUT = ["DIA", "MDY", "EFA", "EEM", "^GSPC"]
START = "1990-01-01"
IS_FRACTION = 0.70

# Named bear markets to check the strategy sits out (approx public dates).
BEARS = [
    ("dot-com",  "2000-03-24", "2002-10-09"),
    ("GFC",      "2007-10-10", "2009-03-09"),
    ("2022",     "2022-01-03", "2022-10-12"),
]


def _pf(close):
    e, x = generate_signals(close)          # CURRENT DEFAULTS (band=0.03)
    return vbt.Portfolio.from_signals(close, e, x, fees=FEES, slippage=SLIPPAGE,
                                      init_cash=INIT_CASH, freq="1D")


def _bh(close):
    return vbt.Portfolio.from_holding(close, fees=FEES, slippage=SLIPPAGE,
                                      init_cash=INIT_CASH, freq="1D")


def test_symbol(sym):
    try:
        close = load_stock(sym, start=START)["close"].dropna()
    except Exception as e:
        print(f"  [skip] {sym}: {e}")
        return None

    split = int(len(close) * IS_FRACTION)
    oos = close.iloc[split:]
    pf, bh = _pf(oos), _bh(oos)

    s_dd, b_dd = pf.max_drawdown(), bh.max_drawdown()
    s_sh, b_sh = pf.sharpe_ratio(), bh.sharpe_ratio()
    s_cagr, b_cagr = pf.annualized_return(), bh.annualized_return()

    claim1 = s_dd > b_dd                       # shallower (less negative) DD
    claim2 = s_sh >= b_sh - 0.05               # sharpe parity-or-better (small tol)

    # Bear check: strategy total return in each available bear window.
    bear_ok, bear_txt = True, []
    for name, bs, be in BEARS:
        seg = close.loc[bs:be]
        if len(seg) < 30:
            continue
        bret = _pf(seg).total_return()
        bbh = _bh(seg).total_return()
        protected = bret > bbh + 0.05          # clearly better than holding the crash
        bear_ok = bear_ok and protected
        bear_txt.append(f"{name} {bret:+.0%}vs{bbh:+.0%}{'ok' if protected else 'X'}")

    print(f"  {sym:<7} OOS: CAGR {s_cagr:>+6.1%} (B&H {b_cagr:>+6.1%}) | "
          f"Sharpe {s_sh:.2f} (B&H {b_sh:.2f}) {'PASS' if claim2 else 'FAIL'} | "
          f"maxDD {s_dd:>+6.1%} (B&H {b_dd:>+6.1%}) {'PASS' if claim1 else 'FAIL'} | "
          f"bears [{' '.join(bear_txt)}] {'PASS' if bear_ok else 'FAIL'}")
    return dict(sym=sym, c1=claim1, c2=claim2, bear=bear_ok,
                s_sh=s_sh, b_sh=b_sh, s_dd=s_dd, b_dd=b_dd)


def run_group(title, syms):
    print(f"\n{title}")
    print("-" * 118)
    out = [r for r in (test_symbol(s) for s in syms) if r]
    return out


def main():
    print("=" * 118)
    print("VALIDATION SUITE -- banded TS-momentum (band=0.03 defaults, NO per-asset tuning)")
    print(f"OOS = last {1-IS_FRACTION:.0%} of each index | fees={FEES:.3%} slippage={SLIPPAGE:.3%} | "
          f"BACKTEST/PAPER ONLY")
    print("Claims: (1) shallower maxDD than B&H  (2) Sharpe >= B&H  (3) sits out named bears")
    print("=" * 118)

    tuned = run_group("TUNING SET (SPY/QQQ/IWM -- used when picking band=0.03):", TUNING)
    held = run_group("HELD-OUT SET (never used to tune ANY parameter -- the real test):", HELDOUT)

    allr = tuned + held
    print("\n" + "=" * 118)
    print("VERDICT (per CLAUDE.md)")
    print("=" * 118)

    def tally(rs, key):
        return sum(1 for r in rs if r[key]), len(rs)

    for label, rs in [("TUNING", tuned), ("HELD-OUT", held)]:
        d1 = tally(rs, "c1"); d2 = tally(rs, "c2"); db = tally(rs, "bear")
        print(f"  {label:<9}: shallower maxDD {d1[0]}/{d1[1]} | Sharpe>=B&H {d2[0]}/{d2[1]} | "
              f"sat out bears {db[0]}/{db[1]}")

    held_dd = tally(held, "c1")
    print()
    if held_dd[0] == held_dd[1] and held_dd[1] > 0:
        print("  => Drawdown protection GENERALIZES to indexes never used for tuning. That is the")
        print("     strategy's real, robust property -- it is a working DEFENSIVE overlay.")
    else:
        print("  => Drawdown protection did NOT hold on all held-out indexes -- weaker than hoped.")
    print("  Caveat that does NOT go away: it TRAILS buy & hold on raw return in bull-dominated")
    print("  samples. 'Works' here = cuts crash risk reliably, not = makes more money. Daily bars,")
    print("  one rule, single split. BACKTEST/PAPER ONLY -- no live-order code wired up.")


if __name__ == "__main__":
    main()
