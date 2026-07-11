"""Telegram PUSH notifier for the paper-trading runners -- READ-ONLY, PAPER-ONLY.

WHY THIS EXISTS
---------------
telegram_bot.py is PULL-only: it answers when you ask. It runs as a long-polling
process and is silent unless messaged. But the things you most want to hear about
happen on the DAILY job, which the bot isn't part of:

  * the daily run finished (what it traded, new equity)   -> daily summary
  * today's signal differs from yesterday's                -> signal-change alert
  * the daily run crashed / exited non-zero                -> error alert
  * equity dropped past a drawdown threshold from its peak  -> drawdown alert

So this is a SEPARATE, short-lived script the PowerShell runners call after each
run. It POSTs to Telegram's sendMessage HTTP API directly (just `requests`, no
event loop), so it can't conflict with the always-on polling bot or trip its
single-instance lock. It only READS account state and SENDS chat messages -- it
never places, modifies, or cancels orders (CLAUDE.md).

ACCOUNTS
--------
Each entry maps an account KEY to the isolated client module + runner module that
own it -- the same guarded clients the bot and runners already use (the tsmom
client even has a tripwire against the $1,500 book). Add an entry to wire a new
account into alerts.

CLI (called by run_daily*.ps1; also handy by hand):
    python live/notify.py --text "hello"
    python live/notify.py --daily-summary <logfile> <account_key> [--exit-code N]
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

from live.state import load as state_load, save as state_save

load_dotenv()

# --- Config (reuse the SAME env vars as the bot) -----------------------------

TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
ALLOWED_IDS = [
    int(x) for x in os.getenv("TELEGRAM_ALLOWED_IDS", "").replace(" ", "").split(",") if x
]

# Alert when equity falls this far below its running peak (fraction). Module-level
# so it's easy to find and tune. -10% default.
DRAWDOWN_THRESHOLD = 0.10

# Account key -> how to reach it. `client` owns the (paper-guarded) Alpaca keys;
# `runner` exposes fetch_daily_closes / desired_long_symbols for that strategy.
ACCOUNTS: dict[str, dict] = {
    "trend_momentum": {
        "label": "trend_momentum ($1.5k book)",
        "client": "utils.alpaca_client",
        "runner": "live.paper_trader",
    },
    "ts_momentum": {
        "label": "ts_momentum ($10k book)",
        "client": "utils.alpaca_client_tsmom",
        "runner": "live.ts_momentum_paper_trader",
    },
}


# --- Core sender -------------------------------------------------------------

def send(text: str, parse_mode: str | None = "Markdown") -> None:
    """Send `text` to every allowed Telegram id. Fails SOFT -- never raises.

    A notify failure must not break a trading run, so any error here is printed
    and swallowed. If Markdown parsing fails on Telegram's side we retry the same
    message as plain text so the content still gets through.
    """
    if not TOKEN:
        print("[notify] TELEGRAM_BOT_TOKEN not set -- cannot send.")
        return
    if not ALLOWED_IDS:
        print("[notify] TELEGRAM_ALLOWED_IDS empty -- nobody to notify.")
        return

    import requests

    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    for chat_id in ALLOWED_IDS:
        payload = {"chat_id": chat_id, "text": text}
        if parse_mode:
            payload["parse_mode"] = parse_mode
        try:
            r = requests.post(url, data=payload, timeout=20)
            if not r.ok and parse_mode:
                # Most failures are Markdown parse errors on odd chars -> resend plain.
                requests.post(
                    url, data={"chat_id": chat_id, "text": text}, timeout=20
                )
            elif not r.ok:
                print(f"[notify] send to {chat_id} failed: {r.status_code} {r.text}")
        except Exception as e:  # network hiccup etc. -- log, don't raise
            print(f"[notify] send to {chat_id} errored: {type(e).__name__}: {e}")


# --- Log reading (shared NUL/BOM handling with the bot's /status) ------------

def _read_log_text(path: str) -> str:
    """Decode a runner log that mixes ASCII headers with UTF-16LE redirect output.

    Same handling as telegram_bot._status_text: PowerShell's Add-Content writes
    ASCII header lines while the native `*>> $log` redirect writes UTF-16LE,
    leaving NUL bytes. Strip the BOM and NULs, then decode as UTF-8.
    """
    if not os.path.exists(path):
        return ""
    with open(path, "rb") as f:
        raw = f.read()
    raw = raw.replace(b"\xff\xfe", b"").replace(b"\xfe\xff", b"").replace(b"\x00", b"")
    return raw.decode("utf-8", errors="replace")


def _last_run_lines(text: str) -> list[str]:
    """Return the lines of the MOST RECENT run block (after the last '===== RUN')."""
    lines = text.splitlines()
    start = 0
    for i, ln in enumerate(lines):
        if "===== RUN" in ln:
            start = i
    return lines[start:]


def _parse_run(lines: list[str]) -> dict:
    """Pull the handful of facts the runner prints into a small dict.

    The runners print stable, greppable lines (paper_trader.py / ts_momentum_*):
        equity: $X        -> on the 'account status' line
        signalled LONG today: [...] / entered TODAY: [...]
        planned orders:   -> followed by '  BUY SPY ~$...' lines
    We parse leniently; anything missing just comes back as None/empty.
    """
    info: dict = {"equity": None, "longs": None, "orders": [], "no_action": False}
    grabbing_orders = False
    for ln in lines:
        s = ln.strip()
        if "equity: $" in s:
            try:
                info["equity"] = float(
                    s.split("equity: $", 1)[1].split()[0].replace(",", "")
                )
            except (ValueError, IndexError):
                pass
        if s.startswith("signalled LONG today:") or s.startswith("entered TODAY:"):
            payload = s.split(":", 1)[1].strip()
            if "(none" in payload:
                info["longs"] = []
            else:
                info["longs"] = [
                    t.strip(" []'\"") for t in payload.strip("[]").split(",") if t.strip(" []'\"")
                ]
        if s.startswith("no actions"):
            info["no_action"] = True
        if s.startswith("planned orders:"):
            grabbing_orders = True
            continue
        if grabbing_orders:
            # Order lines look like 'BUY   SPY   ~$123.45'; stop at a blank/section line.
            if not s or s.startswith("DRY RUN") or s.startswith("submitting") \
                    or s.startswith("done") or s.startswith("====="):
                grabbing_orders = False
            elif any(s.upper().startswith(side) for side in ("BUY", "SELL", "CLOSE")):
                info["orders"].append(s)
    return info


def _exit_code_from_log(lines: list[str]) -> int | None:
    """Parse the '===== END ... (exit N) =====' line the PowerShell wrapper writes."""
    for ln in reversed(lines):
        if "===== END" in ln and "(exit" in ln:
            try:
                return int(ln.split("(exit", 1)[1].split(")", 1)[0].strip())
            except (ValueError, IndexError):
                return None
    return None


# --- Account reads (paper-only, read-only) -----------------------------------

def _account_equity(client_module: str) -> float:
    import importlib

    mod = importlib.import_module(client_module)
    tc = mod.get_trading_client()
    return float(tc.get_account().portfolio_value)


def _today_longs(runner_module: str, client_module: str) -> list[str]:
    import importlib

    runner = importlib.import_module(runner_module)
    client = importlib.import_module(client_module)
    closes = runner.fetch_daily_closes(client.get_data_client())
    return runner.desired_long_symbols(closes)


# --- Alert builders ----------------------------------------------------------

def _signal_change_block(key: str, label: str, longs: list[str]) -> str | None:
    """Compare today's longs to the stored set; return an alert block if changed."""
    store = state_load("last_signal", {})
    prev = store.get(key)
    if prev is None:
        # First time we've seen this account: record, don't alert (nothing to diff).
        store[key] = longs
        state_save("last_signal", store)
        return None
    if set(prev) == set(longs):
        return None
    added = [s for s in longs if s not in prev]
    removed = [s for s in prev if s not in longs]
    store[key] = longs
    state_save("last_signal", store)
    lines = ["🔔 *Signal change*"]
    if added:
        lines.append("🟢 new LONG:  " + ", ".join(added))
    if removed:
        lines.append("⚫ now flat:  " + ", ".join(removed))
    lines.append("📋 holding now:  " + (", ".join(longs) if longs else "cash"))
    return "\n".join(lines)


def _held_cushion_rows(runner_module: str, client_module: str) -> list[dict]:
    """Return exit-level rows for the HELD positions, closest-to-exit first.

    Read-only: pulls the runner's daily closes + the strategy's exit_levels via the
    runner's report helper. Returns [] if the runner doesn't expose exit reporting
    (e.g. a strategy without exit_levels) or nothing is held.
    """
    import importlib

    runner = importlib.import_module(runner_module)
    client = importlib.import_module(client_module)
    if not hasattr(runner, "report_exit_levels"):
        return []

    closes = runner.fetch_daily_closes(client.get_data_client())
    held = set(runner.current_positions(client.get_trading_client()))
    if not held:
        return []

    rows = runner.report_exit_levels(closes, held=held)
    held_rows = [r for r in rows if r.get("held")]
    held_rows.sort(key=lambda r: r["cushion"])  # closest to exit first
    return held_rows


def _format_cushion_row(r: dict) -> str:
    """One Telegram line for a held position's exit cushion."""
    mom = r.get("momentum")
    mom_txt = f"  (mom {mom:+.1%})" if mom == mom else ""  # skip NaN (no momentum gate)
    flag = "  ⚠️ mom<=0" if r.get("momentum_exit") else ""
    return (f"   {r['symbol']}: {r['pct_above_band_exit']:+.1%} above exit "
            f"${r['band_exit']:,.2f}{mom_txt}{flag}")


def _exit_cushion_block(runner_module: str, client_module: str) -> str | None:
    """Report the held position CLOSEST to its exit (least cushion). Fails soft."""
    rows = _held_cushion_rows(runner_module, client_module)
    if not rows:
        return None
    tightest = rows[0]  # already sorted closest-first
    lines = ["🛡️ *Exit cushion* (closest to selling)", _format_cushion_row(tightest)]
    return "\n".join(lines)


def _all_cushions_block(runner_module: str, client_module: str) -> str | None:
    """List EVERY held position's exit cushion, closest-to-exit first. Fails soft."""
    rows = _held_cushion_rows(runner_module, client_module)
    if not rows:
        return None
    lines = ["🛡️ Exit cushions (closest first):"]
    lines.extend(_format_cushion_row(r) for r in rows)
    return "\n".join(lines)


def _drawdown_block(key: str, equity: float) -> str | None:
    """Update stored peak; return an alert block if drawdown crosses the threshold."""
    store = state_load("peak_equity", {})
    peak = float(store.get(key, 0.0))
    if equity > peak:
        store[key] = equity
        state_save("peak_equity", store)
        return None
    if peak <= 0:
        return None
    dd = (equity - peak) / peak  # negative
    if dd <= -DRAWDOWN_THRESHOLD:
        return (
            "⚠️ *Drawdown alert*\n"
            f"   Equity  ${equity:,.2f}\n"
            f"   Peak    ${peak:,.2f}\n"
            f"   Down    {dd:.1%} from peak (threshold {-DRAWDOWN_THRESHOLD:.0%})"
        )
    return None


def daily_summary(logfile: str, key: str, exit_code: int | None) -> None:
    """Build and send the daily summary + any triggered alerts for one account.

    Order of concerns:
      1. If the run FAILED (non-zero exit), send an error alert and stop -- the
         account reads below would likely be unreliable anyway.
      2. Otherwise send the run summary (parsed from the log).
      3. Then the signal-change and drawdown alerts (live account reads).
    """
    spec = ACCOUNTS.get(key)
    if spec is None:
        print(f"[notify] unknown account key {key!r}; known: {list(ACCOUNTS)}")
        return
    label = spec["label"]

    text = _read_log_text(logfile)
    lines = _last_run_lines(text)
    # Prefer the explicitly-passed exit code; fall back to parsing the log.
    if exit_code is None:
        exit_code = _exit_code_from_log(lines)

    # 1. Error alert -- run crashed.
    if exit_code is not None and exit_code != 0:
        tail = "\n".join(lines[-15:]).strip()
        if len(tail) > 3000:
            tail = "...\n" + tail[-3000:]
        send(
            f"🚨 *Daily run FAILED* — {label}\n"
            f"   exit code {exit_code}\n\n"
            f"```\n{tail}\n```"
        )
        return

    # 2. Run summary from the log.
    info = _parse_run(lines)
    summary = [f"✅ *Daily run* — {label}"]
    if info["equity"] is not None:
        summary.append(f"💰 Equity  ${info['equity']:,.2f}")
    if info["longs"] is not None:
        summary.append(
            "📈 LONG:  " + (", ".join(info["longs"]) if info["longs"] else "none (cash)")
        )
    if info["no_action"]:
        summary.append("⚖️ No orders — already on target.")
    elif info["orders"]:
        summary.append("🧾 Orders:")
        summary.extend("   • " + o for o in info["orders"])
    send("\n".join(summary))

    # 3. Signal-change + drawdown alerts (live reads; fail soft).
    try:
        longs = info["longs"] if info["longs"] is not None else _today_longs(
            spec["runner"], spec["client"]
        )
        block = _signal_change_block(key, label, longs)
        if block:
            send(block)
    except Exception as e:
        print(f"[notify] signal-change check failed: {type(e).__name__}: {e}")

    try:
        equity = info["equity"] if info["equity"] is not None else _account_equity(
            spec["client"]
        )
        block = _drawdown_block(key, equity)
        if block:
            send(block)
    except Exception as e:
        print(f"[notify] drawdown check failed: {type(e).__name__}: {e}")

    try:
        block = _exit_cushion_block(spec["runner"], spec["client"])
        if block:
            send(block)
    except Exception as e:
        print(f"[notify] exit-cushion check failed: {type(e).__name__}: {e}")


def _account_block(key: str, spec: dict) -> str:
    """Build a per-account block (label, equity, longs, full cushions) for the
    combined message. Reads live via the account's OWN isolated client; each piece
    fails soft so one unreachable account can't blank the whole report.
    """
    lines = [f"*{spec['label']}*"]
    try:
        equity = _account_equity(spec["client"])
        lines.append(f"💰 Equity  ${equity:,.2f}")
    except Exception as e:
        lines.append(f"💰 Equity  (unavailable: {type(e).__name__})")
    try:
        longs = _today_longs(spec["runner"], spec["client"])
        lines.append("📈 LONG:  " + (", ".join(longs) if longs else "none (cash)"))
    except Exception as e:
        lines.append(f"📈 LONG:  (unavailable: {type(e).__name__})")
    try:
        cushions = _all_cushions_block(spec["runner"], spec["client"])
        lines.append(cushions if cushions else "🛡️ no held positions")
    except Exception as e:
        lines.append(f"🛡️ cushions (unavailable: {type(e).__name__})")
    return "\n".join(lines)


def combined_summary(logfile: str, trigger_key: str, exit_code: int | None) -> None:
    """Send ONE message covering EVERY account in ACCOUNTS.

    Triggered by the later daily run. Each account is read live via its own
    isolated client, so the combined report does not depend on the other run's log.
    The triggering run's exit code still gates an error alert (a failed run's reads
    would be unreliable). Signal-change + drawdown STATE alerts still fire per
    account afterwards so that memory keeps updating.
    """
    spec = ACCOUNTS.get(trigger_key)
    if spec is None:
        print(f"[notify] unknown account key {trigger_key!r}; known: {list(ACCOUNTS)}")
        return

    text = _read_log_text(logfile)
    lines = _last_run_lines(text)
    if exit_code is None:
        exit_code = _exit_code_from_log(lines)

    # If the triggering run crashed, alert on that and stop (reads unreliable).
    if exit_code is not None and exit_code != 0:
        tail = "\n".join(lines[-15:]).strip()
        if len(tail) > 3000:
            tail = "...\n" + tail[-3000:]
        send(
            f"🚨 *Daily run FAILED* — {spec['label']}\n"
            f"   exit code {exit_code}\n\n"
            f"```\n{tail}\n```"
        )
        return

    # Build the single combined message: header + one block per account.
    blocks = ["📊 *Daily paper-trading report*"]
    for key, acc in ACCOUNTS.items():
        try:
            blocks.append(_account_block(key, acc))
        except Exception as e:
            blocks.append(f"*{acc['label']}*\n   (report failed: {type(e).__name__}: {e})")
    send("\n\n".join(blocks))

    # Per-account state alerts (signal change + drawdown) still update + fire.
    for key, acc in ACCOUNTS.items():
        try:
            longs = _today_longs(acc["runner"], acc["client"])
            block = _signal_change_block(key, acc["label"], longs)
            if block:
                send(block)
        except Exception as e:
            print(f"[notify] signal-change ({key}) failed: {type(e).__name__}: {e}")
        try:
            equity = _account_equity(acc["client"])
            block = _drawdown_block(key, equity)
            if block:
                send(block)
        except Exception as e:
            print(f"[notify] drawdown ({key}) failed: {type(e).__name__}: {e}")


# --- CLI ---------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Telegram push notifier (paper-only).")
    ap.add_argument("--text", help="send this literal message and exit")
    ap.add_argument(
        "--daily-summary",
        nargs=2,
        metavar=("LOGFILE", "ACCOUNT_KEY"),
        help="parse LOGFILE's last run for ACCOUNT_KEY and send summary + alerts",
    )
    ap.add_argument(
        "--combined-summary",
        nargs=2,
        metavar=("LOGFILE", "TRIGGER_KEY"),
        help="send ONE message covering ALL accounts; TRIGGER_KEY's exit code "
             "gates the error alert (use on the later daily run)",
    )
    ap.add_argument(
        "--exit-code",
        type=int,
        default=None,
        help="exit code of the daily run (non-zero -> error alert)",
    )
    args = ap.parse_args()

    if args.text:
        send(args.text)
        return
    if args.combined_summary:
        logfile, key = args.combined_summary
        combined_summary(logfile, key, args.exit_code)
        return
    if args.daily_summary:
        logfile, key = args.daily_summary
        daily_summary(logfile, key, args.exit_code)
        return
    ap.print_help()


if __name__ == "__main__":
    main()
