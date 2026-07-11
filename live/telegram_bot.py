"""Telegram monitor bot for this paper-trading project -- READ-ONLY, PAPER-ONLY.

WHAT THIS IS
------------
A personal Telegram bot that lets you query your paper-trading setup from your
phone. It wraps functions THIS project already has -- it does not introduce any
new trading logic and it NEVER places, modifies, or cancels orders. Per CLAUDE.md
(safety), a chat app cannot move money here; the most it does is read account
state.

COMMANDS
--------
  /start, /help         show the command menu
  /equity               paper account equity, cash, and open positions (Alpaca)
  /positions            per-holding detail: entry, current, unrealized P&L, % book
  /pnl                  return today / 1W / 1M / all-time (Alpaca portfolio history)
  /orders               open (unfilled) orders across both accounts
  /history [N]          recent FILLED orders (what was actually bought/sold)
  /signal               which ETFs the trend_momentum rule signals LONG today
  /compare              both paper books side by side
  /price <SYM>          latest price -- crypto (e.g. BTC/USDT) or an Alpaca stock
  /health               did the daily jobs run? Alpaca reachable? market clock
  /status               tail of the daily paper-trader log

SECURITY
--------
  * Token + allowlist come from .env (gitignored): TELEGRAM_BOT_TOKEN,
    TELEGRAM_ALLOWED_IDS (comma-separated numeric user IDs).
  * Every handler is gated by the allowlist -- non-allowed users get a flat
    refusal and their request never touches your account.

RUN (from project root, venv python):
    .venv\\Scripts\\python.exe live\\telegram_bot.py
The bot uses long-polling, so it only responds while this process is running and
your machine is awake. No public URL / webhook needed.
"""

from __future__ import annotations

import asyncio
import functools
import os
import sys
from typing import Callable

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from telegram import BotCommand, Update
from telegram.constants import ParseMode
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
)

load_dotenv()

# --- Config / allowlist ------------------------------------------------------

TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
ALLOWED_IDS = {
    int(x) for x in os.getenv("TELEGRAM_ALLOWED_IDS", "").replace(" ", "").split(",") if x
}

# --- Auth decorator ----------------------------------------------------------

def restricted(handler: Callable):
    """Reject anyone whose Telegram user ID isn't in TELEGRAM_ALLOWED_IDS."""

    @functools.wraps(handler)
    async def wrapper(update: Update, context: ContextTypes.DEFAULT_TYPE):
        user = update.effective_user
        if user is None or user.id not in ALLOWED_IDS:
            uid = user.id if user else "?"
            if update.message:
                await update.message.reply_text(
                    f"Not authorized (your id: {uid}). "
                    "Add it to TELEGRAM_ALLOWED_IDS in .env to use this bot."
                )
            print(f"[telegram_bot] denied user id={uid}")
            return
        return await handler(update, context)

    return wrapper


# --- Blocking workers (run off the event loop via asyncio.to_thread) ---------
# The Alpaca/ccxt/vectorbt calls are synchronous and can take seconds, so we run
# them in a thread to keep the bot responsive.

# Both paper accounts to report on /equity. Each names the isolated client module
# that owns its keys, so the bot reuses the SAME guarded clients the runners use
# (the tsmom client even has a tripwire against the $1,500 book). Add more here if
# you spin up further accounts.
ACCOUNTS = [
    ("trend_momentum ($1.5k book)", "utils.alpaca_client"),
    ("ts_momentum ($10k book)", "utils.alpaca_client_tsmom"),
]


def _one_account_text(label: str, client_module: str) -> str:
    """Render one account's equity/cash/positions, or an inline error for it."""
    import importlib

    from live.paper_trader import current_positions

    try:
        mod = importlib.import_module(client_module)
        tc = mod.get_trading_client()
        acct = tc.get_account()
        equity = float(acct.portfolio_value)
        cash = float(acct.cash)
        invested = equity - cash
        lines = [
            f"*{label}*",
            f"💰 Equity   ${equity:,.2f}",
            f"💵 Cash     ${cash:,.2f}",
        ]
        held = current_positions(tc)
        if held:
            lines.append(f"📊 Invested ${invested:,.2f}")
            for sym, mv in sorted(held.items(), key=lambda kv: -kv[1]):
                pct = (mv / equity * 100) if equity else 0.0
                lines.append(f"   • {sym}  ${mv:,.2f}  ({pct:.0f}%)")
        else:
            lines.append("📊 In cash — no positions")
        return "\n".join(lines)
    except Exception as e:
        return f"*{label}*\n   ⚠️ {type(e).__name__}: {e}"


def _equity_text() -> str:
    """Report every account in ACCOUNTS, each labelled and independently fetched."""
    blocks = [_one_account_text(label, mod) for label, mod in ACCOUNTS]
    return "🏦 *Accounts* _(paper)_\n\n" + "\n\n".join(blocks)


def _one_account_orders(label: str, client_module: str) -> str:
    """Render one account's OPEN (unfilled) orders, or an inline error for it."""
    import importlib

    from alpaca.trading.enums import QueryOrderStatus
    from alpaca.trading.requests import GetOrdersRequest

    try:
        mod = importlib.import_module(client_module)
        tc = mod.get_trading_client()
        orders = tc.get_orders(GetOrdersRequest(status=QueryOrderStatus.OPEN))
        if not orders:
            return f"*{label}*\n   ✅ none open"
        lines = [f"*{label}*"]
        for o in orders:
            # Orders may be by qty or by notional ($); show whichever is set.
            if getattr(o, "notional", None):
                amt = f"${float(o.notional):,.0f}"
            elif getattr(o, "qty", None):
                amt = f"{float(o.qty):g} sh"
            else:
                amt = "?"
            side = str(o.side).split(".")[-1].lower()
            arrow = "🟢 BUY " if side == "buy" else "🔴 SELL"
            lines.append(f"   {arrow} {o.symbol}  {amt}")
        return "\n".join(lines)
    except Exception as e:
        return f"*{label}*\n   ⚠️ {type(e).__name__}: {e}"


def _orders_text() -> str:
    """Report open orders for every account in ACCOUNTS."""
    blocks = [_one_account_orders(label, mod) for label, mod in ACCOUNTS]
    return "📋 *Open orders* _(unfilled)_\n\n" + "\n\n".join(blocks)


def _signal_text() -> str:
    from live.paper_trader import UNIVERSE, desired_long_symbols, fetch_daily_closes
    from utils.alpaca_client import get_data_client

    closes = fetch_daily_closes(get_data_client())
    longs = desired_long_symbols(closes)
    lines = ["📈 *Signal* — trend\\_momentum _(today)_", ""]
    if longs:
        lines.append("🟢 *LONG:*  " + ", ".join(longs))
        lines.append(f"⚖️ ~{100 / len(longs):.0f}% each (equal-split)")
    else:
        lines.append("⚪ *LONG:*  none — stay in cash")
    flat = [s for s in UNIVERSE if s not in longs]
    if flat:
        lines.append(f"⚫ Flat:  {', '.join(flat)}")
    lines.append("\n_Forward-test signal, not a validated edge._")
    return "\n".join(lines)


def _price_text(arg: str) -> str:
    sym = (arg or "BTC/USDT").strip().upper()

    # Crypto symbols look like BASE/QUOTE; route those to the ccxt loader.
    if "/" in sym:
        from utils.crypto_loader import load_crypto

        df = load_crypto(symbol=sym, timeframe="1d", limit=2)
        last = df["close"].iloc[-1]
        when = df.index[-1].date()
        prev = df["close"].iloc[-2] if len(df) > 1 else last
        chg = (last / prev - 1) * 100 if prev else 0.0
        arrow = "🟢▲" if chg > 0 else "🔴▼" if chg < 0 else "⚪"
        return (
            f"💲 *{sym}*\n"
            f"${last:,.2f}   {arrow} {chg:+.2f}%\n"
            f"_daily close · {when}_"
        )

    # Otherwise treat it as an Alpaca stock/ETF ticker.
    from datetime import datetime, timedelta, timezone

    from alpaca.data.enums import DataFeed
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame

    from utils.alpaca_client import get_data_client

    dc = get_data_client()
    end = datetime.now(timezone.utc) - timedelta(days=1)
    start = end - timedelta(days=10)
    req = StockBarsRequest(
        symbol_or_symbols=[sym],
        timeframe=TimeFrame.Day,
        start=start,
        end=end,
        feed=DataFeed.IEX,
    )
    df = dc.get_stock_bars(req).df
    if df.empty:
        return f"⚠️ No recent bars for *{sym}* — is it a valid ticker?"
    closes = df["close"]
    last = float(closes.iloc[-1])
    prev = float(closes.iloc[-2]) if len(closes) > 1 else last
    chg = (last / prev - 1) * 100 if prev else 0.0
    arrow = "🟢▲" if chg > 0 else "🔴▼" if chg < 0 else "⚪"
    when = closes.index[-1][1].date() if isinstance(closes.index[-1], tuple) else "recent"
    return (
        f"💲 *{sym}*\n"
        f"${last:,.2f}   {arrow} {chg:+.2f}%\n"
        f"_daily close · IEX ~15m delayed · {when}_"
    )


def _status_text(n: int = 25) -> str:
    log = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "live", "logs", "paper_trader.log",
    )
    if not os.path.exists(log):
        return "No paper_trader.log yet -- the daily runner hasn't logged a run."
    # The log is mixed-encoding: PowerShell's Add-Content header lines are ASCII,
    # but the native-command redirect (`*>> $log`) writes UTF-16LE, leaving NUL
    # bytes between characters. Dropping the NULs and any BOM, then decoding as
    # UTF-8, yields clean text for both halves without per-line guessing.
    with open(log, "rb") as f:
        raw = f.read()
    raw = raw.replace(b"\xff\xfe", b"").replace(b"\xfe\xff", b"").replace(b"\x00", b"")
    text = raw.decode("utf-8", errors="replace")
    tail = text.splitlines()[-n:]
    body = "\n".join(tail).strip() or "(log is empty)"
    if len(body) > 3400:  # Telegram message limit is 4096
        body = "...\n" + body[-3400:]
    return f"🗒️ *Daily run log* _(last {n} lines)_\n```\n{body}\n```"


# --- New: P&L, fills history, position detail, comparison, health ------------
# Each reuses the SAME guarded clients in ACCOUNTS. All read-only.

def _portfolio_returns(tc) -> dict:
    """Return today / 1W / 1M / all-time return % from Alpaca portfolio history.

    Alpaca's get_portfolio_history gives an equity series; we derive each window's
    return from first-vs-last equity over that period. Returns NaNs if unavailable.
    """
    from alpaca.trading.requests import GetPortfolioHistoryRequest

    def _window(period: str, timeframe: str) -> float:
        try:
            ph = tc.get_portfolio_history(
                GetPortfolioHistoryRequest(period=period, timeframe=timeframe)
            )
            eq = [e for e in (ph.equity or []) if e]  # drop None/0 gaps
            if len(eq) < 2 or not eq[0]:
                return float("nan")
            return eq[-1] / eq[0] - 1.0
        except Exception:
            return float("nan")

    return {
        "today": _window("1D", "5Min"),
        "week": _window("1W", "1D"),
        "month": _window("1M", "1D"),
        "all": _window("all", "1D"),
    }


def _one_account_pnl(label: str, client_module: str) -> str:
    import importlib

    def _pct(x: float) -> str:
        return "n/a" if x != x else f"{x:+.2%}"  # x!=x catches NaN

    try:
        mod = importlib.import_module(client_module)
        tc = mod.get_trading_client()
        r = _portfolio_returns(tc)
        equity = float(tc.get_account().portfolio_value)
        return (
            f"*{label}*\n"
            f"   💰 ${equity:,.2f}\n"
            f"   Today {_pct(r['today'])}   ·   1W {_pct(r['week'])}\n"
            f"   1M {_pct(r['month'])}   ·   All {_pct(r['all'])}"
        )
    except Exception as e:
        return f"*{label}*\n   ⚠️ {type(e).__name__}: {e}"


def _pnl_text() -> str:
    blocks = [_one_account_pnl(label, mod) for label, mod in ACCOUNTS]
    return "📊 *P&L* _(paper)_\n\n" + "\n\n".join(blocks)


def _one_account_history(label: str, client_module: str, n: int) -> str:
    """Recent FILLED orders (what was actually bought/sold), newest first."""
    import importlib

    from alpaca.trading.enums import QueryOrderStatus
    from alpaca.trading.requests import GetOrdersRequest

    try:
        mod = importlib.import_module(client_module)
        tc = mod.get_trading_client()
        orders = tc.get_orders(
            GetOrdersRequest(status=QueryOrderStatus.CLOSED, limit=n)
        )
        fills = [o for o in orders if getattr(o, "filled_at", None)]
        if not fills:
            return f"*{label}*\n   (no recent fills)"
        lines = [f"*{label}*"]
        for o in fills[:n]:
            side = str(o.side).split(".")[-1].lower()
            arrow = "🟢 BUY " if side == "buy" else "🔴 SELL"
            # Show filled qty (and avg fill price when present).
            qty = getattr(o, "filled_qty", None) or getattr(o, "qty", None)
            amt = f"{float(qty):g} sh" if qty else "?"
            px = getattr(o, "filled_avg_price", None)
            price = f" @ ${float(px):,.2f}" if px else ""
            when = o.filled_at.date() if o.filled_at else ""
            lines.append(f"   {arrow} {o.symbol}  {amt}{price}  _{when}_")
        return "\n".join(lines)
    except Exception as e:
        return f"*{label}*\n   ⚠️ {type(e).__name__}: {e}"


def _history_text(n: int = 10) -> str:
    blocks = [_one_account_history(label, mod, n) for label, mod in ACCOUNTS]
    return f"🧾 *Recent fills* _(last {n})_\n\n" + "\n\n".join(blocks)


def _one_account_positions(label: str, client_module: str) -> str:
    """Per-holding detail: entry, current, unrealized P&L $ and %, % of book."""
    import importlib

    try:
        mod = importlib.import_module(client_module)
        tc = mod.get_trading_client()
        acct = tc.get_account()
        equity = float(acct.portfolio_value)
        positions = tc.get_all_positions()
        if not positions:
            return f"*{label}*\n   📊 In cash — no positions"
        lines = [f"*{label}*  _(${equity:,.2f})_"]
        for p in sorted(positions, key=lambda x: -float(x.market_value)):
            mv = float(p.market_value)
            entry = float(p.avg_entry_price)
            cur = float(p.current_price) if getattr(p, "current_price", None) else float("nan")
            upl = float(p.unrealized_pl) if getattr(p, "unrealized_pl", None) is not None else 0.0
            uplpc = float(p.unrealized_plpc) if getattr(p, "unrealized_plpc", None) is not None else 0.0
            pct_book = (mv / equity * 100) if equity else 0.0
            sign = "🟢" if upl >= 0 else "🔴"
            cur_s = f"${cur:,.2f}" if cur == cur else "—"
            lines.append(
                f"   *{p.symbol}*  ${mv:,.2f}  ({pct_book:.0f}% of book)\n"
                f"      entry ${entry:,.2f} → {cur_s}   {sign} {upl:+,.2f} ({uplpc:+.1%})"
            )
        return "\n".join(lines)
    except Exception as e:
        return f"*{label}*\n   ⚠️ {type(e).__name__}: {e}"


def _positions_text() -> str:
    blocks = [_one_account_positions(label, mod) for label, mod in ACCOUNTS]
    return "📊 *Positions* _(paper)_\n\n" + "\n\n".join(blocks)


def _compare_text() -> str:
    """Side-by-side of both books: equity, all-time return, #positions, holdings."""
    import importlib

    from live.paper_trader import current_positions

    blocks = []
    for label, client_module in ACCOUNTS:
        try:
            mod = importlib.import_module(client_module)
            tc = mod.get_trading_client()
            equity = float(tc.get_account().portfolio_value)
            allret = _portfolio_returns(tc)["all"]
            allret_s = "n/a" if allret != allret else f"{allret:+.2%}"
            held = current_positions(tc)
            held_s = ", ".join(sorted(held)) if held else "cash"
            blocks.append(
                f"*{label}*\n"
                f"   💰 ${equity:,.2f}   ·   All-time {allret_s}\n"
                f"   📊 {len(held)} pos: {held_s}"
            )
        except Exception as e:
            blocks.append(f"*{label}*\n   ⚠️ {type(e).__name__}: {e}")
    return "⚖️ *Compare books* _(paper)_\n\n" + "\n\n".join(blocks)


def _health_text() -> str:
    """Operational status: did the daily jobs run, is Alpaca reachable, market clock.

    Answers 'is everything actually working?' rather than reporting market data.
    """
    import importlib

    logs_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "live", "logs"
    )
    # (label, log filename) per runner.
    run_logs = [
        ("trend_momentum", "paper_trader.log"),
        ("ts_momentum", "ts_momentum_paper_trader.log"),
    ]

    lines = ["🩺 *Health check*", "", "*Daily runs:*"]

    # 1. Last daily-run status per runner (parsed from the END line of each log).
    for label, fname in run_logs:
        path = os.path.join(logs_dir, fname)
        if not os.path.exists(path):
            lines.append(f"   ⚪ {label}: no log yet")
            continue
        with open(path, "rb") as f:
            raw = f.read()
        raw = raw.replace(b"\xff\xfe", b"").replace(b"\xfe\xff", b"").replace(b"\x00", b"")
        text = raw.decode("utf-8", errors="replace")
        end_lines = [ln for ln in text.splitlines() if "===== END" in ln]
        if not end_lines:
            lines.append(f"   ⚪ {label}: never completed a run")
            continue
        last = end_lines[-1]
        ok = "(exit 0)" in last
        stamp = last.replace("===== END", "").replace("=", "").strip()
        emoji = "🟢" if ok else "🔴"
        lines.append(f"   {emoji} {label}: {stamp}")

    # 2. Alpaca reachability + market clock (one cheap call per client).
    lines.append("\n*Alpaca:*")
    clock_done = False
    for label, client_module in ACCOUNTS:
        try:
            mod = importlib.import_module(client_module)
            tc = mod.get_trading_client()
            acct = tc.get_account()
            lines.append(f"   🟢 {label}: {acct.status}")
            if not clock_done:
                clk = tc.get_clock()
                state = "OPEN 🟢" if clk.is_open else "CLOSED 🔴"
                when = clk.next_open if not clk.is_open else clk.next_close
                lines.append(f"\n*Market:* {state}")
                lines.append(
                    f"   next {'open' if not clk.is_open else 'close'}: "
                    f"{when:%Y-%m-%d %H:%M %Z}"
                )
                clock_done = True
        except Exception as e:
            lines.append(f"   🔴 {label}: {type(e).__name__}: {e}")

    return "\n".join(lines)


# --- Command handlers --------------------------------------------------------

HELP = (
    "🤖 *Paper-trading monitor*\n\n"
    "💰 /equity — equity, cash, positions\n"
    "📊 /positions — per-holding detail + unrealized P&L\n"
    "📈 /pnl — return today / 1W / 1M / all-time\n"
    "📋 /orders — open orders (both books)\n"
    "🧾 /history `[N]` — recent fills (default 10)\n"
    "📡 /signal — today's LONG signal\n"
    "⚖️ /compare — both books side by side\n"
    "💲 /price `<SYM>` — e.g. BTC/USDT or SPY\n"
    "🩺 /health — did the daily jobs run? Alpaca + clock\n"
    "🗒️ /status — daily run log\n"
    "❔ /help — this menu\n\n"
    "_Read-only · paper-only · never places orders._"
)


@restricted
async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(HELP, parse_mode=ParseMode.MARKDOWN)


async def _reply_worker(update: Update, fn, *args):
    """Run a blocking worker in a thread; reply with its text or an error."""
    msg = await update.message.reply_text("working...")
    try:
        text = await asyncio.to_thread(fn, *args)
    except Exception as e:  # surface the failure instead of going silent
        text = f"Error: `{type(e).__name__}: {e}`"
    try:
        await msg.edit_text(text, parse_mode=ParseMode.MARKDOWN)
    except Exception:
        # If markdown parsing fails (odd chars in data), resend as plain text.
        await msg.edit_text(text)


@restricted
async def cmd_equity(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await _reply_worker(update, _equity_text)


@restricted
async def cmd_orders(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await _reply_worker(update, _orders_text)


@restricted
async def cmd_signal(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await _reply_worker(update, _signal_text)


@restricted
async def cmd_price(update: Update, context: ContextTypes.DEFAULT_TYPE):
    arg = " ".join(context.args) if context.args else "BTC/USDT"
    await _reply_worker(update, _price_text, arg)


@restricted
async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await _reply_worker(update, _status_text)


@restricted
async def cmd_pnl(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await _reply_worker(update, _pnl_text)


@restricted
async def cmd_history(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # Optional count argument: /history 20  (clamped to a sane range).
    n = 10
    if context.args:
        try:
            n = max(1, min(50, int(context.args[0])))
        except ValueError:
            pass
    await _reply_worker(update, _history_text, n)


@restricted
async def cmd_positions(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await _reply_worker(update, _positions_text)


@restricted
async def cmd_compare(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await _reply_worker(update, _compare_text)


@restricted
async def cmd_health(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await _reply_worker(update, _health_text)


async def _on_error(update: object, context: ContextTypes.DEFAULT_TYPE):
    """Make failures loud instead of silent.

    The most common 'bot running but no reply' cause is a SECOND instance polling
    the same token -- Telegram rejects that with a Conflict and updates vanish.
    Surface it clearly so it's obvious what to fix.
    """
    from telegram.error import Conflict

    err = context.error
    if isinstance(err, Conflict):
        print(
            "\n[telegram_bot] *** CONFLICT ***  Another process is already polling "
            "this bot token.\n  Close any other run_bot.ps1 / telegram_bot.py window "
            "(only ONE may run at a time), then restart.\n"
        )
        return
    print(f"[telegram_bot] handler error: {type(err).__name__}: {err}")


async def _post_init(app: Application):
    """Register the slash-command menu so the Telegram '/' button lists them."""
    await app.bot.set_my_commands([
        BotCommand("equity", "Account equity, cash, positions"),
        BotCommand("positions", "Per-holding detail + unrealized P&L"),
        BotCommand("pnl", "Return today / 1W / 1M / all-time"),
        BotCommand("orders", "Open (unfilled) orders, both accounts"),
        BotCommand("history", "Recent fills: /history [N]"),
        BotCommand("signal", "trend_momentum LONG signal today"),
        BotCommand("compare", "Both books side by side"),
        BotCommand("price", "Latest price: /price BTC/USDT or /price SPY"),
        BotCommand("health", "Daily-run status, Alpaca reachability, clock"),
        BotCommand("status", "Tail of the daily paper-trader log"),
        BotCommand("help", "Show the command menu"),
    ])
    print("[telegram_bot] command menu registered; polling for messages...")


def _acquire_single_instance_lock():
    """Refuse to start if another bot instance is already running.

    Two processes polling the same token CONFLICT -- Telegram hands each update to
    whichever grabs it first, so a stale instance silently 'eats' replies and code
    changes appear not to take effect. We guard against that by holding an OS-level
    exclusive lock on a file for this process's lifetime; a second start fails fast
    with a clear message instead of fighting over your commands.

    Returns the open lock-file handle (keep it alive -- closing it frees the lock).
    """
    lock_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "telegram_bot.lock")
    try:
        import msvcrt  # Windows

        f = open(lock_path, "w")
        msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
        return f
    except ImportError:
        import fcntl  # POSIX fallback

        f = open(lock_path, "w")
        fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        return f
    except OSError:
        sys.exit(
            "Another telegram_bot instance is already running (lock held).\n"
            "Only ONE may run at a time. Close the other window, or run:\n"
            "    Get-Process python | Stop-Process -Force\n"
            "then start this one again."
        )


def main():
    if not TOKEN:
        sys.exit("TELEGRAM_BOT_TOKEN is not set in .env -- see .env.example.")
    if not ALLOWED_IDS:
        sys.exit("TELEGRAM_ALLOWED_IDS is empty in .env -- refusing to run an open bot.")

    _lock = _acquire_single_instance_lock()  # noqa: F841 -- held for process lifetime

    print(f"[telegram_bot] starting; allowed user ids: {sorted(ALLOWED_IDS)}")
    app = Application.builder().token(TOKEN).post_init(_post_init).build()

    app.add_handler(CommandHandler(["start", "help"], cmd_help))
    app.add_handler(CommandHandler("equity", cmd_equity))
    app.add_handler(CommandHandler("positions", cmd_positions))
    app.add_handler(CommandHandler("pnl", cmd_pnl))
    app.add_handler(CommandHandler("orders", cmd_orders))
    app.add_handler(CommandHandler("history", cmd_history))
    app.add_handler(CommandHandler("signal", cmd_signal))
    app.add_handler(CommandHandler("compare", cmd_compare))
    app.add_handler(CommandHandler("price", cmd_price))
    app.add_handler(CommandHandler("health", cmd_health))
    app.add_handler(CommandHandler("status", cmd_status))
    app.add_error_handler(_on_error)

    # drop_pending_updates: ignore commands queued while the bot was offline.
    app.run_polling(drop_pending_updates=True, allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
