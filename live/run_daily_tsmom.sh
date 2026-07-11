#!/usr/bin/env bash
# Daily runner for the ts_momentum (banded time-series momentum) paper strategy
# (Linux / Raspberry Pi). Linux equivalent of run_daily_tsmom.ps1 -- invoked by
# cron (weekdays, AFTER the US close). Appends timestamped output to
# live/logs/ts_momentum_paper_trader.log so unattended runs leave a reviewable trail.
#
# ISOLATION: this drives ONLY the ts_momentum strategy on its DEDICATED account
# (.env.tsmom / TSMOM_ALPACA_* keys, $10k paper). Completely separate from
# run_daily.sh / paper_trader.py. The two share no script, account, log, or cron line.
#
# PAPER ONLY -- utils/alpaca_client_tsmom.py refuses any non-paper endpoint (CLAUDE.md).

set -u

# Resolve project root = parent of this script's folder (live/).
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root="$(dirname "$script_dir")"
py="$root/.venv/bin/python"
script="$root/live/ts_momentum_paper_trader.py"
log_dir="$root/live/logs"
log="$log_dir/ts_momentum_paper_trader.log"

mkdir -p "$log_dir"

# Run from the project root so relative imports / .env.tsmom resolve correctly.
cd "$root"

stamp="$(date '+%Y-%m-%d %H:%M:%S %z')"
printf '\n===== RUN %s =====\n' "$stamp" >> "$log"

# Redirect both streams to the log file. Do NOT 'set -e' around this: a non-zero
# exit from the trader is data we want to capture and report, not a reason to abort
# before notify.py runs.
"$py" "$script" --live-paper >> "$log" 2>&1
run_exit=$?

stamp_end="$(date '+%Y-%m-%d %H:%M:%S %z')"
printf '===== END  %s  (exit %d) =====\n' "$stamp_end" "$run_exit" >> "$log"

# Push ONE COMBINED Telegram report covering BOTH accounts (each with equity,
# today's longs, and full per-position exit cushions), plus signal-change /
# drawdown / error alerts. This is the LATER daily run, so it owns the combined
# message -- the earlier run (run_daily.sh) no longer sends its own summary. This
# mirrors the desktop run_daily_tsmom.ps1. Read-only, paper-only -- never places
# orders; errors are logged, not fatal.
notify="$root/live/notify.py"
"$py" "$notify" --combined-summary "$log" "ts_momentum" --exit-code "$run_exit" >> "$log" 2>&1
