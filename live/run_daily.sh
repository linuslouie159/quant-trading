#!/usr/bin/env bash
# Daily runner for the trend_momentum paper strategy (Linux / Raspberry Pi).
# Linux equivalent of run_daily.ps1 -- invoked by cron (weekdays, pre-market).
# Runs the trader with --live-paper and appends timestamped output to
# live/logs/paper_trader.log so unattended runs leave a reviewable trail.
#
# PAPER ONLY -- utils/alpaca_client.py refuses any non-paper endpoint (CLAUDE.md).

set -u

# Resolve project root = parent of this script's folder (live/).
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root="$(dirname "$script_dir")"
py="$root/.venv/bin/python"
script="$root/live/paper_trader.py"
log_dir="$root/live/logs"
log="$log_dir/paper_trader.log"

mkdir -p "$log_dir"

# Run from the project root so relative imports / .env resolve correctly.
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

# NOTE: this EARLIER run does NOT send its own Telegram message. The LATER
# ts_momentum run (run_daily_tsmom.sh) emits a SINGLE combined report covering
# BOTH accounts -- including this one, read live -- plus signal-change / drawdown
# alerts. This mirrors the desktop run_daily.ps1 (see combined_summary in
# live/notify.py). The run + log above are unchanged; only the duplicate
# per-account chat message was removed.
