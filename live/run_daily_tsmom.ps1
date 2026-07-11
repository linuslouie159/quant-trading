# Daily runner for the ts_momentum (banded time-series momentum) paper strategy.
# Invoked by Windows Task Scheduler (weekdays, after the US close). Runs the trader
# with --live-paper and appends timestamped output to
# live\logs\ts_momentum_paper_trader.log so unattended runs leave a reviewable trail.
#
# ISOLATION: this drives ONLY the ts_momentum strategy on its DEDICATED account
# (.env.tsmom / TSMOM_ALPACA_* keys, $10k paper). It is completely separate from
# run_daily.ps1 / paper_trader.py, which automate the OTHER strategy on a different
# account. The two share no script, no account, no log, and no scheduled task.
#
# PAPER ONLY -- utils\alpaca_client_tsmom.py refuses any non-paper endpoint (CLAUDE.md).

$ErrorActionPreference = "Stop"

# Resolve project root = parent of this script's folder (live\).
$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$py = Join-Path $root ".venv\Scripts\python.exe"
$script = Join-Path $root "live\ts_momentum_paper_trader.py"
$logDir = Join-Path $root "live\logs"
$log = Join-Path $logDir "ts_momentum_paper_trader.log"

if (-not (Test-Path $logDir)) { New-Item -ItemType Directory -Path $logDir | Out-Null }

$stamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss zzz"
Add-Content -Path $log -Value "`n===== RUN $stamp ====="

# Run from the project root so relative imports / .env.tsmom resolve correctly.
# Redirect both streams to the log file directly (avoids PowerShell wrapping native
# stderr as error records, which would otherwise mask the real output).
Set-Location $root
& $py $script --live-paper *>> $log

# Capture the runner's exit code NOW, before any later command overwrites it.
$runExit = $LASTEXITCODE

$stampEnd = Get-Date -Format "yyyy-MM-dd HH:mm:ss zzz"
Add-Content -Path $log -Value "===== END  $stampEnd  (exit $runExit) ====="

# Push ONE COMBINED Telegram report covering BOTH accounts (each with equity,
# today's longs, and full per-position exit cushions), plus signal-change /
# drawdown / error alerts. This is the LATER daily run (21:30), so it owns the
# combined message -- the earlier run (run_daily.ps1, 21:00) no longer sends its
# own summary. Read-only, paper-only -- never places orders; errors are logged.
$notify = Join-Path $root "live\notify.py"
& $py $notify --combined-summary $log "ts_momentum" --exit-code $runExit *>> $log
