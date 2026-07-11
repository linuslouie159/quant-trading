# Daily runner for the trend_momentum paper strategy.
# Invoked by Windows Task Scheduler (weekdays, pre-market). Runs the trader with
# --live-paper and appends timestamped output to live\logs\paper_trader.log so
# unattended runs leave a reviewable trail.
#
# PAPER ONLY -- utils\alpaca_client.py refuses any non-paper endpoint (CLAUDE.md).

$ErrorActionPreference = "Stop"

# Resolve project root = parent of this script's folder (live\).
$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$py = Join-Path $root ".venv\Scripts\python.exe"
$script = Join-Path $root "live\paper_trader.py"
$logDir = Join-Path $root "live\logs"
$log = Join-Path $logDir "paper_trader.log"

if (-not (Test-Path $logDir)) { New-Item -ItemType Directory -Path $logDir | Out-Null }

$stamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss zzz"
Add-Content -Path $log -Value "`n===== RUN $stamp ====="

# Run from the project root so relative imports / .env resolve correctly.
# Redirect both streams to the log file directly (avoids PowerShell wrapping native
# stderr as error records, which would otherwise mask the real output).
Set-Location $root
& $py $script --live-paper *>> $log

# Capture the runner's exit code NOW, before any later command overwrites it.
$runExit = $LASTEXITCODE

$stampEnd = Get-Date -Format "yyyy-MM-dd HH:mm:ss zzz"
Add-Content -Path $log -Value "===== END  $stampEnd  (exit $runExit) ====="

# NOTE: this earlier run (21:00) no longer sends its own Telegram message. The
# LATER ts_momentum run (run_daily_tsmom.ps1, 21:30) emits a SINGLE combined
# report covering BOTH accounts -- including this one, read live -- plus this
# account's signal-change / drawdown alerts. Kept the run + log above unchanged;
# only the duplicate per-account chat message was removed (see combined_summary
# in live\notify.py).
