# Launcher for the read-only Telegram monitor bot (telegram_bot.py).
# Registered to run AtLogOn via Task Scheduler so the bot is always available to
# answer on-demand commands (/equity, /positions, /exits-style status, etc.).
#
# READ-ONLY, PAPER-ONLY: the bot never places, modifies, or cancels orders
# (CLAUDE.md). It long-polls Telegram, so it just needs to stay running.
#
# The bot holds a single-instance lock, so if one is already running this launch
# simply exits -- safe to fire on every login.

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$py = Join-Path $root ".venv\Scripts\python.exe"
$script = Join-Path $root "live\telegram_bot.py"
$logDir = Join-Path $root "live\logs"
$log = Join-Path $logDir "telegram_bot.log"

if (-not (Test-Path $logDir)) { New-Item -ItemType Directory -Path $logDir | Out-Null }

$stamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss zzz"
Add-Content -Path $log -Value "`n===== BOT START $stamp ====="

# Run from project root so .env / relative imports resolve. Append all output to the
# log. This stays in the foreground of THIS process (the scheduled task), which is
# fine -- the task runs the bot for its whole lifetime.
Set-Location $root
& $py $script *>> $log
