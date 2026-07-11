# Launcher for the Telegram monitor bot (live\telegram_bot.py).
# Long-polling: this process must stay running for the bot to respond, and your
# machine must be awake. READ-ONLY / PAPER-ONLY -- it never places orders.
#
#   Start it:   powershell -ExecutionPolicy Bypass -File live\run_bot.ps1
#   Stop it:    Ctrl+C in this window.
#
# Token + allowlist are read from .env (TELEGRAM_BOT_TOKEN, TELEGRAM_ALLOWED_IDS).

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$py = Join-Path $root ".venv\Scripts\python.exe"
$script = Join-Path $root "live\telegram_bot.py"
$logDir = Join-Path $root "live\logs"
$log = Join-Path $logDir "telegram_bot.log"

if (-not (Test-Path $logDir)) { New-Item -ItemType Directory -Path $logDir | Out-Null }

# Append a start banner, then run the bot with all output going to the log file.
# (-u = unbuffered, so lines land in the log immediately.) When launched by Task
# Scheduler there is no console, so the log is how you see what the bot is doing.
$stamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss zzz"
Add-Content -Path $log -Value "`n===== BOT START $stamp ====="

Set-Location $root
& $py -u $script *>> $log
