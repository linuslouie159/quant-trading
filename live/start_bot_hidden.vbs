' Launches the Telegram bot completely hidden (no PowerShell window flash).
' A copy of this file is placed in the Windows Startup folder so the bot starts
' automatically at login. It calls live\run_bot.ps1, which logs to
' live\logs\telegram_bot.log. Single-instance lock in telegram_bot.py prevents
' duplicates, so this is safe even if you also start the bot manually.
'
' To stop auto-start: delete the copy in your Startup folder
'   (shell:startup -> "VectorBT Telegram Bot.vbs").

Dim shell, fso, scriptDir, root, ps1
Set shell = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")

' This .vbs lives in <root>\live\ ; derive the project root from its own path.
scriptDir = fso.GetParentFolderName(WScript.ScriptFullName)
root = fso.GetParentFolderName(scriptDir)
ps1 = root & "\live\run_bot.ps1"

' 0 = hidden window, False = don't wait (let it run in the background).
shell.Run "powershell.exe -NoProfile -WindowStyle Hidden -ExecutionPolicy Bypass -File """ & ps1 & """", 0, False
