@echo off
REM Fizgig Chaotic - console mode (shows logs)
cd /d "%~dp0"
if exist venv\Scripts\python.exe (
    venv\Scripts\python.exe launch_chaotic.pyw
) else (
    python launch_chaotic.pyw
)
pause

