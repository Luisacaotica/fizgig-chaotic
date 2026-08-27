@echo off
REM Fizgig Chaotic - console mode (shows logs, unbuffered)
cd /d "%~dp0"
echo [chaotic] launching with venv python (console, unbuffered)...
echo [chaotic] python: venv\Scripts\python.exe
if exist "venv\Scripts\python.exe" (
    "venv\Scripts\python.exe" -u launch_chaotic.pyw
    echo.
    echo [chaotic] exit code: %errorlevel%
) else (
    echo [chaotic] no venv, using system python
    python -u launch_chaotic.pyw
    echo.
    echo [chaotic] exit code: %errorlevel%
)
echo.
if exist launch_chaotic_error.log (
    echo --- launch_chaotic_error.log ---
    type launch_chaotic_error.log
    echo --- end log ---
) else (
    echo No error log.
)
pause
