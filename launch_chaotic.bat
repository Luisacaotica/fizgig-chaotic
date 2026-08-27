@echo off
REM Fizgig Chaotic - Luisa Caotica Edition (8GB VRAM tuned)
cd /d "%~dp0"
if exist "%~dp0venv\Scripts\pythonw.exe" (
    start "" /b wscript //nologo //b "%~dp0run_chaotic_silent.vbs"
) else (
    echo No venv found - run install_fizgig.bat first
    pause
)

