@echo off
setlocal

:: ---------------------------------------------------------------------------
:: Energie Delen — Dashboard launcher (Windows)
:: Installs uv if not present, then starts the Panel dashboard.
:: Safe to run multiple times: uv and Python are only downloaded once.
:: ---------------------------------------------------------------------------

:: Check if uv is already on PATH
where uv >nul 2>&1
if %errorlevel% == 0 (
    set "UV=uv"
    goto :launch
)

:: Check the default install location used by the uv installer
if exist "%USERPROFILE%\.local\bin\uv.exe" (
    set "UV=%USERPROFILE%\.local\bin\uv.exe"
    goto :launch
)

:: uv not found — install it
echo Setting up (one-time, this may take a moment)...
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex" >nul 2>&1
if not exist "%USERPROFILE%\.local\bin\uv.exe" (
    echo ERROR: Setup failed. Please install uv manually from https://docs.astral.sh/uv/
    pause
    exit /b 1
)
set "UV=%USERPROFILE%\.local\bin\uv.exe"

:launch
:: Move to the directory containing this script
cd /d "%~dp0"

echo Starting Energie Delen dashboard...
echo Dashboard running. Your browser will open automatically. Press Ctrl+C to stop.
"%UV%" run --quiet --group dashboard panel serve dashboard/app.py --show 2>nul
