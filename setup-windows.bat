@echo off
setlocal DisableDelayedExpansion
cd /d "%~dp0"
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup-windows.ps1" %*
set "LAZYRIG_SETUP_EXIT=%ERRORLEVEL%"
echo.
pause
exit /b %LAZYRIG_SETUP_EXIT%
