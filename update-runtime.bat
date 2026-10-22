@echo off
setlocal DisableDelayedExpansion
cd /d "%~dp0"
echo LazyRig - stable OpenVINO runtime updater
echo.
if not exist ".venv\Scripts\python.exe" (
  echo Run start-server.bat first to install LazyRig.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" -m lazyrig.runtime_update %*
set "LAZYRIG_UPDATE_EXIT=%ERRORLEVEL%"
echo.
pause
exit /b %LAZYRIG_UPDATE_EXIT%
