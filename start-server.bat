@echo off
setlocal DisableDelayedExpansion
cd /d "%~dp0"
echo Starting LazyRig Server...
if not exist ".venv\Scripts\python.exe" (
  powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup-windows.ps1"
  if errorlevel 1 goto setup_failed
)
.venv\Scripts\python.exe -u -m lazyrig --mode openvino %*
set "LAZYRIG_SERVER_EXIT=%ERRORLEVEL%"
if not "%LAZYRIG_SERVER_EXIT%"=="0" (
  echo.
  echo If the environment is incomplete, run setup-windows.bat or setup-portable.bat again.
)
pause
exit /b %LAZYRIG_SERVER_EXIT%
:setup_failed
echo.
echo Setup did not complete. See the setup log in the logs folder.
pause
exit /b 1
