@echo off
setlocal DisableDelayedExpansion
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo LazyRig runtime is missing. Run setup-windows.bat first.
  pause
  exit /b 1
)
echo Checking GPU drivers and small OpenCL kernels. No models will be loaded.
.venv\Scripts\python.exe -u -m lazyrig.gpu_diagnostics %*
set "LAZYRIG_DIAGNOSTICS_EXIT=%ERRORLEVEL%"
pause
exit /b %LAZYRIG_DIAGNOSTICS_EXIT%
