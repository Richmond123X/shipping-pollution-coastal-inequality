@echo off
setlocal
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_policy_simulation_2024.ps1" %*
exit /b %ERRORLEVEL%
