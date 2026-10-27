@echo off
title Arcade Apprentice - tune for this PC
rem Times the real trainer on this PC and keeps the fastest settings (files\benchmark.json). About 25 minutes.
rem Nothing else should be training while it runs. Run it again after new hardware, or if things feel slow.
cd /d "%~dp0files"
set PYTHONHOME=
set PYTHONPATH=
set "PY=runtime\python\python.exe"
if exist python-path.txt set /p PY=<python-path.txt
if not exist "%PY%" (
  echo.
  echo  The kit isn't set up on this PC yet - double-click START HERE.bat first, then come back to this.
  echo.
  pause & exit /b 1
)
echo.
echo  Tuning Arcade Apprentice for this PC - about 25 minutes. Leave it to it.
"%PY%" -u bench.py %*
echo.
pause
