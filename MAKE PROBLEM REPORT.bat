@echo off
title Arcade Apprentice - problem report
cd /d "%~dp0files"
set PYTHONHOME=
set PYTHONPATH=
set "PY=runtime\python\python.exe"
if exist python-path.txt set /p PY=<python-path.txt
if not exist "%PY%" (
  echo  The AI libraries never finished installing, so send Jonny this file instead:  files\logs\setup.log
  pause & exit /b 1
)
echo  Making a problem report - about 30 seconds...
"%PY%" -B diagnose.py
pause
