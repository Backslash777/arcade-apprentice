@echo off
title Arcade Apprentice - the page's server (close this window to stop it)
cd /d "%~dp0files"
set "PY=runtime\python\python.exe"
if exist python-path.txt set /p PY=<python-path.txt
"%PY%" -B app.py
pause
