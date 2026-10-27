@echo off
title Arcade Apprentice - keep this window open while using the page
cd /d "%~dp0files"
set PYTHONHOME=
set PYTHONPATH=
set "PY=runtime\python\python.exe"
if exist python-path.txt set /p PY=<python-path.txt
if not exist "%PY%" (
  powershell -NoProfile -ExecutionPolicy Bypass -File setup.ps1
  set "PY=runtime\python\python.exe"
  if exist python-path.txt set /p PY=<python-path.txt
)
if not exist "%PY%" (
  echo.
  echo  Setup did not finish. Check your internet connection and try again.
  echo  If it keeps failing, send Jonny this file:  files\logs\setup.log
  pause & exit /b 1
)
"%PY%" -B gpucheck.py
if errorlevel 3 (
  echo.
  echo  Setting up PyTorch for your graphics card - one-off download, a few minutes...
  powershell -NoProfile -ExecutionPolicy Bypass -File setup.ps1 -FixTorch
)
rem Start the page's little server WITHOUT a black window (pythonw), then this window closes.
rem "Close the kit" at the bottom of the page stops it again. Old way (with a window): START HERE - with window.bat
set "PYW=%PY:python.exe=pythonw.exe%"
if not exist "%PYW%" set "PYW=%PY%"
start "" "%PYW%" -B app.py
exit /b 0
