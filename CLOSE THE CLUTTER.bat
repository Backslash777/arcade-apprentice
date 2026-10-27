@echo off
rem ===========================================================================
rem  CLOSE THE CLUTTER  -  Alfred, 21 Oct 2026 (v2, 22:30)
rem  Same as the page's "Free up this PC" button, without opening the page.
rem  Uses files\clutter.py and its list files\clutter_list.txt (edit that in Notepad).
rem  Apps you untick on the page are remembered and left alone here too.
rem  Add "dry" after the name to just list what it WOULD close.
rem ===========================================================================
cd /d "%~dp0files"
set PYTHONHOME=
set PYTHONPATH=
set "PY=runtime\python\python.exe"
if exist python-path.txt set /p PY=<python-path.txt
if not exist "%PY%" (
  echo Run START HERE.bat once first - the kit isn't set up on this PC yet.
  pause & exit /b 1
)
"%PY%" -B clutter.py %1
echo.
pause
