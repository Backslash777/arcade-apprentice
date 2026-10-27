@echo off
title Arcade Apprentice - uninstall
rem Works from Windows' Temp folder, so this window isn't holding on to the kit's folder while it's being removed.
cd /d "%TEMP%"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0files\uninstall.ps1" -Kit "%~dp0." & exit /b
