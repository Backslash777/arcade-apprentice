@echo off
title Arcade Apprentice - close the kit
rem For when the page is already shut: stops training (each AI saves first), then closes everything the kit has open.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0files\close_kit.ps1" -Kit "%~dp0."
