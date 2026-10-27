@echo off
title Arcade Apprentice - reopen the page
rem Closed the browser by accident? This brings the page back. Training carries on regardless.
rem If the kit is still running it just reopens the page; if not, it starts the kit as normal.
powershell -NoProfile -Command "$c=New-Object Net.Sockets.TcpClient; try{$c.Connect('127.0.0.1',8795); exit 0}catch{exit 1}finally{$c.Close()}"
if %errorlevel%==0 (
  start "" "http://127.0.0.1:8795/"
) else (
  call "%~dp0START HERE.bat"
)
exit /b 0