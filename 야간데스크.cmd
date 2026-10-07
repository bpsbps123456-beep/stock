@echo off
cd /d "%~dp0"
py -m desk open
if errorlevel 1 pause
