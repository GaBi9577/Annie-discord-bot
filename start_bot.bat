@echo off
REM Move to the directory this .bat file is in (project root),
REM so it works no matter where it's launched from.
cd /d "%~dp0"

REM Assumes python is available on PATH (e.g. a conda/venv env is already activated).
REM If you use a specific virtual environment, activate it manually before running this file.
python main.py
pause
