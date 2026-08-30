@echo off
REM Move to the directory this .bat file is in (project root),
REM so it works no matter where it's launched from.
cd /d "%~dp0"

REM Activate the conda environment that has this project's dependencies installed.
REM Change "app-dev-py311" below if you rename the environment.
call conda activate app-dev-py311
if errorlevel 1 (
    echo Failed to activate conda environment "app-dev-py311".
    echo Make sure conda is installed and the environment exists.
    pause
    exit /b 1
)

python main.py
pause