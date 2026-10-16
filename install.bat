@echo off
setlocal
cd /d "%~dp0"
py -3.11 --version >nul 2>&1 || (echo Python 3.11 is needed: https://www.python.org/downloads/ & exit /b 1)
if not exist .venv py -3.11 -m venv .venv || exit /b 1
.venv\Scripts\python -m pip install --upgrade pip >nul
.venv\Scripts\python -m pip install -r requirements.txt || exit /b 1
.venv\Scripts\python tools\get_model.py || exit /b 1
echo Game data and icons ...
.venv\Scripts\python tools\fetch_assets.py || exit /b 1
echo MetaTFT data and reference files ...
.venv\Scripts\python tools\fetch_meta.py || exit /b 1
.venv\Scripts\python tools\build_reference.py || exit /b 1
echo.
echo Installed. Start a game with:  .venv\Scripts\python tools\play.py
