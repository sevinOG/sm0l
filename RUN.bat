@echo off
cd /d "%~dp0"
if exist "dist\sm0l\sm0l.exe" (
  start "" "dist\sm0l\sm0l.exe"
  exit /b 0
)
if not exist ".venv\Scripts\python.exe" (
  python -m venv .venv
  ".venv\Scripts\python.exe" -m pip install -r requirements.txt
)
".venv\Scripts\python.exe" sm0l.py
